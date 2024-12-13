import importlib
import parameters

importlib.reload(parameters)

from trino.dbapi import connect
from trino.auth import BasicAuthentication
import pandas as pd
import datetime
from parameters import tgb_chains_tables, monthly_stats_tables, last_stats_tables
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from itertools import repeat

###### account information ######


def query_trino_admin(query_string: str, is_prod: bool = True):
    cfg = {
        "host": (
            "trino.telegraphbay.app" if is_prod else "trino.staging.telegraphbay.app"
        ),
        "port": "443",
        "user": "flora",
        "password": "vmt-qyh*fxu*ufb2HFW",
        "catalog": "iceberg",
        "schema": "animoca",
    }
    conn = connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        auth=BasicAuthentication(cfg["user"], cfg["password"]),
        http_scheme="https",
        catalog=cfg["catalog"],
        schema=cfg["schema"],
    )
    cur = conn.cursor()
    cur.execute(query_string)
    columns = []
    for i in range(len(cur.description)):
        columns.append(cur.description[i][0])
    try:
        rows = cur.fetchall()
        result = pd.DataFrame(rows, columns=columns)
        cur.close()
        conn.close()
        # print("Query successful")
        return result
    except Exception as e:
        print(e)
        return None


###### check hot wallets (based on transaction frequency) ######


def query_wallet_transaction_counts(
    table_name, w, start_date_, end_date_, is_token_transfer
):
    input_condition = "" if is_token_transfer else "and input = '0x'"
    query = f"""
    select 
        count(distinct {"transaction_hash" if is_token_transfer else "hash"})
    from "{table_name}"
    where 
        (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= date '{start_date_}'
        and block_timestamp <= date '{end_date_}'
        {input_condition}
    """
    return query_trino_admin(query).values[0][0]


def wallet_transaction_count(w, start_date_, end_date_, tables_):
    start_date_ = datetime.datetime.strptime(start_date_, "%Y-%m-%d").strftime(
        "%Y-%m-%d"
    )
    end_date_ = datetime.datetime.strptime(end_date_, "%Y-%m-%d").strftime("%Y-%m-%d")
    if end_date_ <= start_date_:
        print("Invalid date range.")
        return 0

    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda t: query_wallet_transaction_counts(
                t[0], w, start_date_, end_date_, t[1]
            ),
            tables_,
        )

    return sum(results)


def check_frequent_wallets(
    w_lst, start_date_, end_date_, f_threshold=1000, tables_=tgb_chains_tables
):
    # Helper function to check if a wallet is frequent
    def is_frequent(wallet):
        return (
            wallet
            if wallet_transaction_count(wallet, start_date_, end_date_, tables_)
            > f_threshold
            else None
        )

    # Return early if the wallet list is empty
    if not w_lst:
        return []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        frequent_wallets = list(filter(None, executor.map(is_frequent, w_lst)))

    return frequent_wallets


###### get clustering variables ######

## monthly stats ##


def query_wallet_monthly_info(table_name, w_lst, start_date_, end_date_):
    query_wallet_filter = ", ".join([f"'{w}'" for w in w_lst])
    if start_date_ or end_date_:
        conditions = []
        if start_date_:
            conditions.append(f"last_txn_time >= date '{start_date_}'")
        if end_date_:
            conditions.append(f"last_txn_time <= date '{end_date_}'")
        date_filter_clause = "and " + " and ".join(conditions)
    else:
        date_filter_clause = ""
    query = f"""
    select 
        address
        , coalesce(sum(cardinality(active_date_list)), 0) as active_days
        , coalesce(sum(total_gas_fee_in_usd), 0) as gas_fee 
        , coalesce(sum(number_of_txn), 0) as total_txn
        , coalesce(count(distinct interact_address), 0) as unique_tokens
        , max(date(last_txn_time)) as last_txn_date
    from "{table_name}"
    where 
        address IN ({query_wallet_filter})
        {date_filter_clause}
    group by 1
    """
    return query_trino_admin(query).values


def wallets_monthly_info(w_lst, start_date_, end_date_, tables_=monthly_stats_tables):
    # Validate date range
    if start_date_ and end_date_ and end_date_ <= start_date_:
        print("Invalid date range.")
        return None

    # Format dates
    if start_date_:
        start_date_ = datetime.datetime.strptime(start_date_, "%Y-%m-%d").strftime(
            "%Y-%m-%d"
        )
    if end_date_:
        end_date_ = datetime.datetime.strptime(end_date_, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Helper function to aggregate results
    def aggregate_results(results):
        aggregated = defaultdict(
            lambda: [0, 0, 0, 0, 0, None, None, None]
        )
        for result in results:
            for row in result:
                wallet = row[0]  # Wallet address
                active_days = row[1]
                gas_fee = row[2]
                total_txn = row[3]
                unique_interacted_wallets = row[4]
                last_txn_date = datetime.datetime.strptime(row[5], "%Y-%m-%d")

                # Increment aggregated values for the wallet
                aggregated[wallet][0] += active_days
                aggregated[wallet][1] += gas_fee
                aggregated[wallet][2] += total_txn
                aggregated[wallet][3] += unique_interacted_wallets
                if (
                    aggregated[wallet][5] is None
                    or last_txn_date > aggregated[wallet][5]
                ):
                    aggregated[wallet][5] = last_txn_date
                aggregated[wallet][4] = last_txn_date.toordinal()
                aggregated[wallet][6] = start_date_
                aggregated[wallet][7] = end_date_
        return dict(aggregated)  # Convert back to a regular dictionary if desired

    # Execute queries in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda t: query_wallet_monthly_info(t, w_lst, start_date_, end_date_),
            tables_,
        )
        results = list(results)  # Ensure all tasks are completed

    # Aggregate results across tables
    aggregated_results = aggregate_results(results)
    return aggregated_results


## last_stats_full ##

def query_wallet_last_info(table_name, w_lst, start_date_, end_date_):
    query_wallet_filter = ", ".join([f"'{w}'" for w in w_lst])
    if start_date_ or end_date_:
        conditions = []
        if start_date_:
            conditions.append(f"last_txn_time >= date '{start_date_}'")
        if end_date_:
            conditions.append(f"last_txn_time <= date '{end_date_}'")
        date_filter_clause = "and " + " and ".join(conditions)
    else:
        date_filter_clause = ""
        
    query = f"""
    select 
        address
        , max(date(last_txn_time)) as last_txn_date
        , count(distinct last_txn_token_address) as unique_tokens
    from "{table_name}"
    where 
        address IN ({query_wallet_filter})
        {date_filter_clause}
    group by 1
    """
    return query_trino_admin(query).values


def wallets_last_info(w_lst, start_date_ = None, end_date_ = None, tables_=last_stats_tables):

    # Validate date range
    if start_date_ and end_date_ and end_date_ <= start_date_:
        print("Invalid date range.")
        return None

    # Format dates
    if start_date_:
        start_date_ = datetime.datetime.strptime(start_date_, "%Y-%m-%d").strftime(
            "%Y-%m-%d"
        )
    if end_date_:
        end_date_ = datetime.datetime.strptime(end_date_, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    
    # Helper function to aggregate results
    def aggregate_results(results):
        aggregated = defaultdict(lambda: [None, 0])
        for result in results:
            for row in result:
                wallet = row[0]  # Wallet address
                last_txn_date = datetime.datetime.strptime(row[1], "%Y-%m-%d")  # Parse date
                unique_tokens = row[2]

                # Increment aggregated values for the wallet
                if (
                    aggregated[wallet][0] is None
                    or last_txn_date > aggregated[wallet][0]
                ):
                    aggregated[wallet][0] = last_txn_date
                aggregated[wallet][1] += unique_tokens
        
        return dict(aggregated)  # Convert back to a regular dictionary if desired

    # Execute queries in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda t: query_wallet_last_info(t, w_lst, start_date_, end_date_),
            tables_,
        )
        results = list(results)  # Ensure all tasks are completed

    # Aggregate results across tables
    aggregated_results = aggregate_results(results)
    return aggregated_results


##### Playground #####
def query_trino(w, trace_back_days):
    trace_back_days = int(trace_back_days)
    if trace_back_days <= 0:
        result_dict = {
            "unique_interacted_wallets": None,
            "unique_transactions": None,
            "days_with_transactions": None,
            "mean_interval": None,
            "max_interval": None,
            "median_interval": None,
            "one_fourth_interval": None,
        }
        result_df = pd.DataFrame()
        return result_dict, result_df

    query_string = f"""
    with latest_activity_tbl as (
        select transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "bsc_token_transfers_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '{trace_back_days}' DAY AS timestamp)
        group by 1
        
        UNION ALL
        select transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "ethereum_token_transfers_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '{trace_back_days}' DAY AS timestamp)
        group by 1
        
        UNION ALL
        select transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address  = '{w}' then to_address else from_address end) as the_other_wallet
        from "polygon_token_transfers_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '{trace_back_days}' DAY AS timestamp)
        group by 1
        
        UNION ALL
        select hash as transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "bsc_transactions_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '{trace_back_days}' DAY AS timestamp)
        and input = '0x'
        group by 1
        
        UNION ALL
        select hash as transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "ethereum_transactions_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '{trace_back_days}' DAY AS timestamp)
        and input = '0x'
        group by 1
        
        UNION ALL
        select hash as transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "polygon_transactions_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '{trace_back_days}' DAY AS timestamp)
        and input = '0x'
        group by 1
    )
    
    select * from latest_activity_tbl
        """
    result_df = query_trino_admin(query_string)

    # sort result by timestamp
    result_df["ts"] = pd.to_datetime(result_df["ts"])
    result_df = result_df.sort_values(by="ts")

    unique_wallets = result_df["the_other_wallet"].nunique()
    unique_transactions = result_df["transaction_hash"].nunique()
    mean_interval = str(result_df["ts"].diff().mean())
    max_interval = str(result_df["ts"].diff().max())
    median_interval = str(result_df["ts"].diff().median())
    one_fourth_interval = str(result_df["ts"].diff().quantile(0.25))
    min_interval = str(result_df["ts"].diff().min())
    days_with_transactions = result_df["ts"].dt.date.nunique()

    result_dict = {
        "days_with_transactions": days_with_transactions,
        "unique_interacted_wallets": unique_wallets,
        "unique_transactions": unique_transactions,
        "mean_interval": mean_interval,
        "max_interval": max_interval,
        "median_interval": median_interval,
        "one_fourth_interval": one_fourth_interval,
        "min_interval": min_interval,
    }

    return result_dict, result_df
