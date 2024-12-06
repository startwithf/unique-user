from trino.dbapi import connect
from trino.auth import BasicAuthentication
import pandas as pd
import datetime


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


def query_trino(w, trace_back_days=30):
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

    # , ts_interval as (
    #     SELECT
    #         transaction_hash,
    #         ts,
    #         cast(date_diff('hour', LAG(ts) OVER (ORDER BY ts), ts) as double) AS interval
    #     FROM
    #         latest_activity_tbl
    #     ORDER BY
    #         ts
    # )

    # select avg(interval) as mean_interval
    #     , max(interval) as max_interval
    #     , approx_percentile(interval, 0.5) AS median_interval
    #     , approx_percentile(interval, 0.25) AS one_fourth_interval
    #     , (SELECT COUNT(DISTINCT transaction_hash) FROM latest_activity_tbl) AS ttl_hash_count
    #     , (SELECT COUNT(DISTINCT the_other_wallet) FROM latest_activity_tbl) AS interacted_wallet_count
    # from ts_interval
