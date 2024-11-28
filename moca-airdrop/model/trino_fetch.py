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


def query_trino(w):
    query_string = f"""
    with latest_activity_tbl as (
        select transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "bsc_token_transfers_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '100' DAY AS timestamp)
        group by 1
        
        UNION ALL
        select transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "ethereum_token_transfers_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '100' DAY AS timestamp)
        group by 1
        
        UNION ALL
        select transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address  = '{w}' then to_address else from_address end) as the_other_wallet
        from "polygon_token_transfers_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '100' DAY AS timestamp)
        group by 1
        
        UNION ALL
        select hash as transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "bsc_transactions_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '100' DAY AS timestamp)
        and input = '0x'
        group by 1
        
        UNION ALL
        select hash as transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "ethereum_transactions_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '100' DAY AS timestamp)
        and input = '0x'
        group by 1
        
        UNION ALL
        select hash as transaction_hash
        , max(block_timestamp) as ts
        , max(case when from_address = '{w}' then to_address else from_address end) as the_other_wallet
        from "polygon_transactions_full"
        where (from_address = '{w}' or to_address = '{w}')
        and block_timestamp >= CAST(CURRENT_DATE - INTERVAL '100' DAY AS timestamp)
        and input = '0x'
        group by 1
    )
    
    select * from latest_activity_tbl
        """
    result = query_trino_admin(query_string)
    
    # sort result by timestamp
    result["ts"] = pd.to_datetime(result["ts"])
    result = result.sort_values(by="ts")
    
    unique_wallets = result["the_other_wallet"].nunique()
    unique_transactions = result["transaction_hash"].nunique()
    mean_interval = result["ts"].diff().mean()
    max_interval = result["ts"].diff().max()
    median_interval = result["ts"].diff().median()
    one_fourth_interval = result["ts"].diff().quantile(0.25)
    
    result_dict = {
        "unique_wallets": unique_wallets,
        "unique_transactions": unique_transactions,
        "mean_interval": mean_interval,
        "max_interval": max_interval,
        "median_interval": median_interval,
        "one_fourth_interval": one_fourth_interval
    }
    
    return result_dict, result




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