bsc_key = "V7JWKTWF9FUJRNWVKTA6IXD9ZQD8BSFUCE"
polygon_key = "H2C2HE9G2MZ41VVVGY5RKY61HVSEYNNP4B"
eth_key = "TS8E4YKWPPNP8JGT7U8K935NUYH3YKH1R9"

tgb_chains_tables = [
    ("bsc_token_transfers_full", True),
    ("ethereum_token_transfers_full", True),
    ("polygon_token_transfers_full", True),
    ("bsc_transactions_full", False),
    ("ethereum_transactions_full", False),
    ("polygon_transactions_full", False),
]

monthly_stats_tables = [
    "ethereum_address_monthly_stats_full",
    "bsc_address_monthly_stats_full",
    "polygon_address_monthly_stats_full",
]

last_stats_tables = [
    "ethereum_address_last_stats_full",
    "bsc_address_last_stats_full",
    "polygon_address_last_stats_full",
]