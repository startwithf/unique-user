import requests
import re
from concurrent.futures import ThreadPoolExecutor
from parameters import bsc_key, polygon_key, eth_key


def check_contract(
    wallet_address, bsc_key_=bsc_key, polygon_key_=polygon_key, eth_key_=eth_key
):
    bsc_contract_link = f"https://api.bscscan.com/api?module=contract&action=getabi&address={wallet_address}&apikey={bsc_key_}"
    polygon_contract_link = f"https://api.polygonscan.com/api?module=contract&action=getabi&address={wallet_address}&apikey={polygon_key_}"
    eth_contract_link = f"https://api.etherscan.io/v2/api?chainid=1&module=contract&action=getabi&address={wallet_address}&apikey={eth_key_}"

    endpoint_lst = [eth_contract_link, bsc_contract_link, polygon_contract_link]
    for endpoint in endpoint_lst:
        response = requests.get(endpoint).json()
        if response["status"] == "1":
            print(
                f'{wallet_address} is a contract on {re.findall(r"https://api\.([a-z]+)\.", endpoint)[0]}'
            )
            return wallet_address

    return None


def check_contract_wallet_addresses(wallet_addresses, max_threads=10):
    contract_wallets = []
    with ThreadPoolExecutor(max_threads) as executor:
        # Submit tasks for concurrent execution
        futures = [
            executor.submit(check_contract, wallet) for wallet in wallet_addresses
        ]
        for future in futures:
            results = future.result()
            if results:
                contract_wallets.append(results)
    return contract_wallets