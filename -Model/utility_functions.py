import os
import pandas as pd
import numpy as np
from collections import Counter




##### Importing data #####
def read_from_multiple_csv(path, name_lst):
    df = pd.DataFrame()
    for name in name_lst:
        file_path = os.path.join(path, name)
        df = pd.concat([df, pd.read_csv(file_path)], ignore_index=True)
    return df


def file_name_lst(path, startswith=None, endswith=None):
    file_lst = os.listdir(path)
    if startswith:
        file_lst = [file for file in file_lst if file.startswith(startswith)]
    if endswith:
        file_lst = [file for file in file_lst if file.endswith(endswith)]
    return file_lst


##### List functions #####

def refine_lst(lst, exclude_lst):
    exclude_set = set(exclude_lst)
    return [x for x in lst if x not in exclude_set]





##### Math functions #####

def stretched_sigmoid(x, s=1):
    return 1 / (1 + np.exp(-x * s))

###### Dataframe functions ######

def columns_unique_items(df, column_name_lst):
    unique_items = []
    for column_name in column_name_lst:
        unique_items.extend(df[column_name].unique().tolist())
    return list(set(unique_items))

def columns_item_count(unique_pairs_df, w_cols=["wallet_a", "wallet_b"]):
    # Flatten the selected columns into a single list
    wallet_z = unique_pairs_df[w_cols].values.flatten()
    # Count occurrences of each wallet using Counter
    wallet_z_count = Counter(wallet_z)
    # Convert the Counter object into a DataFrame
    wallet_z_count_df = pd.DataFrame(
        wallet_z_count.items(), columns=["wallet", "interacted_wallets"]
    )
    # Return the resulting DataFrame
    return wallet_z_count_df


def columns_item_unique_pairs(
    raw_transaction_df, address_cols=["from_address", "to_address"]
):
    # Ensure input columns are present in the DataFrame
    if not all(col in raw_transaction_df.columns for col in address_cols):
        raise ValueError(f"Columns {address_cols} not found in the DataFrame.")

    # Convert the columns to strings to ensure consistent dtype
    raw_transaction_df[address_cols] = raw_transaction_df[address_cols].astype(str)
    # Create sorted pairs using numpy operations (faster than apply)
    pairs = np.sort(raw_transaction_df[address_cols].to_numpy(dtype=str), axis=1)
    # Deduplicate pairs using numpy.unique
    unique_pairs = np.unique(pairs, axis=0)
    # Create a DataFrame for unique pairs
    unique_pairs_df = pd.DataFrame(unique_pairs, columns=["wallet_a", "wallet_b"])

    return unique_pairs_df


def columns_item_unique_pair_counts(
    raw_transaction_df, address_cols=["from_address", "to_address"]
):
    unique_pairs_df = columns_item_unique_pairs(raw_transaction_df, address_cols)
    # Use the optimized columns_item_count function
    wallet_count = columns_item_count(unique_pairs_df, w_cols=["wallet_a", "wallet_b"])
    # Reset index before returning
    wallet_count.reset_index(drop=True, inplace=True)
    return wallet_count
