import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from cdlib import algorithms
import plotly.graph_objects as go
from tqdm import tqdm
import random


def read_from_multiple_excels(path, name_lst):
    df = pd.DataFrame()
    for name in name_lst:
        file_path = os.path.join(path, name)
        df = pd.concat([df, pd.read_excel(file_path)], ignore_index=True)
    return df


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


def check_df(df):
    display(df.shape)
    display(df.dtypes)
    display(df.describe())
    display(df.head())


def remove_contract_transactions(df, col_lst, contract_lst):
    for col in col_lst:
        df = df[~df[col].isin(contract_lst)]
    return df


def normalize_pair(w1, w2):
    return tuple(sorted([w1, w2]))


def count_pairs(df, w1_col, w2_col):
    pairs = df[[w1_col, w2_col]].apply(
        lambda x: normalize_pair(x[w1_col], x[w2_col]), axis=1
    )
    return pairs.value_counts()


######################  weight Related Functions  ######################


def get_weight_df(template_df):
    # Copy the merged pair dataframe
    weight_df = template_df.copy()

    # Get wallets
    weight_df.reset_index(inplace=True)
    weight_df.rename(columns={"index": "pair"}, inplace=True)
    weight_df["wallet_a"] = weight_df["pair"].apply(lambda x: x[0])
    weight_df["wallet_b"] = weight_df["pair"].apply(lambda x: x[1])

    return weight_df


def plot_weight_vs_count(df):
    plt.figure(figsize=(5, 3))
    plt.scatter(
        df["pure_transfer_count"], df["trade_count"], c=df["weight"], cmap="viridis"
    )
    plt.xlabel("pure_transfer_count")
    plt.ylabel("trade_count")
    plt.title("trade_count vs pure_transfer_count")
    plt.colorbar()
    plt.show()


def plot_weight_dist(df, figsize=(15, 3), bins=100):
    plt.figure(figsize=figsize)
    plt.hist(df["weight"], bins=bins)
    plt.xlabel("weight")
    plt.ylabel("count")
    plt.title("weight distribution")
    plt.show()


def plot_weight_cumulative_dist(df, figsize=(15, 3), bins=100):
    plt.figure(figsize=(5, 3))
    plt.hist(df["weight"], bins=100, cumulative=True, density=True)
    plt.xlabel("weight")
    plt.ylabel("count")
    plt.title("cumulative distribution of weight")
    plt.show()


def stretched_sigmoid(x, s=1):
    return 1 / (1 + np.exp(-x * s))


######################  Community Related Functions  ######################


def create_community(df, method="surprise", resolution=1):
    print("Method:", method)
    num_unique_wallets = len(set(df["wallet_a"]).union(set(df["wallet_b"])))
    num_edges = df.shape[0]
    print(f"Number of unique wallets: {num_unique_wallets}")
    print(f"Number of edges: {num_edges}")

    # Create a graph
    G = nx.Graph()

    # Add edges to the graph
    for _, row in df.iterrows():
        G.add_edge(row["wallet_a"], row["wallet_b"], weight=row["weight"])

    # Detect communities
    if method == "louvain":
        communities = algorithms.louvain(G, weight="weight", resolution=resolution)
    elif method == "surprise":
        communities = algorithms.surprise_communities(G, weights="weight")
    elif method == "leiden":
        communities = algorithms.leiden(G, weights="weight")

    communities_list = communities.communities
    # communities_list = sorted(communities_list, key=lambda x: x[0])
    print(f"Number of communities detected: {len(communities_list)}")
    print("-")
    print(
        f"Average community size: {np.mean([len(community) for community in communities_list])}"
    )
    print(
        f"Max community size: {np.max([len(community) for community in communities_list])}"
    )
    print(
        f"Min community size: {np.min([len(community) for community in communities_list])}"
    )

    print("")

    communities_list = sorted(communities_list, key=lambda x: len(x), reverse=True)

    return communities_list


def community_visualization(df):
    # Create a directed graph
    Gt = nx.from_pandas_edgelist(
        df,
        source="wallet_a",
        target="wallet_b",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )

    # Generate 3D positions
    pos = nx.spring_layout(Gt, dim=3)

    # Extract node and edge positions
    edge_x = []
    edge_y = []
    edge_z = []
    edge_text = []  # List to hold weight information for hover
    for edge in Gt.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
        edge_text.append(f"weight: {edge[2]['weight']}")

    node_x = [pos[node][0] for node in Gt.nodes()]
    node_y = [pos[node][1] for node in Gt.nodes()]
    node_z = [pos[node][2] for node in Gt.nodes()]

    # Create Plotly figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="black", width=5),
            text=edge_text,
            hoverinfo="text",
        )
    )

    # Add nodes
    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker=dict(size=5, color="blue"),
            text=list(Gt.nodes()),  # Show node labels on hover
            hoverinfo="text",
            textfont=dict(size=5),
            # hoverinfo="none",
        )
    )

    axis_setting = dict(
        showbackground=True,
        titlefont=dict(size=10),
        tickfont=dict(size=10),
    )
    # Update layout
    fig.update_layout(
        showlegend=False,
        width=800,
        height=800,
        scene=dict(
            xaxis=axis_setting,
            yaxis=axis_setting,
            zaxis=axis_setting,
        ),
    )

    # Show the plot
    fig.show()


def check_overlap_lst(lst1, lst2):
    return list(set(lst1) & set(lst2))


def find_main_wallet(commu_lst, the_weight_df):
    # Filter the DataFrame once for relevant wallets
    filtered_df = the_weight_df[
        the_weight_df["wallet_a"].isin(commu_lst) | the_weight_df["wallet_b"].isin(commu_lst)
    ]
    # Calculate total transaction volume for each wallet
    check_wallet_a = filtered_df['wallet_a'].value_counts()
    check_wallet_b = filtered_df['wallet_b'].value_counts()
    check_wallet = check_wallet_a.add(check_wallet_b, fill_value=0)
    check_wallet = check_wallet.sort_values(ascending=False)
    return check_wallet


# for each community, check how many wallets are sybil wallets
def check_sybil_community(community_full_lst, sybil_lst):
    community_sybil_condition = {}
    for i in tqdm(range(len(community_full_lst))):
        community_sybil_condition[i] = {}
        community_sybil_condition[i]["community"] = community_full_lst[i]
        community_sybil_condition[i]["community_size"] = len(community_full_lst[i])
        community_sybil_condition[i]["sybil_wallets"] = check_overlap_lst(
            community_full_lst[i], sybil_lst
        )
        community_sybil_condition[i]["sybil_wallets_size"] = len(
            community_sybil_condition[i]["sybil_wallets"]
        )
        community_sybil_condition[i]["sybil_ratio"] = (
            community_sybil_condition[i]["sybil_wallets_size"]
            / community_sybil_condition[i]["community_size"]
        )
    return community_sybil_condition


def check_sybil_avg_ratio(community_sybil_condition, ratio=0):
    return [
        community_sybil_condition[i]["sybil_ratio"]
        for i in range(len(community_sybil_condition) - 1)
        if community_sybil_condition[i]["sybil_ratio"] > ratio
    ]


def random_rate(num_1, num_2, full_lst, trials=100):
    similarity_lst = []
    for i in range(trials):
        sample_1 = random.sample(full_lst, num_1)
        sample_2 = random.sample(full_lst, num_2)
        common_elements = set(sample_1).intersection(set(sample_2))
        similarity = len(common_elements) / min(len(sample_1), len(sample_2))
        similarity_lst.append(similarity)

    return np.mean(similarity_lst)


def find_transfer_for_wallet(wallet, transfer_df, and_or="or"):
    if and_or == "or":
        return transfer_df[
            (transfer_df["wallet_a"] == wallet) | (transfer_df["wallet_b"] == wallet)
        ]
    else:
        return transfer_df[
            (transfer_df["wallet_a"] == wallet) & (transfer_df["wallet_b"] == wallet)
        ]


def find_commu_for_wallet(wallet, community_lst):
    for i, community in enumerate(community_lst):
        if wallet in community:
            return i
    return None


def uncommon_wallets(lst_1, lst_2):
    # find wallets in lst1 but not in lst2
    diff = set(lst_1).difference(set(lst_2))
    return list(diff)


def filter_community_lst(community_lst, wallet_lst):
    wallet_set = set(wallet_lst)
    return [
        [item for item in sublist if item in wallet_set] for sublist in community_lst
    ]


def expand_community_lst(community_lst, wallet_lst):
    # Create a set of all items in community_lst for O(1) lookups
    community_set = {item for sublist in community_lst for item in sublist}
    # Extend with wallets not in any community
    community_lst.extend([[item] for item in wallet_lst if item not in community_set])
    return community_lst
