import random
import os
import sys
from math import isnan
import multiprocessing as mp
import time
from tqdm import tqdm
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import torch
import dgl
import networkx as nx


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")


def featmap_generator(df=None):
    time_span = [
        2,
        3,
        5,
        15,
        20,
        50,
        100,
        150,
        200,
        300,
        864,
        2590,
        5100,
        10000,
        24000,
    ]  # time windows to generate the characteristics
    time_name = [str(i) for i in time_span]
    time_list = df["Time"]
    post = []
    for trans_idx, trans_feat in tqdm(df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = time_list >= temp_time - length
            upbound = time_list <= temp_time
            correct_data = df[lowbound & upbound]
            new_df["trans_at_avg_{}".format(tname)] = correct_data["Amount"].mean()
            new_df["trans_at_totl_{}".format(tname)] = correct_data["Amount"].sum()
            new_df["trans_at_std_{}".format(tname)] = correct_data["Amount"].std()
            new_df["trans_at_bias_{}".format(tname)] = (
                temp_amt - correct_data["Amount"].mean()
            )
            new_df["trans_at_num_{}".format(tname)] = len(correct_data)
            new_df["trans_target_num_{}".format(tname)] = len(
                correct_data.Target.unique()
            )
            new_df["trans_location_num_{}".format(tname)] = len(
                correct_data.Location.unique()
            )
            new_df["trans_type_num_{}".format(tname)] = len(correct_data.Type.unique())
        post.append(new_df)
    return pd.DataFrame(post)


def sparse_to_adjlist(sp_matrix, filename):
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, "wb") as file:
        pickle.dump(adj_lists, file)
    file.close()


def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    return (data - mind) / (maxd - mind)


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1,
) -> torch.Tensor:
    """return indices of risk k-hop neighbors

    Args:
        graph (dgl.DGLGraph): dgl graph dataset
        center_idx (int): center node idx
        k (int): k-hop neighs
        where (str): {"predecessor", "successor"}
        risk_label (int, optional): value of fruad label. Defaults to 1.
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]

            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata["label"][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(graph: dgl.DGLGraph, risk_label: int = 1) -> torch.Tensor:
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata["label"][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map(graph, edge_feats):
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor(
            [
                edge_feats[neighs_1_of_center, 0].sum().item(),
                edge_feats[neighs_2_of_center, 0].sum().item(),
                edge_feats[neighs_1_of_center, 1].sum().item(),
                edge_feats[neighs_2_of_center, 1].sum().item(),
            ]
        )
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


def data_engineer_benchmark(feat_df):
    pool = mp.Pool(processes=4)
    args_all = [(card_n, card_df) for card_n, card_df in feat_df.groupby("Source")]
    jobs = [pool.apply_async(featmap_generator, args=args) for args in args_all]

    post_fe_df = []
    num_job = len(jobs)
    for i, job in enumerate(jobs):
        post_fe_df.append(job.get())
        sys.stdout.flush()
        sys.stdout.write("FE: {}/{}\r".format(i + 1, num_job))
        sys.stdout.flush()
    post_fe_df = pd.concat(post_fe_df)
    post_fe_df = post_fe_df.fillna(0.0)
    return post_fe_df


def calcu_trading_entropy(data_2):
    """calculate trading entropy of given data
    Args:
        data (pd.DataFrame): 2 cols, Amount and Type
    Returns:
        float: entropy
    """
    # if empty
    if len(data_2) == 0:
        return 0

    amounts = np.array(
        [
            data_2[data_2["Type"] == type]["Amount"].sum()
            for type in data_2["Type"].unique()
        ]
    )
    proportions = amounts / amounts.sum() if amounts.sum() else np.ones_like(amounts)
    ent = -np.array(
        [proportion * np.log(1e-5 + proportion) for proportion in proportions]
    ).sum()
    return ent


def span_data_2d(data, time_windows=[1, 3, 5, 10, 20, 50, 100, 500]):
    """transform transaction record into feature matrices

    Args:
        df (pd.DataFrame): transaction records
        time_windows (list): feature generating time length

    Returns:
        np.ndarray: (sample_num, |time_windows|, feat_num) transaction feature matrices
    """
    data = data[data["Labels"] != 2]

    nume_feature_ret, label_ret = [], []
    for row_idx in tqdm(range(len(data))):
        record = data.iloc[row_idx]
        acct_no = record["Source"]
        feature_of_one_record = []

        for time_span in time_windows:
            feature_of_one_timestamp = []
            prev_records = data.iloc[(row_idx - time_span) : row_idx, :]
            prev_and_now_records = data.iloc[(row_idx - time_span) : row_idx + 1, :]
            prev_records = prev_records[prev_records["Source"] == acct_no]

            feature_of_one_timestamp.append(prev_records["Amount"].sum() / time_span)
            feature_of_one_timestamp.append(prev_records["Amount"].sum())
            feature_of_one_timestamp.append(
                record["Amount"] - feature_of_one_timestamp[0]
            )
            feature_of_one_timestamp.append(len(prev_records))

            old_ent = calcu_trading_entropy(prev_records[["Amount", "Type"]])
            new_ent = calcu_trading_entropy(prev_and_now_records[["Amount", "Type"]])
            feature_of_one_timestamp.append(old_ent - new_ent)

            feature_of_one_record.append(feature_of_one_timestamp)

        nume_feature_ret.append(feature_of_one_record)
        label_ret.append(record["Labels"])

    nume_feature_ret = np.array(nume_feature_ret).transpose(0, 2, 1)

    return nume_feature_ret.astype(np.float32), np.array(label_ret).astype(np.int64)


if __name__ == "__main__":
    set_seed(0)

    data = pd.read_csv(os.path.join(DATADIR, "S-FFSD.csv"))
    data = featmap_generator(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, "S-FFSDneofull.csv"), index=None)
    data = pd.read_csv(os.path.join(DATADIR, "S-FFSDneofull.csv"))
