from predictor.model import GraphAttnModel, load_lpa_subtensor
from predictor.train import load_data
from predictor.config import Config
from predictor.data import data_engineer_benchmark, span_data_2d

import os
import tqdm
import copy
import pickle
import yaml
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd

from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader as NodeDataLoader


class Test:
    device = "cpu"

    def __init__(self):
        with open("config.yaml") as file:
            self.args = yaml.safe_load(file)

        self.prefix = os.path.join(os.path.dirname(__file__), "data/")
        cat_feat = ["Target", "Location", "Type"]
        self.df = pd.read_csv(self.prefix + "S-FFSDneofull.csv")

        feat_df, _, train_idx, _, _ = load_data(self.df, 0.1)

        self.model = model = GraphAttnModel(
            in_feats=feat_df.shape[1],
            hidden_dim=self.args["hid_dim"] // 4,
            n_classes=2,
            heads=[4] * self.args["n_layers"],
            activation=nn.PReLU(),
            n_layers=self.args["n_layers"],
            drop=self.args["dropout"],
            device=self.device,
            gated=self.args["gated"],
            ref_df=feat_df.iloc[train_idx],
            cat_features=cat_feat,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(__file__), "checkpoints/") + "model.pt"
            ),
            strict=False,
        )

    def add_entry(self, entry):
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
        ]
        time_name = [str(i) for i in time_span]
        time_list = self.df["Time"]
        for trans_idx, trans_feat in entry.iterrows():
            new_df = pd.Series(trans_feat)
            temp_time = new_df.Time
            temp_amt = new_df.Amount
            for length, tname in zip(time_span, time_name):
                lowbound = time_list >= temp_time - length
                upbound = time_list <= temp_time
                correct_data = self.df[lowbound & upbound]
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
                new_df["trans_type_num_{}".format(tname)] = len(
                    correct_data.Type.unique()
                )
        post = pd.DataFrame([new_df])
        post = pd.concat([self.df, post], axis=0)
        return post

    def update_data(self, df):
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]
        data = df[df["Labels"] <= 2]
        data = data.reset_index(drop=True)
        out = []
        alls = []
        allt = []
        pair = ["Source", "Target", "Location", "Type"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend(
                    [
                        sorted_idxs[i]
                        for i in range(df_len)
                        for j in range(edge_per_trans)
                        if i + j < df_len
                    ]
                )
                tgt.extend(
                    [
                        sorted_idxs[i + j]
                        for i in range(df_len)
                        for j in range(edge_per_trans)
                        if i + j < df_len
                    ]
                )
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        cal_list = ["Source", "Target", "Location", "Type"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        feat_data = data.drop("Labels", axis=1)
        labels = data["Labels"]

        feat_data.to_csv(self.prefix + "S-FFSD_feat_data.csv", index=None)
        labels.to_csv(self.prefix + "S-FFSD_label_data.csv", index=None)

        index = list(range(len(labels)))
        g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
        graph_path = self.prefix + "graph-S-FFSD.bin"
        dgl.data.utils.save_graphs(graph_path, [g])

        test_idx = [len(labels) - 1]
        # test_idx = labels==1
        # test_idx = np.where(test_idx)[0]
        # test_idx = torch.from_numpy(test_idx).long()

        return feat_data, labels, test_idx, g

    def test(self, entry):
        df = self.add_entry(entry)
        df.to_csv(self.prefix + "S-FFSDneofull_added.csv", index=None)
        cat_features = ["Target", "Location", "Type"]
        feat_df, labels, test_idx, graph = self.update_data(df)

        num_feat = torch.from_numpy(feat_df.values).float().to(self.device)
        cat_feat = {
            col: torch.from_numpy(feat_df[col].values).long().to(self.device)
            for col in cat_features
        }
        labels = torch.from_numpy(labels.values).long().to(self.device)

        graph = graph.to(self.device)
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(self.device)
        test_sampler = MultiLayerFullNeighborSampler(self.args["n_layers"])
        test_dataloader = NodeDataLoader(
            graph,
            test_ind,
            test_sampler,
            use_ddp=False,
            device=self.device,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        self.model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                (
                    batch_inputs,
                    batch_work_inputs,
                    batch_labels,
                    lpa_labels,
                ) = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, input_nodes, self.device
                )

                blocks = [block.to(self.device) for block in blocks]
                test_batch_logits = self.model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs
                )
                test_batch_pred = torch.argmax(test_batch_logits, dim=1)
                return test_batch_pred.item()

    def query(self, input: dict):
        entry = pd.DataFrame([input])
        return self.test(entry)
