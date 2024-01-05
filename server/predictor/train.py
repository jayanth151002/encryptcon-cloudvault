from predictor.model import GraphAttnModel, load_lpa_subtensor
from predictor.config import Config
from predictor.data import data_engineer_benchmark, span_data_2d

import os
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

logger = logging.getLogger(__name__)


class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopper
        :param patience: the maximum number of rounds tolerated
        :param verbose: whether to stop early
        :param delta: the regularization factor
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        # self.val_preds = []
        # self.val_logits = []

    def earlystop(self, loss, model=None):  # , preds, logits):
        """
        :param loss: the loss score on validation set
        :param model: the model
        """
        value = -loss
        cv = loss
        # value = ap

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to("cpu")
            # self.val_preds = preds
            # self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print("EarlyStoper count: {:02d}".format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to("cpu")
            # self.val_preds = preds
            # self.val_logits = logits
            self.count = 0


def load_data(df, test_size):
    prefix = os.path.join(os.path.dirname(__file__), "data/")

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

    feat_data.to_csv(prefix + "S-FFSD_feat_data.csv", index=None)
    labels.to_csv(prefix + "S-FFSD_label_data.csv", index=None)

    index = list(range(len(labels)))
    g.ndata["label"] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata["feat"] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    graph_path = prefix + "graph-S-FFSD.bin"
    dgl.data.utils.save_graphs(graph_path, [g])

    train_idx, test_idx, _, _ = train_test_split(
        index,
        labels,
        stratify=labels,
        test_size=test_size / 2,
        random_state=0,
        shuffle=True,
    )

    return feat_data, labels, train_idx, test_idx, g


def main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, verbose=True):
    device = args.get("device", "cpu")
    graph = graph.to(device)
    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(
        n_splits=args["n_fold"], shuffle=True, random_state=args["seed"]
    )

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {
        col: torch.from_numpy(feat_df[col].values).long().to(device)
        for col in cat_features
    }

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(
        kfold.split(feat_df.iloc[train_idx], y_target)
    ):
        trn_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device)
        val_ind = torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        train_dataloader = NodeDataLoader(
            graph,
            trn_ind,
            train_sampler,
            device=device,
            use_ddp=False,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        val_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        val_dataloader = NodeDataLoader(
            graph,
            val_ind,
            val_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        model = GraphAttnModel(
            in_feats=feat_df.shape[1],
            hidden_dim=args["hid_dim"] // 4,
            n_classes=2,
            heads=[4] * args["n_layers"],
            activation=nn.PReLU(),
            n_layers=args["n_layers"],
            drop=args["dropout"],
            device=device,
            gated=args["gated"],
            ref_df=feat_df.iloc[train_idx],
            cat_features=cat_feat,
        ).to(device)

        lr = args["lr"] * np.sqrt(args["batch_size"] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args["wd"])
        lr_scheduler = MultiStepLR(
            optimizer=optimizer, milestones=[4000, 12000], gamma=0.3
        )

        earlystoper = early_stopper(patience=args["early_stopping"], verbose=True)
        start_epoch, max_epochs = 0, 2000
        for epoch in range(start_epoch, args["max_epochs"]):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                (
                    batch_inputs,
                    batch_work_inputs,
                    batch_labels,
                    lpa_labels,
                ) = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, input_nodes, device
                )

                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs
                )
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0 and verbose:
                    tr_batch_pred = (
                        torch.sum(
                            torch.argmax(train_batch_logits.clone().detach(), dim=1)
                            == batch_labels
                        )
                        / batch_labels.shape[0]
                    )
                    score = (
                        torch.softmax(train_batch_logits.clone().detach(), dim=1)[:, 1]
                        .cpu()
                        .numpy()
                    )

                    try:
                        print(
                            "In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, "
                            "train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}".format(
                                epoch,
                                step,
                                np.mean(train_loss_list),
                                average_precision_score(
                                    batch_labels.cpu().numpy(), score
                                ),
                                tr_batch_pred.detach(),
                                roc_auc_score(batch_labels.cpu().numpy(), score),
                            )
                        )
                    except:
                        pass

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    (
                        batch_inputs,
                        batch_work_inputs,
                        batch_labels,
                        lpa_labels,
                    ) = load_lpa_subtensor(
                        num_feat, cat_feat, labels, seeds, input_nodes, device
                    )

                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(
                        blocks, batch_inputs, lpa_labels, batch_work_inputs
                    )
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + loss_fn(
                        val_batch_logits, batch_labels
                    )
                    # val_all_list += 1
                    val_batch_pred = torch.sum(
                        torch.argmax(val_batch_logits, dim=1) == batch_labels
                    ) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * torch.tensor(
                        batch_labels.shape[0]
                    )
                    val_all_list = val_all_list + batch_labels.shape[0]

                    if step % 10 == 0 and verbose:
                        score = (
                            torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                                :, 1
                            ]
                            .cpu()
                            .numpy()
                        )
                        try:
                            print(
                                "In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, "
                                "val_acc:{:.4f}, val_auc:{:.4f}".format(
                                    epoch,
                                    step,
                                    val_loss_list / val_all_list,
                                    average_precision_score(
                                        batch_labels.cpu().numpy(), score
                                    ),
                                    val_batch_pred.detach(),
                                    roc_auc_score(batch_labels.cpu().numpy(), score),
                                )
                            )
                        except:
                            pass

            earlystoper.earlystop(val_loss_list / val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break

        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))

        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args["n_layers"])
        test_dataloader = NodeDataLoader(
            graph,
            test_ind,
            test_sampler,
            use_ddp=False,
            device=device,
            batch_size=args["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                (
                    batch_inputs,
                    batch_work_inputs,
                    batch_labels,
                    lpa_labels,
                ) = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, input_nodes, device
                )

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(
                    blocks, batch_inputs, lpa_labels, batch_work_inputs
                )
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(
                    torch.argmax(test_batch_logits, dim=1) == batch_labels
                ) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print("In test batch:{:04d}".format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(
        y_target, torch.softmax(oof_predictions, dim=1).cpu()[train_idx, 1]
    )
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = (
        earlystoper.best_model.to("cpu"),
        oof_predictions,
        test_predictions,
    )

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))

    torch.save(b_models.state_dict(), "checkpoints/model.pt")


if __name__ == "__main__":
    with open("config.yaml") as file:
        args = yaml.safe_load(file)

    prefix = os.path.join(os.path.dirname(__file__), "data/")
    cat_features = ["Target", "Location", "Type"]
    df = pd.read_csv(prefix + "S-FFSDneofull.csv")

    feat_data, labels, train_idx, test_idx, g = load_data(df, args["test_size"])
    main(feat_data, g, train_idx, test_idx, labels, args, cat_features)
