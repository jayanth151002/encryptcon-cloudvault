import os
import copy
import tqdm

import numpy as np
import pandas as pd
import swifter
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestClassifier

from IPython.display import clear_output

tqdm.tqdm.pandas()


class LoanDataset(nn.Module):
    device = "cpu"

    def __init__(
        self,
        df,
        label_column,
        amount_column,
        period_column,
        interest_column,
        embed_size=10,
    ):
        super().__init__()

        self.label_column = label_column
        self.amount_column = amount_column
        self.period_column = period_column
        self.interest_column = interest_column
        df = df.drop("Loan_ID", axis=1)
        X = df.drop(
            [label_column, amount_column, period_column, interest_column], axis=1
        )
        y1 = df[label_column]
        y2 = df[amount_column]
        y3 = df[period_column]
        y4 = df[interest_column]

        self.feat_encoder = {}
        for col in X.columns:
            if X[col].dtype == "object":
                if col not in self.feat_encoder:
                    self.feat_encoder[col] = {
                        "Label": LabelEncoder(),
                        "OneHot": OneHotEncoder(),
                    }
                    self.feat_encoder[col]["Label"].fit(X[col])
                    self.feat_encoder[col]["OneHot"].fit(
                        self.feat_encoder[col]["Label"].transform(X[col]).reshape(-1, 1)
                    )
                X[col] = X[col].swifter.apply(
                    lambda x: np.array(
                        self.feat_encoder[col]["OneHot"]
                        .transform(
                            self.feat_encoder[col]["Label"]
                            .transform([x])
                            .reshape(-1, 1)
                        )
                        .toarray()[0]
                    )
                )
            elif X[col].dtype == "int64" or X[col].dtype == "float64":
                if col not in self.feat_encoder:
                    self.feat_encoder[col] = {"MinMax": MinMaxScaler()}
                    self.feat_encoder[col]["MinMax"].fit(X[col].values.reshape(-1, 1))
                X[col] = self.feat_encoder[col]["MinMax"].transform(
                    X[col].values.reshape(-1, 1)
                )
            else:
                raise Exception("Unknown column type: {}".format(col))

        y1 = y1.swifter.apply(lambda x: 1 if x == "Y" else 0)
        y2 = y2.swifter.apply(
            lambda x: 0 if x < y2.max() // 3 else 1 if x < 2 * y2.max() // 3 else 2
        )

        self.label_encoder = {"y3": LabelEncoder(), "y4": LabelEncoder()}
        y3 = self.label_encoder["y3"].fit_transform(y3)
        y4 = self.label_encoder["y4"].fit_transform(y4)

        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4

        self.embed_size = embed_size
        self.encoder = {}
        for col in self.X.columns:
            try:
                self.encoder[col] = nn.Linear(
                    np.stack(self.X[col].values).shape[1], embed_size
                )
            except:
                self.encoder[col] = nn.Linear(
                    np.stack(self.X[col].values.reshape(-1, 1)).shape[1], embed_size
                )

        self.aggregator = nn.Sequential(
            nn.Linear(2 * embed_size, 4 * embed_size),
            nn.BatchNorm1d(4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
        )

        self.classifier_1 = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.BatchNorm1d(embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2),
            nn.ReLU(),
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.BatchNorm1d(embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 3),
            nn.ReLU(),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.BatchNorm1d(embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, self.label_encoder["y3"].classes_.shape[0]),
            nn.ReLU(),
        )
        self.classifier_4 = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.BatchNorm1d(embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, self.label_encoder["y4"].classes_.shape[0]),
            nn.ReLU(),
        )

        self.fit(epochs=100)

    def encode(self, df):
        X = df.copy()
        for col in X.columns:
            if col == self.label_column:
                y1 = X[col]
            elif col == self.amount_column:
                y2 = X[col]
            elif col == self.period_column:
                y3 = X[col]
            elif col == self.interest_column:
                y4 = X[col]
            elif X[col].dtype == "object":
                X[col] = X[col].swifter.apply(
                    lambda x: self.feat_encoder[col]["OneHot"]
                    .transform(
                        self.feat_encoder[col]["Label"].transform([x]).reshape(-1, 1)
                    )
                    .toarray()[0]
                )
            elif X[col].dtype == "int64" or X[col].dtype == "float64":
                X[col] = self.feat_encoder[col]["MinMax"].transform(
                    X[col].values.reshape(-1, 1)
                )
            else:
                raise Exception("Unknown column type: {}".format(col))

        X = X.drop(
            [
                self.label_column,
                self.amount_column,
                self.period_column,
                self.interest_column,
            ],
            axis=1,
        )
        y1 = y1.swifter.apply(lambda x: 1 if x == "Y" else 0)
        y2 = y2.swifter.apply(
            lambda x: 0 if x < y2.max() // 3 else 1 if x < 2 * y2.max() // 3 else 2
        )
        y3 = self.label_encoder["y3"].transform(y3)
        y4 = self.label_encoder["y4"].transform(y4)
        return X, y1, y2, y3, y4

    def forward(self, X):
        x = torch.empty((X.shape[0], 0, self.embed_size))
        for col in X.columns:
            try:
                out = self.encoder[col](
                    torch.tensor(np.stack(X[col].values), dtype=torch.float32).to(
                        self.device
                    )
                )
            except:
                out = self.encoder[col](
                    torch.tensor(
                        np.stack(X[col].values.reshape(-1, 1)), dtype=torch.float32
                    ).to(self.device)
                )
            if X.shape[0] == 1 and out.shape[0] != 1:
                out = out.unsqueeze(0)

            x = torch.cat((x, out.unsqueeze(1)), dim=1)

        x1 = torch.sum(x, dim=1)
        x2 = torch.einsum("ijk, ilk->ik", x, x)
        x = torch.cat((x1, x2), dim=1)
        x = self.aggregator(x)

        o1 = self.classifier_1(x + x1)
        o2 = self.classifier_2(x + x1)
        o3 = self.classifier_3(x + x1)
        o4 = self.classifier_4(x + x1)
        return o1, o2, o3, o4

    def fit(self, epochs):
        y = np.stack((self.y1, self.y2, self.y3, self.y4), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=0.2)
        y_test, y_train = torch.tensor(y_test).to(self.device), torch.tensor(
            y_train
        ).to(self.device)
        y1_train, y2_train, y3_train, y4_train = (
            y_train[:, 0],
            y_train[:, 1],
            y_train[:, 2],
            y_train[:, 3],
        )
        y1_test, y2_test, y3_test, y4_test = (
            y_test[:, 0],
            y_test[:, 1],
            y_test[:, 2],
            y_test[:, 3],
        )

        self.train()
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            o1, o2, o3, o4 = self.forward(X_train)

            l1 = criterion(o1, y1_train)

            o2, y2t = o2[y1_train == 1], y2_train[y1_train == 1]
            o3, y3t = o3[y1_train == 1], y3_train[y1_train == 1]
            o4, y4t = o4[y1_train == 1], y4_train[y1_train == 1]
            l2 = criterion(o2, y2t) + criterion(o3, y3t) + criterion(o4, y4t)

            loss = l1 + 0.4 * l2

            loss.backward()
            optimizer.step()
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
        self.eval()
        with torch.no_grad():
            o1, o2, o3, o4 = self.forward(X_test)

            l1 = criterion(o1, y1_test)

            o2, y2t = o2[y1_test == 1], y2_test[y1_test == 1]
            o3, y3t = o3[y1_test == 1], y3_test[y1_test == 1]
            o4, y4t = o4[y1_test == 1], y4_test[y1_test == 1]
            l2 = criterion(o2, y2t) + criterion(o3, y3t) + criterion(o4, y4t)

            loss = l1 + 0.4 * l2

            print("Test Loss: {}".format(loss.item()))
            y_pred = np.argmax(o1.cpu().numpy(), axis=1)
            print(
                "Test Accuracy: {}".format(
                    accuracy_score(y1_test.cpu().numpy(), y_pred)
                )
            )

    def predict(self, df):
        X, _, _, _, _ = self.encode(df)
        self.eval()
        with torch.no_grad():
            o1, o2, o3, o4 = self.forward(X)
            y1_pred = np.argmax(o1.cpu().numpy(), axis=1)
            y2_pred = np.argmax(o2.cpu().numpy(), axis=1)
            y3_pred = np.argmax(o3.cpu().numpy(), axis=1)
            y4_pred = np.argmax(o4.cpu().numpy(), axis=1)

        return (
            y1_pred,
            y2_pred,
            self.label_encoder["y3"].inverse_transform(y3_pred),
            self.label_encoder["y4"].inverse_transform(y4_pred),
        )

    def __get_item__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
