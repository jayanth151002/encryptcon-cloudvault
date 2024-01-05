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


class CreditDataset(nn.Module):
    device = "cpu"

    def __init__(self, df, label_column, embed_size=32):
        super().__init__()

        self.label_column = label_column
        df = df.drop("ID", axis=1)
        X = df.drop(label_column, axis=1)
        y = df[label_column]

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

        y = y.swifter.apply(lambda x: 1 if x == "Y" else 0)

        self.X = X
        self.y = y

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

        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.BatchNorm1d(embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2, 2),
            nn.ReLU(),
        )

        self.fit(epochs=25)

    def encode(self, df):
        X = df.copy()
        for col in X.columns:
            if col == self.label_column:
                y = X[col]
                continue
            if X[col].dtype == "object":
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

        X = X.drop(self.label_column, axis=1)
        y = y.swifter.apply(lambda x: 1 if x == "Y" else 0)
        return X, y

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

        x = self.classifier(x + x1)
        return x

    def fit(self, epochs):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )
        y_test, y_train = torch.tensor(y_test.values).to(self.device), torch.tensor(
            y_train.values
        ).to(self.device)

        self.train()
        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
        self.eval()
        with torch.no_grad():
            output = self.forward(X_test)
            loss = criterion(output, y_test)
            print("Test Loss: {}".format(loss.item()))
            y_pred = np.argmax(output.cpu().numpy(), axis=1)
            print(
                "Test Accuracy: {}".format(accuracy_score(y_test.cpu().numpy(), y_pred))
            )

    def predict(self, dinput):
        df = pd.DataFrame([dinput])
        X, y = self.encode(df)
        self.eval()
        with torch.no_grad():
            output = self.forward(X)
            y_pred = np.argmax(output.cpu().numpy(), axis=1)
        return y_pred.item()

    def __get_item__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
