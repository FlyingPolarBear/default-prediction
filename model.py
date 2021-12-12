'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:55:26
LastEditors: Derry
LastEditTime: 2021-11-20 22:22:54
Description: Standard model file of a neural network
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


class MLP(torch.nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(args.n_in)
        self.bn1 = nn.BatchNorm1d(args.n_hid)
        self.hid1 = nn.Linear(args.n_in, args.n_hid)
        self.out = nn.Linear(args.n_hid, args.n_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.bn0(X)

        X = self.bn1(self.relu(self.hid1(X)))
        X = self.dropout(X)

        y_out = self.out(X)
        y_poss = self.softmax(y_out)
        return y_poss


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Dropout(p=0.5)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(p=0.5)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout(p=0.5)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout(p=0.5)
        )
        self.fn1 = torch.nn.Sequential(
            torch.nn.Linear(2*2*64, args.n_hid),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(args.n_hid),
            torch.nn.Dropout(p=0.5)
        )
        self.fn2 = torch.nn.Sequential(
            torch.nn.Linear(args.n_hid, args.n_out),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(args.n_out),
            torch.nn.Dropout(p=0.5)
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fn1(x.view(x.size(0), -1))
        y_out = self.fn2(x)
        y_poss = torch.nn.functional.softmax(y_out, dim=1)
        return y_poss


class ScaledDotProductAttention(nn.Module):
    # todo 模型不适应
    '''
    Scaled dot-product attention
    '''

    def __init__(self, args, h=3):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        d_model = 1
        d_k = args.n_in
        d_v = args.n_in

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=args.dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.bn0 = nn.BatchNorm1d(args.n_in)
        self.bn1 = nn.BatchNorm1d(args.n_hid)
        self.hid1 = nn.Linear(args.n_in, args.n_hid)
        self.out = nn.Linear(args.n_hid, args.n_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        x = x.unsqueeze(-1)
        queries = x
        keys = x
        values = x

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(
            0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(
            0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(
            0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(
            b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        out = out.squeeze(-1)
        X = self.bn0(out)
        X = self.bn1(self.relu(self.hid1(X)))
        X = self.dropout(X)
        y_out = self.out(X)
        y_poss = self.softmax(y_out)
        return y_poss


def ML_xgboost(X_train, y_train):
    from xgboost import XGBRegressor
    clf = XGBRegressor()
    clf.fit(X=X_train, y=y_train)
    return clf


def ML_lgbm(X_train, y_train):
    from lightgbm import LGBMRegressor
    clf = LGBMRegressor(n_estimators=200)
    clf.fit(X=X_train, y=y_train)
    clf.booster_.save_model('LGBMmode.txt')
    return clf


def ML_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor()
    clf = clf.fit(X_train, y_train)
    return clf


def ML_gbdt(X_train, y_train):
    from sklearn.ensemble import GradientBoostingRegressor
    clf = GradientBoostingRegressor()
    clf = clf.fit(X_train, y_train)
    return clf


def ML_svm(X_train, y_train):
    from sklearn.svm import SVR
    clf = SVR()
    clf = clf.fit(X_train, y_train)
    return clf


def ML_mlp(X_train, y_train):
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor(hidden_layer_sizes=(256, 128),
                       max_iter=200, early_stopping=True)
    clf = clf.fit(X_train, y_train)
    return clf
