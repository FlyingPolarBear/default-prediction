'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-27 17:05:23
LastEditors: Derry
LastEditTime: 2021-12-12 19:41:29
Description: Utils file
'''
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def load_data():  # 加载数据
    train_bank = pd.read_csv('./data/train_public.csv')
    train_internet = pd.read_csv('./data/train_internet.csv')
    test = pd.read_csv('./data/test_public.csv')
    return train_bank, train_internet, test


def prepare_data(train_bank, train_internet, test):  # 数据预处理
    def data_cleaning(data):  # 数据清洗
        # 去除无用特征
        data = data.drop(['earlies_credit_mon', 'loan_id',
                          'user_id'], axis=1, inplace=False)
        # 离散特征issue_date的连续化
        data['issue_date'] = pd.to_datetime(data['issue_date'])
        base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
        data['issue_date_y'] = data['issue_date'].dt.year
        data['issue_date_m'] = data['issue_date'].dt.month
        data['issue_date_diff'] = (data['issue_date']-base_time).dt.days
        data.drop('issue_date', axis=1, inplace=True)
        # 离散特征work_year的连续化
        work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
                         '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
        data['work_year'].fillna('10+ years', inplace=True)
        data['work_year'] = data['work_year'].map(work_year_map)
        # 离散特征class的连续化
        class_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        data['class'] = data['class'].map(class_map)
        # 离散特征industry的连续化
        with open('data/work_feature.csv', encoding='utf-8') as f:
            industry_map = {}
            for line in f:
                work_fea = line.strip().split(',')
                industry_map[work_fea[0]] = int(work_fea[1])
        data['industry'] = data['industry'].map(industry_map)
        # 使用均值填充na
        na_cols = data.columns[data.isna().any()]
        for na_col in na_cols:
            data[na_col].fillna(data[na_col].mean(), inplace=True)
        # 使用onehot编码其他离散特征
        data = pd.get_dummies(data)
        return data

    def sampling(data, mode="down"):  # 使用采样的方法平衡正样本和负样本的数量
        data_pos = data[data['isDefault'] == 1]
        data_neg = data[data['isDefault'] == 0]
        pn_frac = data_pos.shape[0]/data_neg.shape[0]
        if mode == "down":  # 下采样
            if pn_frac > 1:
                data_pos = data_pos.sample(frac=1/pn_frac)
            else:
                data_neg = data_neg.sample(frac=pn_frac)
        elif mode == "upper":  # 上采样
            if pn_frac > 1:
                data_neg = data_neg.sample(replace=True, frac=pn_frac)
            else:
                data_pos = data_pos.sample(replace=True, frac=1/pn_frac)
        data = pd.concat([data_pos, data_neg])
        data = data.sample(frac=1)
        return data

    # 统一label列名称为isDefault
    train_internet = train_internet.rename(columns={'is_default': 'isDefault'})

    # 合并train_bank和train_internet
    common_cols = [
        col for col in train_bank.columns if col in train_internet.columns]
    train_data = pd.concat(
        [train_internet[common_cols], train_bank[common_cols]])

    # 使用采样的方法平衡正负样本
    train_data = sampling(train_data, "down")
    train_data = data_cleaning(train_data)

    # 划分训练集和测试集
    X_train = np.array(train_data.drop(['isDefault'], axis=1, inplace=False))
    y_train = np.array(train_data['isDefault'])
    test_data = test[common_cols[:-1]]
    test_data = data_cleaning(test_data)
    X_test = np.array(test_data)

    return X_train, y_train, X_test


def evaluate(model, X, y, loss_fun):
    model.eval()
    y_poss = model(X)
    y_pred = y_poss[:, 1]
    loss = loss_fun(y_pred, y)
    return loss.item(), y_pred


def ml_valid(X_train, y_train, ML_model):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, random_state=123, test_size=0.2)
    model = ML_model(X_tr, y_tr)
    y_pred = model.predict(X_val)
    # 计算auc
    auc = roc_auc_score(y_val, y_pred)
    print("auc: %.4f" % auc)


def ml_predict(X_train, y_train, ML_model, X_test):
    model = ML_model(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def plot(loss, AUC, args):
    sns.set()
    plt.figure('loss')
    plt.title("loss of each epoch")
    plt.xlim(0, args.epoch)
    plt.plot(loss, color='b', label='loss (min=%0.4f)' % min(loss))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig(args.tmp_path+'/loss.png')

    plt.figure('auc')
    plt.title("auc of each epoch")
    plt.xlim(0, args.epoch)
    plt.ylim(0.5, 1)
    plt.plot(AUC, color='b', label='auc (max=%0.4f)' % max(AUC))
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.legend(loc="lower right")
    plt.savefig(args.tmp_path+'/acc.png')

    plt.close('all')


def plot_roc(fpr, tpr, args):
    roc_auc = auc(fpr, tpr)
    sns.set()
    plt.figure('roc')
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='b',
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(args.tmp_path+'/roc.png')


def plot_catboost():
    data = []
    with open('catboost_info/test_error.tsv', 'r') as f:
        for line in f:
            data.append(line.strip().split('\t'))
    data = np.array(data[1:]).astype(np.float64)

    # 画折线图
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1])
    plt.title('AUC')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.savefig('tmp/catboost_auc.png')

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 2])
    plt.title('Test Loss')
    plt.xlabel('Iteration')
    plt.ylabel('LogLoss')
    plt.savefig('tmp/catboost_logloss.png')


def load_pretrained(my_model, optimizer, args):  # 加载预训练模型
    state = torch.load(args.model_path)
    my_model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch']
    best_acc = state['AUC']
    print("loaded pretrained model: epoch{:3d} auc= {:.4f}".format(
        epoch, best_acc))
    return my_model, optimizer, epoch+1, best_acc


def save_submission(id, y_pred, filename='out/submission.csv'):  # 保存结果到文件
    submission = pd.DataFrame({'id': id, 'isDefault': y_pred})
    submission.to_csv(filename, index=None)
