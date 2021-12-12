'''
Author: Derry
Email: drlv@mail.ustc.edu.cn
Date: 2021-07-25 23:39:03
LastEditors: Derry
LastEditTime: 2021-12-12 19:39:49
Description: Standard main file of a neural network
'''
import argparse
import os
import time

import catboost as cb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import *
from utils import *


def train(my_model, train_loader, test_loader, optimizer, args, start_epoch=0, best_auc=0.5):
    if args.pretrained and os.path.exists(args.model_path):
        my_model, optimizer, start_epoch, best_auc = load_pretrained(
            my_model, optimizer,  args)

    test_loss_all, test_auc_all = [], []
    for epoch in range(start_epoch, args.epoch):
        start = time.time()
        for batch, (X_train, y_train) in enumerate(train_loader):
            my_model.train()

            if args.cuda:
                X_train = X_train.cuda()
                y_train = y_train.cuda()

            optimizer.zero_grad()
            y_poss = my_model(X_train)
            loss = loss_fun(y_poss[:, 1], y_train)
            loss.backward()
            optimizer.step()

        if not args.fastmode:
            print("Epoch {:3d}".format(epoch),
                  "time= {:.2f} s".format(time.time()-start), end=' ')
            test_loss, test_auc, fpr, tpr = test(
                my_model, test_loader, loss_fun, args)

            # Saving the best model
            if test_auc > best_auc:
                best_auc = test_auc
                state = {'model': my_model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch+1,
                         'AUC': test_auc}
                torch.save(state, args.model_path)
                plot_roc(fpr, tpr, args)

            test_loss_all.append(test_loss)
            test_auc_all.append(test_auc)
            plot(test_loss_all, test_auc_all, args)


@torch.no_grad()
def test(my_model, test_loader, loss_fun, args):
    loss_all, auc_all = 0, 0
    y_test_all, y_poss_all = [], []
    for batch, (X_test, y_test) in enumerate(test_loader):
        if args.cuda:
            X_test = X_test.cuda()
            y_test = y_test.cuda()
        loss, y_poss = evaluate(my_model, X_test, y_test, loss_fun)
        loss_all += loss
        y_test_all.extend(y_test.cpu().numpy().tolist())
        y_poss_all.extend(y_poss.cpu().numpy().tolist())

    y_test_all = np.array(y_test_all)
    y_poss_all = np.array(y_poss_all)
    fpr, tpr, threshold = roc_curve(y_test_all, y_poss_all)
    test_num = len(test_loader)
    AUC = auc(fpr, tpr)
    print("Test set results:",
          "loss= {:.4f}".format(loss_all/test_num),
          "auc= {:.4f}".format(AUC))
    return loss_all/test_num, AUC, fpr, tpr


@torch.no_grad()
def infer(my_model, X_test):
    optimizer = torch.optim.AdamW(my_model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    my_model, optimizer, start_epoch, best_auc = load_pretrained(
        my_model, optimizer, args)
    my_model.eval()
    if args.cuda:
        my_model.cuda()
        X_test = X_test.cuda()
    y_poss = my_model(X_test)
    y_pred = y_poss[:, 1].cpu().numpy()
    return y_pred


def main_ml(X_train, y_train, X_test, iter=10000):
    # for model in (ML_lgbm, ML_rf, ML_xgboost, ML_gbdt, ML_mlp):
    #     name = model.__name__
    #     ml_valid(X_train, y_train, model)
    #     y_pred = ml_predict(X_train, y_train, model, X_test)
    #     print(name)
    #     save_submission(id, y_pred, 'out/submission_'+name+'.csv')

    cat_model = cb.CatBoostClassifier(iterations=iter,
                                      depth=8,
                                      learning_rate=0.001,
                                      loss_function='Logloss',
                                      eval_metric='AUC',
                                      logging_level='Verbose',
                                      metric_period=100)
    cat_model.fit(X_train, y_train, eval_set=(X_train, y_train))
    y_pred = cat_model.predict_proba(X_test)[:, 1]
    plot_catboost()
    return y_pred


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()
    # Training outer arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Using pretrained model parameter.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epochs to train.')
    # Training inner arguments
    parser.add_argument('--batch_size', type=int, default=20000,
                        help='Number of samples in a batch.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    # Model arguments
    parser.add_argument('--n_in', type=int, default=100,
                        help='Number of input units, self-tuning by input data.')
    parser.add_argument('--n_out', type=int, default=100,
                        help='Number of output unit, self-tuning by output data.')
    parser.add_argument('--n_hid', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--cat_iter', type=int, default=10000,
                        help="iteration number of catboost")
    # Path arguments
    parser.add_argument('--data_path', type=str, default="./data",
                        help='Path of dataset')
    parser.add_argument('--tmp_path', type=str, default="./tmp",
                        help='Path of tmporary output')
    parser.add_argument('--model_path', type=str, default="./model/best_model.tar",
                        help='Path of model parameter')
    parser.add_argument('--submission_path', type=str,
                        default="./out/submission.csv", help='Path of submission')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    train_bank, train_internet, test_data = load_data()
    X_train, y_train, X_test = prepare_data(
        train_bank, train_internet, test_data)

    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32)

    args.n_in = X_train_tensor.shape[1]
    args.n_out = len(set(list(y_train_tensor.numpy())))

    # Construct data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    # Model, loss function and optimizer
    my_model = MLP(args)
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.AdamW(my_model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        print("cuda is available!")
        torch.cuda.manual_seed(args.seed)
        my_model.cuda()

    # NN training
    train(my_model, train_loader, valid_loader, optimizer, args)
    y_pred_nn = infer(my_model, X_test_tensor)
    save_submission(test_data['loan_id'], y_pred_nn,
                    filename='out/submission_nn.csv')

    # ML training
    y_pred_ml = main_ml(X_train, y_train, X_test, iter=10000)
    save_submission(test_data['loan_id'], y_pred_ml, 'out/submission_cat.csv')

    # Combine NN and ML predict result
    y_pred = 0.6*y_pred_ml + 0.4*y_pred_nn
    save_submission(test_data['loan_id'], y_pred,
                    filename=args.submission_path)
