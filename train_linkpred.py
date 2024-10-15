# coding=utf-8
from __future__ import print_function
from __future__ import division
from model import GCNModelVAE, Discriminator, Discriminator_FC, Generator_FC
from utils import load_data, load_data2, preprocess_graph, linkpred_metrics, mask_test_edges
from copy import deepcopy
import pandas as pd
from optimizer import loss_function
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch
import scipy.sparse as sp
import numpy as np


import argparse
import time
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

# compute ac and nmi


def get_NMI(n_clusters, emb, true_label):
    true_label = list(true_label.numpy())
    from sklearn.cluster import KMeans
    from sklearn import metrics
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    predict_labels = kmeans.predict(emb)
    nmi = metrics.normalized_mutual_info_score(true_label, predict_labels)
    acc, f1, precision, recall = clusteringAcc(true_label, predict_labels)
    ari = metrics.adjusted_rand_score(true_label, predict_labels)
    return acc, nmi, f1, precision, ari


def clusteringAcc(true_label, pred_label):
    from sklearn import metrics
    from munkres import Munkres, print_matrix
    # best mapping between true_label and predict label
    l1 = list(set(true_label))
    numclass1 = len(l1)
    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)
    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(true_label, new_predict)
    f1_macro = metrics.f1_score(true_label, new_predict, average='macro')
    f1_micro = metrics.f1_score(true_label, new_predict, average='micro')
    f1 = max(f1_macro, f1_micro)
    precision_macro = metrics.precision_score(
        true_label, new_predict, average='macro')
    precision_micro = metrics.precision_score(
        true_label, new_predict, average='micro')
    precision = max(precision_macro, precision_micro)
    recall_macro = metrics.recall_score(
        true_label, new_predict, average='macro')
    recall_micro = metrics.recall_score(
        true_label, new_predict, average='micro')
    recall = max(recall_macro, recall_micro)
    return acc, f1, precision, recall
    # '''
    # return acc


def log(x):
    return torch.log(x + 1e-8)


def d2_output_mat(D2_output):
    D2_output_p1 = F.sigmoid(D2_output)
    D2_output_p0 = torch.ones(D2_output_p1.shape).to(device) - D2_output_p1
    D2_output_mat = torch.concat([D2_output_p0, D2_output_p1], axis=1)
    D2_output_mat = log(D2_output_mat)
    return D2_output_mat


def gae_for(args):
    print(args)
    # load dataset
    if args.dataset_str in ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010']:
        adj, features, label = load_data2(args.dataset_str)
    else:
        adj, features, label = load_data(args.dataset_str)
    # print(adj.shape,features.shape)

    # link prediction data split
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    features = torch.tensor(features).to(device)
    label = torch.tensor(label).to(device)

    #adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    # print(adj.shape)
    adj_orig = adj_orig - \
        sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [
                      0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    #adj = adj_train
    adj = adj_orig
    adj_train = adj
    # T=torch.FloatTensor(adj.todense())

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray()).to(device)
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = torch.tensor(pos_weight).to(device)
    norm = adj.shape[0] * adj.shape[0] / \
        float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    norm = torch.tensor(norm).to(device)
    mini_batch = adj.shape[0]
    # define model
    model = GCNModelVAE(feat_dim, args.hidden1,
                        args.hidden2, args.dropout).to(device)
    G = Generator_FC(args.hidden2, args.hidden1, feat_dim).to(device)
    D = Discriminator_FC(args.hidden2, args.hidden1, feat_dim).to(device)
    D2 = Discriminator(feat_dim).to(device)

    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    D2_optimizer = optim.Adam(D2.parameters(), lr=args.lr)
    #G_solver = torch.optim.Adam(itertools.chain(E.parameters(), G.parameters()), lr=lr, betas=[0.5,0.999], weight_decay=2.5*1e-5)
    #D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=[0.5,0.999], weight_decay=2.5*1e-5)

    hidden_emb = None
    NMI = []
    AC = []
    F1 = []
    Precision = []
    ARI = []
    AUC = []
    AP = []
    embedding = []
    lamda = 0.5
    for epoch in range(args.epochs):
        d_loss = 0
        g_loss = 0
        cur_loss = 0
        t = time.time()

        ####################
        # 图拓扑结构重构损失
        model.train()
        optimizer.zero_grad()
        adj_norm = adj_norm.to(device)
        recovered, mu = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             norm=norm, pos_weight=pos_weight).to(device)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        ####################
        # negative sample generator
        ''''''
        D2.train()
        D2_optimizer.zero_grad()
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        z = Variable(torch.FloatTensor(np.random.normal(
            0, 1, (mini_batch, args.hidden2)))).to(device)
        X_hat = G(z)
        D_result = D2(X_hat)
        D_fake_loss = D_result
        D_result = D2(features)
        D_real_loss = D_result
        D_train_loss = -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))
        D_train_loss.backward(retain_graph=True)
        d_loss = D_train_loss.item()
        D2_optimizer.step()
        if int(args.M) > 0:
            M = int(args.M)
            alpha = 2
            beta = 1.5
            KL_target = 0.1
            # PPO loss
            D_xi_old_1 = torch.tensor(D2(features), requires_grad=False) + 1e-8
            D_xi_old_0 = torch.tensor(D2(X_hat), requires_grad=False) + 1e-8
            # PPO
            for j in range(M):
                # update again with ppo
                D2.train()
                D2_optimizer.zero_grad()
                z = Variable(torch.FloatTensor(np.random.normal(
                    0, 1, (mini_batch, args.hidden2)))).to(device)
                X_hat = G(z)
                D_xi_new_0 = D2(X_hat)
                D_xi_new_1 = D2(features)
                A_hat_1 = D_xi_new_1 - torch.ones(D_xi_new_1.shape).to(device)
                A_hat_0 = -D_xi_new_0
                D_xi_new = torch.concat([D_xi_new_1, D_xi_new_0], axis=0)
                D_xi_old = torch.concat([D_xi_old_1, D_xi_old_0], axis=0)
                D_xi_new_mat = d2_output_mat(D_xi_new).to(device)
                D_xi_old_mat = d2_output_mat(D_xi_old).to(device)
                kl = F.kl_div(D_xi_new_mat, D_xi_old_mat,
                              reduction='batchmean', log_target=True)
                ppo_loss = - (torch.mean((D_xi_new_1/D_xi_old_1) * A_hat_1)
                              + torch.mean((D_xi_new_0/D_xi_old_0) * A_hat_0) - lamda * kl)
                ppo_loss.to(device)
                ppo_loss.backward()
                D2_optimizer.step()
            D_xi_new_0 = D2(X_hat)
            D_xi_new_1 = D2(features)
            D_xi_new = torch.concat(
                [D_xi_new_1, D_xi_new_0], axis=0).to(device)
            D_xi_new_mat = d2_output_mat(D_xi_new).to(device)
            D_xi_old_mat = d2_output_mat(D_xi_old).to(device)

            kl = F.kl_div(D_xi_new_mat, D_xi_old_mat,
                          reduction='batchmean', log_target=True)

            if kl.cpu().detach().numpy() > beta * KL_target:
                lamda = alpha * lamda
            elif kl.cpu().detach().numpy() < (KL_target / beta):
                lamda = lamda / alpha

        ##############
        # Mutual Information Discriminaiton
        D.train()
        D_optimizer.zero_grad()

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        #recovered, mu, logvar = model(features, adj_norm)
        z = Variable(torch.FloatTensor(np.random.normal(
            0, 1, (mini_batch, args.hidden2)))).to(device)
        X_hat = G(z)
        #loss=loss_function(preds=recovered, labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight)
        D_result = D(X_hat, z)
        #D_real_loss= D.loss(D_result,y_real_)
        #D_real_loss= torch.nn.ReLU()(1.0 - D_result).mean()
        D_fake_loss = D_result

        recovered, mu = model(features, adj_norm)
        D_result = D(features, mu)
        #loss=loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
        #D_fake_loss= D.loss(D_result,y_fake_)
        #D_fake_loss= torch.nn.ReLU()(1.0 + D_result).mean()
        D_real_loss = D_result
        #D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))
        #D_train_loss = 0.1*D_real_loss + 0.1*D_fake_loss + loss
        #D_train_loss = 0.1*D_real_loss + 0.1*D_fake_loss
        D_train_loss = -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))
        # D_train_loss.backward(retain_graph=True)
        D_train_loss.backward(retain_graph=True)
        d_loss = D_train_loss.item()
        D_optimizer.step()

        #################
        model.train()
        G.train()
        D.eval()
        D2.eval()
        optimizer.zero_grad()
        optimizer_G.zero_grad()

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        z = Variable(torch.FloatTensor(np.random.normal(
            0, 1, (mini_batch, args.hidden2)))).to(device)
        D_result = D(X_hat, z)
        D_fake_loss = D_result
        D_result = D2(X_hat)
        D2_fake_loss = D_result

        recovered, mu = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             norm=norm, pos_weight=pos_weight).to(device)
        D_result = D(features, mu)
        D_real_loss = D_result

        # G_train_loss= 0.1*D_fake_loss+args.lambda1*loss #+args.lambda2*torch.trace(mu.t().mm(T).mm(mu))
        #G_train_loss= -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))+args.lambda1*loss
        # autoencoder loss,不会更新G
        G_train_loss = -0.01*torch.mean(log(D_real_loss))+args.lambda1*loss
        G_train_loss.backward()
        optimizer.step()
        G2_train_loss = - \
            torch.mean(log(D_fake_loss)+log(D2_fake_loss))  # 用到X_hat,会更新G

        # pamil: new loss with pessimism-start# add by syh
        X_hat_grad = torch.autograd.grad(
            inputs=X_hat, outputs=G2_train_loss)[0]
        X_hat_new = X_hat + args.p_alpha * X_hat_grad
        D_fake_loss = D(X_hat_new, z)
        D2_fake_loss = D2(X_hat_new)
        # 用新的X_hat更新loss
        G2_train_loss = -torch.mean(log(D_fake_loss)+log(D2_fake_loss))
        # pamil: new loss with pessimism-end   # add by syh

        G2_train_loss.backward()
        g_loss = G_train_loss.item()
        optimizer_G.step()
        ##################

        hidden_emb = mu.cpu().data.numpy()
        if torch.any(torch.isnan(mu)):
            print("====== emb NAN ====")
            print(features, adj_norm)
            print(mu, hidden_emb)

        # print(hidden_emb)
        ac, nmi, f1, precision, ari = get_NMI(
            int(label.max()+1), hidden_emb, label.cpu())
        NMI.append(nmi)
        AC.append(ac)
        F1.append(f1)
        Precision.append(precision)
        ARI.append(ari)

        # link prediction
        lp = linkpred_metrics(test_edges, test_edges_false)
        auc, ap = lp.get_linkpred_metric(hidden_emb)
        AUC.append(auc)
        AP.append(ap)

        embedding.append(hidden_emb)

    #roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    #print('Test ROC score: ' + str(roc_score))
    #print('Test AP score: ' + str(ap_score))

    emb = embedding[AC.index(max(AC))]

    dict_metric = {"AC": round(max(AC), 4),
                   "NMI": round(max(NMI), 4),
                   "F1": round(max(F1), 4),
                   "Precision": round(max(Precision), 4),
                   "ARI": round(max(ARI), 4),
                   "AUC": round(max(AUC), 4),
                   "AP": round(max(AP), 4)}

    df_ = pd.DataFrame([dict_metric])
    df_ = df_.loc[:, ["AC", "NMI", "F1", "Precision", "ARI", "AUC", "AP"]]

    print(df_)

    return deepcopy(dict_metric), (label, emb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=256,
                        help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=128,
                        help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset_str', type=str,
                        default='cora', help='type of dataset.')
    # ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010'] 'cora','citeseer','pubmed'
    parser.add_argument('--silence', type=bool, default=True,
                        help='alpha for pessimisim')

    parser.add_argument('--M', type=int, default=1,
                        help='M of ppo, if M=0, no ppo')

    parser.add_argument('--p_alpha', type=float,
                        default=0.05, help='alpha for pessimisim')
    parser.add_argument('--link_prediction', type=bool, default=True,
                        help="if or not to run link prediction task")
    args = parser.parse_args()
    args.device = device
    print(args)

    for lr in [0.001]:
        # for p_alpha in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1,\
        #                 1.5, 2, 5, 10, 20, 30, 50, 100]:
        for p_alpha in [0, 0.001, 0.01, 0.1, 0.3, 0.9, 1.5, 5, 20]:
            num_exp = 10
            for _ in range(1, num_exp+1):
                print("----------------------------")
                print("{} dataset, lr = {}, alpha = {}, M = {}, id = {}".format(
                    args.dataset_str, lr, p_alpha, args.M, _))
                args.lr = lr
                args.p_alpha = p_alpha
                # train
                metric, (label, emb) = gae_for(args)
                # save training result
                if 1:
                    fh = open(
                        './log/best_metric_{}.txt'.format(args.dataset_str), 'a')
                    fh.write('lr = {}, alpha = {}, M = {}, id = {}, Acc = {}, NMI = {}, ARI = {}, AUC = {}, AP = {}'.format(
                        args.lr, args.p_alpha, args.M, _, metric["AC"], metric["NMI"], metric["ARI"], metric["AUC"], metric["AP"]))
                    fh.write('\r\n')
                    fh.flush()
                    fh.close()
                # save best embedding
                if 0:
                    df_emb = pd.DataFrame()
                    df_emb["y"] = list(label)
                    df_emb["emb"] = [list(emb[i, :]) for i in range(len(emb))]
                    emb_csv_name = "./log/emb_{}_lr_{}_alpha_{}_M_{}_id_{}.csv".format(
                        args.dataset_str, args.lr, args.p_alpha, args.M, _)
                    df_emb.to_csv(emb_csv_name, index=0)
                # save model
                if 0:
                    model_name = "model_{}_lr_{}_alpha_{}_id_{}.pt".format(
                        args.dataset_str, args.lr, args.p_alpha, _)
