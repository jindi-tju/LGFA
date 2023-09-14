import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from numpy import mat

from hps import get_hyper_param
from utils import sparse_mx_to_torch_sparse_tensor, get_augmented_features, loss_dependence, normalize_features
from model import LogReg, myModel, Model
from dataset import load
import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='mvgrl')
sys.path.append('/data/wzq/model_code/mvgrl-master/node')
exc_path = sys.path[-1]

parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=1, help='GPU index.')
parser.add_argument('--epochs', type=int, default=1000, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of mvgrl.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
parser.add_argument("--sample_size", type=int, default=2000, help='args.sample_size.')
parser.add_argument("--concat", type=int, default=4, help='concat for Augmentation')
parser.add_argument("--batch_size", type=int, default=1, help='args.batch_size.')
parser.add_argument('--hidden_aug', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=bool, default=True, help='dropout.')
parser.add_argument('--k', type=int, default=32, help='ppr k top.')
parser.add_argument('--eps', type=float, default=0.005, help='.')
parser.add_argument('--my', type=bool, default=False, help='.')


# 如果换数据集 记得改 k eps

args = parser.parse_args()

hp = get_hyper_param(args.dataset)
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

def train(dataset, verbose=True):
    # Step 1: Prepare data =================================================================== #
    sparse = False
    adj, diff, features, labels, idx_train, idx_val, idx_test = load(args)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(args.batch_size, args.sample_size * 2)
    lbl_2 = torch.zeros(args.batch_size, args.sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    # Step 2: Create model =================================================================== #
    model = Model(ft_size, args.hid_dim)

    # Step 3: Create training components ===================================================== #
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    # Step 4: Training epochs ================================================================ #

    for epoch in range(args.epochs):
        idx = np.random.randint(0, adj.shape[-1] - args.sample_size + 1, args.batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + args.sample_size, i: i + args.sample_size])
            bd.append(diff[i: i + args.sample_size, i: i + args.sample_size])
            bf.append(features[i: i + args.sample_size])

        ba = np.array(ba).reshape(args.batch_size, args.sample_size, args.sample_size)
        bd = np.array(bd).reshape(args.batch_size, args.sample_size, args.sample_size)
        bf = np.array(bf).reshape(args.batch_size, args.sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(args.sample_size) # 输入一个数或者数组，生成一个随机序列，对多维数组来说是不同维度整体随机打乱，某个维度内部不变
        shuf_fts = bf[:, idx, :]  # 打乱节点特征，应该是用于后面的负样本对

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        loss = b_xent(logits, lbl)  # https://blog.csdn.net/qq_22210253/article/details/85222093

        loss.backward()
        optimiser.step()

        if verbose and epoch % 30 == 0:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('model.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()

    embeds, _ = model.embed(features, adj, diff, sparse, None)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    # Step 5:  Linear evaluation ========================================================== #
    for _ in range(50):
        log = LogReg(args.hid_dim, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.cuda()
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print('acc avg:{},  acc std:{},  time:{}s'.format(accs.mean().item(), accs.std().item(), time.time()-start))
    return accs.mean().item()



def mytrain(dataset, verbose=True):
    # Step 1: Prepare data =================================================================== #
    sparse = False
    adj, diff, features, labels, idx_train, idx_val, idx_test = load(args)
    # Pretrain cvae（还是两阶段的 没有融合到一起） ================================= #
    # 只需要运行一次，模型保存后，后面不再需要运行了。
    print("The cave pre-training of the original {} graph is in progress".format(dataset))
    # Mycvae()  # 训练原始图的cvae生成模型
    print("The cave pre-training for diffusion {} graph in progress".format(dataset))
    # Mycvae(48000, mat(diff))  # 训练扩散图的cvae生成模型

    args.sample_size = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    features = normalize_features(features)  # 特征归一化了


    lbl_1 = torch.ones(args.batch_size, args.sample_size * 2)
    lbl_2 = torch.zeros(args.batch_size, args.sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    # Step 2: Create model =================================================================== #
    model = myModel(
        batch_size=args.batch_size,
        concat=args.concat,
        nfeat_aug=features.shape[1],
        nhid_aug=args.hidden_aug,
        nout_aug=features.shape[1],
        nfeat=features.shape[1],
        nout=args.hid_dim,
        dropout=args.dropout,
        nlayer=2,
    )


    # Step 3: Create training components ===================================================== #
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    x_list_l_best, x_list_g_best = None, None

    if torch.cuda.is_available():
        model.to(args.device)
        labels = labels.to(args.device)
        lbl = lbl.to(args.device)
        idx_train = idx_train.to(args.device)
        idx_test = idx_test.to(args.device)

    # Step 4: Training epochs ================================================================ #

    for epoch in range(args.epochs):
        # 生成增强属性
        # 每次采样进行四次增强，然后与原始的数据进行拼接，最后进行映射得出一个最终的输出。最后会有四个输出
        cvae_model_l = torch.load("{}/model_cvae/{}_l.pkl".format(exc_path, args.dataset))
        cvae_model_g = torch.load("{}/model_cvae/{}_g.pkl".format(exc_path, args.dataset))
        cvae_model_l.to(args.device)
        cvae_model_g.to(args.device)

        x_list_l = get_augmented_features(args, cvae_model_l, args.concat, features)
        x_list_g = get_augmented_features(args, cvae_model_g, args.concat, features)
        x_list_l = x_list_l + [torch.FloatTensor(features).to(args.device)]
        x_list_g = x_list_g + [torch.FloatTensor(features).to(args.device)]


        bf = [features]
        bf = np.array(bf).reshape(args.batch_size, args.sample_size, ft_size)


        bf = torch.FloatTensor(bf)
        fts_idx = np.random.permutation(args.sample_size) # 输入一个数或者数组，生成一个随机序列，对多维数组来说是不同维度整体随机打乱，某个维度内部不变
        shuf_fts = bf[:, fts_idx, :]  # 打乱节点特征，应该是用于后面的负样本对

        if torch.cuda.is_available():
            bf = bf.to(args.device)
            shuf_fts = shuf_fts.to(args.device)

        model.train()
        optimiser.zero_grad()

        # print("\n=====迭代开始=====")
        # for name, parms in model.named_parameters():
        #     print('-->name:', name)
        #     print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")



        # x_l, x_g 分别是局部增强视图和全局增强视图(不包含原始属性特征)
        logits, h_1, h_2, seq_Aug = model(bf, shuf_fts, x_list_l, x_list_g,
                                                     torch.FloatTensor(adj).to(args.device),
                                                     torch.FloatTensor(diff).to(args.device), sparse, None,
                                                     args.sample_size)
        t_b, t_dx, t_dh, t_cx, t_ch = 1, 0, 0, 0, 1
        loss = b_xent(logits, lbl)  # https://blog.csdn.net/qq_22210253/article/details/85222093
        # loss = t_b * loss_b + \
        #        t_dx * loss_dep_x + \
        #        t_dh * loss_dep_h + \
        #        t_cx * loss_com_x + \
        #        t_ch * loss_com_h

        loss.backward()
        optimiser.step()

        # if verbose and epoch % 1 == 0:
        #     print('Epoch: {0}, Loss: {1:0.6f}, loss_b: {2:0.6f}, loss_dep_x: {3:0.6f}, '
        #           'loss_dep_h: {4:0.6f}, loss_com_x: {5:0.6f}, loss_com_h: {6:0.6f}'.
        #           format(epoch, loss.item(), (t_b * loss_b).item(), (loss_dep_x).item(),
        #                  (loss_dep_h).item(), (loss_com_x).item(), (loss_com_h).item()))
        if verbose and epoch % 1 == 0:
            print('Epoch: {0}, Loss: {1:0.6f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), '{}_model.pkl'.format(args.dataset))
            x_list_l_best, x_list_g_best = x_list_l, x_list_g
            # embeds = torch.cat((h_1, h_2), dim=2).detach()
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('{}_model.pkl'.format(args.dataset)))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    # seq_Aug = torch.FloatTensor(seq_Aug[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.to(args.device)
    adj = adj.to(args.device)
    diff = diff.to(args.device)
    # seq_Aug = seq_Aug.to(args.device)

    embeds = model.embed(features, adj, x_list_l_best, x_list_g_best, seq_Aug, diff, sparse)
    train_embs = embeds[0, idx_train]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    test_lbls = labels[idx_test]

    accs = []
    wd = 0.01 if dataset == 'citeseer' else 0.0

    # Step 5:  Linear evaluation ========================================================== #
    for _ in range(50):
        log = LogReg(args.hid_dim * 2, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.to(args.device)
        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)

    accs = torch.stack(accs)

    print('acc avg:{},  acc std:{},  time:{}s'.format(accs.mean().item(), accs.std().item(), time.time()-start))
    return accs.mean().item()



if __name__ == '__main__':
    start = time.time()
    res = []
    for i in range(10):
        if args.my:
            print(args)
            print("Ours Round {} for {}".format(i, args.dataset))
            re = mytrain(args.dataset)
            res.append(re)
        else:
            print(args)
            print("Baseline Round {} for {}".format(i, args.dataset))
            re = train(args.dataset)
            res.append(re)
    print(res, 'mean:{}, var:{}'.format(np.mean(res), np.var(res)))
    with open('result_{}.txt'.format(args.dataset), 'w') as f:
        f.write(str(re) + 'mean:{}, var:{}'.format(np.mean(res), np.var(res)))


