import torch
from dgl.data import CoraGraphDataset, CitationGraphDataset, WikiCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from torch.utils.data import random_split

from utils import preprocess_features, normalize_adj, get_top_k_matrix, get_clipped_matrix
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os
import dgl


def download(dataset):
    if dataset == 'cora':
        return CoraGraphDataset()
    elif dataset == 'citeseer':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'pubmed':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'wikics':
        return WikiCSDataset()
    elif dataset == 'amzcom':
        return AmazonCoBuyComputerDataset()
    elif dataset == 'amzphoto':
        return AmazonCoBuyPhotoDataset()
    else:
        return None


def load(args):
    datadir = os.path.join('data', args.dataset)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(args.dataset)
        graph = ds[0]
        if args.dataset in ['cora', 'citeseer', 'pubmed']:
            nx_g = dgl.to_networkx(graph)
            adj = nx.to_numpy_array(nx_g)
            diff = compute_ppr(nx_g, 0.2)

            feat = ds.features[:]
            labels = ds.labels[:]

            idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
            idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
            idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)
        elif args.dataset in ['wikics']:
            nx_g = dgl.to_networkx(graph)
            adj = nx.to_numpy_array(nx_g)
            diff = compute_ppr(nx_g, 0.2)

            feat = graph.ndata['feat']
            labels = graph.ndata['label']

            train_mask = graph.ndata['train_mask']
            val_mask = graph.ndata['val_mask']
            test_mask = graph.ndata['test_mask']
            idx_train = np.argwhere(train_mask == 1).reshape(-1)
            idx_val = np.argwhere(val_mask == 1).reshape(-1)
            idx_test = np.argwhere(test_mask == 1).reshape(-1)
        elif args.dataset in ['amzcom', 'amzphoto']:
            nx_g = dgl.to_networkx(graph)
            adj = nx.to_numpy_array(nx_g)
            diff = compute_ppr(nx_g, 0.2)

            feat = graph.ndata['feat']
            labels = graph.ndata['label']
            train_mask, test_mask, val_mask = generate_split(adj.shape[0], train_ratio=0.1, val_ratio=0.1)
            idx_train = np.argwhere(train_mask == 1).reshape(-1)
            idx_val = np.argwhere(val_mask == 1).reshape(-1)
            idx_test = np.argwhere(test_mask == 1).reshape(-1)

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)

        if args.k:
            print(f'Selecting top {args.k} edges per node.')
            diff = get_top_k_matrix(diff, k=args.k)
        if args.eps:
            print(f'Selecting edges with weight greater than {args.eps}.')
            diff = get_clipped_matrix(diff, eps=args.eps)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        if args.k:
            print(f'Selecting top {args.k} edges per node.')
            diff = get_top_k_matrix(diff, k=args.k)
        if args.eps:
            print(f'Selecting edges with weight greater than {args.eps}.')
            diff = get_clipped_matrix(diff, eps=args.eps)
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')

    if args.dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)


    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, diff, feat, labels, idx_train, idx_val, idx_test

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask
if __name__ == '__main__':
    load('cora')
