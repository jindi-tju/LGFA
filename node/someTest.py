#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 12:27
# @Author  : zjh
# @File    : someTest.py
# @Software: PyCharm
# @Describe:

import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from torch.nn import functional as F

def clusteringTest(embeddings, labels,my=""):
    # embeddings1 = F.normalize(embeddings, dim=-1, p=2).detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()
    nb_class = len(labels.unique())
    true_y = labels.detach().cpu().numpy()

    estimator = KMeans(n_clusters=nb_class)

    NMI_list = []
    h_list = []
    a_list = []
    for i in range(10):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        h_score = metrics.homogeneity_score(true_y, y_pred)
        ari = metrics.adjusted_rand_score(true_y, y_pred)
        s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)
        h_list.append(h_score)
        a_list.append(ari)

    s1 = sum(NMI_list) / len(NMI_list)
    h_score = sum(h_list) / len(h_list)
    a_score = sum(a_list) / len(a_list)
    print('Evaluate clustering results')
    # with open('./result_cluster_{}.txt'.format(args.dataset),'a+') as f:
    #     f.write('{:.4f} {:.4f}\n'.format(s1, h_score))
    # with open("./resultNMI_Cluster_noNorm_my.txt",'a+') as f:
    #     f.write("{:.4f},".format(s1))
    # with open("./resultARI_Cluster_noNorm_my.txt",'a+') as f:
    #     f.write("{:.4f},".format(a_score))
    print('** Clustering NMI: {:.4f} | ARI : {:.4f} **'.format(s1, a_score))

def clusteringTest2(embeddings, labels,my=""):
    embeddings = F.normalize(embeddings, dim=-1, p=2).detach().cpu().numpy()
    # embeddings = embeddings.detach().cpu().numpy()
    nb_class = len(labels.unique())
    true_y = labels.detach().cpu().numpy()

    estimator = KMeans(n_clusters=nb_class)

    NMI_list = []
    h_list = []
    a_list = []
    for i in range(10):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        h_score = metrics.homogeneity_score(true_y, y_pred)
        ari = metrics.adjusted_rand_score(true_y, y_pred)
        s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)
        h_list.append(h_score)
        a_list.append(ari)

    s1 = sum(NMI_list) / len(NMI_list)
    h_score = sum(h_list) / len(h_list)
    a_score = sum(a_list) / len(a_list)
    print('Evaluate clustering results')
    # with open('./result_cluster_{}.txt'.format(args.dataset),'a+') as f:
    #     f.write('{:.4f} {:.4f}\n'.format(s1, h_score))
    # with open("./resultNMI_Cluster_noNorm_my.txt",'a+') as f:
    #     f.write("{:.4f},".format(s1))
    # with open("./resultARI_Cluster_noNorm_my.txt",'a+') as f:
    #     f.write("{:.4f},".format(a_score))
    print('** Clustering NMI: {:.4f} | ARI : {:.4f} **'.format(s1, a_score))

def get_clustering_label(embeddings, labels):
    # embeddings = F.normalize(embeddings, dim=-1, p=2).detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()
    nb_class = len(labels.unique())
    estimator = KMeans(n_clusters=nb_class)
    estimator.fit(embeddings)
    y_pred = estimator.predict(embeddings)
    return y_pred;

def plot_tsne(data, label,my=""):
    print('Computing t-SNE embedding')
    # label = label.cpu()
    label = torch.from_numpy(label).cpu()
    data = data.cpu().detach().numpy()    #   这里 data变得非常小
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    result = tsne.fit_transform(data)
    # result = tsne.fit_transform(data)
    nb_classes = label.unique().max()+1
    print("class number is {}".format(nb_classes))


    # t1 = time()
    fig = plt.figure(figsize=(8, 8))
    from matplotlib.ticker import NullFormatter
    # print("t-SNE: %.2g sec" % (t1 - t0))  # 算法用时
    print("t-SNE done")
    plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去 x 轴刻度
    # plt.yticks([])  # 去 y 轴刻度
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(bottom=False, top=False, left=False, right=False)  # 移除全部刻度
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    colors = [
            '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
        ]

    for i in range(nb_classes):
        plt.scatter(result[label == i, 0], result[label == i, 1], s=20, color=colors[i])

    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    # plt.show()
    # plt.savefig('/media/user/2FD84FB396A78049/Yuzz/BIGCN_topic/plot_result/plot_tsne.png')
    # plt.savefig('/data/wzq/model_tsne/MVGRL_Cluster_{}{}.png'.format(args.dataset,my))

    plt.savefig('./model_tsne/MVGRL_Cluster_pubmed_my.png')
    print("save done!")


if __name__=='__main__':
    embed = torch.load('./pubmedEmbedWithoutTsne.pth')
    label = torch.load('./pubmedTruelabelWithoutTsne.pth')

    clusteringTest(embed,label)
    predict_label = get_clustering_label(embed, label)
    plot_tsne(embed, predict_label,my="_my")
    print('--------------------------------------------')
    # clusteringTest2(embed, label)