import math
import torch
import torch.nn as nn
import numpy as np
import warnings

from torch.nn.parameter import Parameter

from utils import loss_dependence, common_loss

warnings.filterwarnings('ignore')
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)  # 带有banch的矩阵相乘
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()  # expand_as 采用复制元素的形式扩展Tensor维度； contiguous 将内存变为连续的，满足如view()等 需要Tensor连续的要求
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret

class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk) #代表着图表征
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)  # 生成负样本
        h_4 = self.gcn2(seq2, diff, sparse) # 生成负样本

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class myModel(nn.Module):
    def __init__(self, batch_size, concat, nfeat_aug, nhid_aug, nout_aug, nfeat, nout, dropout, nlayer):
        # 这里的nclass 暂时等于原始属性长度 nfeat_Aug
        super(myModel, self).__init__()
        self.batch_size = batch_size

        self.gcn_l = GCN(nfeat, nhid_aug)
        self.gcn_g = GCN(nfeat, nhid_aug)
        # self.gcn_l = GraphConvolution(nfeat, nhid_aug)
        # self.gcn_g = GraphConvolution(nfeat, nhid_aug)

        ## 视图对比网络
        self.gcn1 = GCN(nfeat, nout)
        self.gcn2 = GCN(nhid_aug*2, nout)
        self.read = Readout()

        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(nout)
        self.dropout = dropout

    def forward(self, seq1, seq2, x_list_l, x_list_g, adj, diff, sparse, msk, sample_size):

        x_l, x_g = torch.mean(torch.stack(x_list_l), dim=0), torch.mean(torch.stack(x_list_g), dim=0)
        adj = adj.unsqueeze(dim=0)
        diff = diff.unsqueeze(dim=0)
        x_l = self.gcn_l(x_l.unsqueeze(dim=0), adj)
        x_g = self.gcn_g(x_g.unsqueeze(dim=0), diff)
        # x_l = self.gcn_l(x_l, adj)
        # x_g = self.gcn_g(x_g, diff)

        # 将全局和局部属性进行拼接,获得增强视图
        seq_Aug = torch.cat((x_l, x_g), dim=-1)
        # seq_Aug = torch.unsqueeze(seq_Aug, dim=0)

        shuf_idx = np.random.permutation(sample_size)  # 输入一个数或者数组，生成一个随机序列，对多维数组来说是不同维度整体随机打乱，某个维度内部不变
        shuf_seq_Aug = seq_Aug[:, shuf_idx, :]

        ## 计算两个视图的对比损失
        h_1 = self.gcn1(seq1, adj, sparse)  # 原始视图经过第一个GCN
        c_1 = self.read(h_1, msk)  # 原始视图的第一个图表征
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq_Aug, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)  # 生成负样本
        h_4 = self.gcn2(shuf_seq_Aug, diff, sparse)  # 生成负样本

        # loss_dep_x = loss_dependence(x_l.device, x_l.squeeze(), x_g.squeeze(), dim=adj.shape[-1])
        # loss_dep_h = loss_dependence(x_l.device, h_1.squeeze(), h_2.squeeze(), dim=adj.shape[-1])
        # loss_com_x = common_loss(x_l.squeeze(), x_g.squeeze())
        # loss_com_h = common_loss(h_1.squeeze(), h_2.squeeze())
        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)

        # 这里应该将局部和全局属性进行return，用来计算HSIC
        return ret, h_1, h_2, seq_Aug  #loss_dep_x, loss_dep_h, loss_com_x, loss_com_h,

    def embed(self, seq, adj, x_list_l, x_list_g, seq_Aug, diff, sparse):
        # x_l, x_g = torch.mean(torch.stack(x_list_l), dim=0), torch.mean(torch.stack(x_list_g), dim=0)
        # x_l = self.gcn_l(x_l.unsqueeze(dim=0), adj)
        # x_g = self.gcn_g(x_g.unsqueeze(dim=0), diff)
        #
        # # 将全局和局部属性进行拼接,获得增强视图
        # seq_Aug = torch.cat((x_l, x_g), dim=-1)

        h_1 = self.gcn1(seq, adj, sparse)
        # h_2 = self.mlp(seq_Aug)

        h_2 = self.gcn2(seq_Aug, diff, sparse)

        # return (h_1 + h_2).detach()
        return torch.cat((h_1, h_2, ), dim=-1).detach()