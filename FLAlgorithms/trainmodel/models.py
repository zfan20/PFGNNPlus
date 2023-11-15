# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import SGConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.utils import add_remaining_self_loops
from torch import Tensor
from torch.nn import Conv2d, KLDivLoss, Linear, Parameter

from torch_geometric.utils import to_dense_batch

EPS = 1e-15
MAX_LOGSTD = 10


class MemPooling(torch.nn.Module):
    r"""Memory based pooling layer from `"Memory-Based Graph Networks"
    <https://arxiv.org/abs/2002.09518>`_ paper, which learns a coarsened graph
    representation based on soft cluster assignments

    .. math::
        S_{i,j}^{(h)} &= \frac{
        (1+{\| \mathbf{x}_i-\mathbf{k}^{(h)}_j \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}{
        \sum_{k=1}^K (1 + {\| \mathbf{x}_i-\mathbf{k}^{(h)}_k \|}^2 / \tau)^{
        -\frac{1+\tau}{2}}}

        \mathbf{S} &= \textrm{softmax}(\textrm{Conv2d}
        (\Vert_{h=1}^H \mathbf{S}^{(h)})) \in \mathbb{R}^{N \times K}

        \mathbf{X}^{\prime} &= \mathbf{S}^{\top} \mathbf{X} \mathbf{W} \in
        \mathbb{R}^{K \times F^{\prime}}

    Where :math:`H` denotes the number of heads, and :math:`K` denotes the
    number of clusters.

    Args:
        in_channels (int): Size of each input sample :math:`F`.
        out_channels (int): Size of each output sample :math:`F^{\prime}`.
        heads (int): The number of heads :math:`H`.
        num_clusters (int): number of clusters :math:`K` per head.
        tau (int, optional): The temperature :math:`\tau`. (default: :obj:`1.`)
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int,
                 num_clusters: int, tau: float = 1.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_clusters = num_clusters
        self.tau = tau

        self.k = Parameter(torch.Tensor(heads, num_clusters, in_channels))
        self.conv = Conv2d(heads, 1, kernel_size=1, padding=0, bias=False)
        #self.lin = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.k.data, -1., 1.)
        self.conv.reset_parameters()
        #self.lin.reset_parameters()


    @staticmethod
    def kl_loss(S: Tensor) -> Tensor:
        r"""The additional KL divergence-based loss

        .. math::
            P_{i,j} &= \frac{S_{i,j}^2 / \sum_{n=1}^N S_{n,j}}{\sum_{k=1}^K
            S_{i,k}^2 / \sum_{n=1}^N S_{n,k}}

            \mathcal{L}_{\textrm{KL}} &= \textrm{KLDiv}(\mathbf{P} \Vert
            \mathbf{S})
        """
        S_2 = S**2
        P = S_2 / S.sum(dim=1, keepdim=True)
        denom = P.sum(dim=2, keepdim=True)
        denom[S.sum(dim=2, keepdim=True) == 0.0] = 1.0
        P /= denom

        loss = KLDivLoss(reduction='batchmean', log_target=False)
        return loss(S.clamp(EPS).log(), P.clamp(EPS))

    @staticmethod
    def linkpred_loss(testitems, S, adj_dense):
        S = S.squeeze(0)
        test_links = torch.matmul(S[testitems], S.transpose(0, 1))

        link_loss = torch.norm(adj_dense - test_links, p=2)
        link_loss = link_loss / adj_dense.numel()

        return link_loss


    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x (Tensor): Dense or sparse node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}` or
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`,
                respectively.
            batch (LongTensor, optional): Batch vector :math:`\mathbf{b} \in
                {\{ 0, \ldots, B-1\}}^N`, which assigns each node to a
                specific example.
                This argument should be just to separate graphs when using
                sparse node features. (default: :obj:`None`)
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}`, which
                indicates valid nodes for each graph when using dense node
                features. (default: :obj:`None`)
        """
        if x.dim() <= 2:
            x, mask = to_dense_batch(x, batch)
        elif mask is None:
            mask = x.new_ones((x.size(0), x.size(1)), dtype=torch.bool)

        (B, N, _), H, K = x.size(), self.heads, self.num_clusters

        dist = torch.cdist(self.k.view(H * K, -1), x.view(B * N, -1), p=2)**2
        dist = (1. + dist / self.tau).pow(-(self.tau + 1.0) / 2.0)

        dist = dist.view(H, K, B, N).permute(2, 0, 3, 1)  # [B, H, N, K]
        S = dist / dist.sum(dim=-1, keepdim=True)

        S = self.conv(S).squeeze(dim=1).softmax(dim=-1)  # [B, N, K]
        S = S * mask.view(B, N, 1)

        #x = self.lin(S.transpose(1, 2) @ x)
        x = S.transpose(1, 2) @ x

        return x, S


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'num_clusters={self.num_clusters})')




class FedVariationalSGCNEncoder(torch.nn.Module):
    def __init__(self, args):
        super(FedVariationalSGCNEncoder, self).__init__()
        self.pooldropout = args.pooldropout
        self.mu_encoder = SGConv(768, args.out_dim, 3, cached=True, add_self_loops=True)
        self.logstd_encoder = SGConv(768, args.out_dim, 3, cached=True, add_self_loops=True)

        self.mem1_mu = MemPooling(args.out_dim, args.out_dim, heads=args.heads, num_clusters=args.num_clusters_level1, tau=args.tau)
        self.mem1_logstd = MemPooling(args.out_dim, args.out_dim, heads=args.heads, num_clusters=args.num_clusters_level1, tau=args.tau)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def pooling_normal(self, mu_embs, logstd_embs):

        level1_clusters_mu_embs, self.s_1_mu = self.mem1_mu(mu_embs)
        level1_clusters_logstd_embs, self.s_1_logstd = self.mem1_logstd(logstd_embs)
        
        return [level1_clusters_mu_embs, level1_clusters_logstd_embs], MemPooling.kl_loss(self.s_1_mu) + MemPooling.kl_loss(self.s_1_logstd)

    def unpooling_normal(self, clusters_embs_mu, clusters_embs_logstd):
        clusters_embs_mu = clusters_embs_mu.squeeze(0)
        clusters_embs_logstd = clusters_embs_logstd.squeeze(0)
        s_1_mu_2d = self.s_1_mu.squeeze(0)
        s_1_logstd_2d = self.s_1_logstd.squeeze(0)

        node_embs_mu = torch.matmul(torch.linalg.pinv(s_1_mu_2d).transpose(0, 1), clusters_embs_mu)
        node_embs_logstd = torch.matmul(torch.linalg.pinv(s_1_logstd_2d).transpose(0, 1), clusters_embs_logstd)

        return node_embs_mu, node_embs_logstd.clamp(max=MAX_LOGSTD)

    def encode(self, node_features, adj_mat):
        self.mu = self.mu_encoder(node_features, adj_mat)
        self.logstd = self.logstd_encoder(node_features, adj_mat).clamp(max=MAX_LOGSTD)
        return self.mu, self.logstd

    def kl_loss(self, mu=None, logstd=None):
        mu = self.mu if mu is None else mu
        logstd = self.logstd if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


    def pool_lploss(self, test_items, adj_dense):
        lp_loss_mu = MemPooling.linkpred_loss(test_items, self.s_1_mu, adj_dense)
        lp_loss_logstd = MemPooling.linkpred_loss(test_items, self.s_1_logstd, adj_dense)

        return lp_loss_mu + lp_loss_logstd


    def forward(self, node_features, adj_mat):
        self.mu, self.logstd = self.encode(node_features, adj_mat)
        lastlevel_clusters_embs_array, pool_kl_loss = self.pooling_normal(self.mu, self.logstd)

        unpooled_item_embs_mu, unpooled_item_embs_logstd = self.unpooling_normal(lastlevel_clusters_embs_array[0], lastlevel_clusters_embs_array[1])
        unpooled_item_embs = self.reparametrize(unpooled_item_embs_mu, unpooled_item_embs_logstd)

        return unpooled_item_embs, lastlevel_clusters_embs_array, pool_kl_loss


class SGCN(nn.Module):
    def __init__(self, num_layers, node_feat_dim, output_dim):
        super(SGCN, self).__init__()
        self.conv_layer = SGConv(node_feat_dim, output_dim, num_layers, cached=True, add_self_loops=True)

    def forward(self, node_features, adj_mat):
        node_embs = self.conv_layer(node_features, adj_mat)

        return node_embs

class FeatMLP(nn.Module):
    def __init__(self, node_feat_dim, output_dim):
        super(FeatMLP, self).__init__()
        self.layer = torch.nn.Linear(node_feat_dim, output_dim)

    def forward(self, data):
        node_features = data.x
        node_embs = self.layer(node_features)

        return node_embs


