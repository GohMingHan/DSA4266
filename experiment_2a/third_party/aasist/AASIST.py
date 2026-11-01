"""AASIST implementation adapted from the authors' MIT-licensed source.

Copyright (c) 2021-present NAVER Corp. See LICENSE in this directory.
Source: https://github.com/clovaai/aasist/blob/main/models/AASIST.py
Inspected: 2026-09-19. This is NOT a byte-identical upstream checkout.

All named modules, dimensions, graph operations, dropout rates and parameter
keys follow the official AASIST model. Changes: condensed comments/formatting,
explicit NumPy arrays when constructing the fixed sinc bank for NumPy 2.x,
and omission of two unused local master-node expansions. The six residual
blocks intentionally preserve upstream's first-convolution behavior. No
simplified/surrogate architecture is substituted. Default Freq_aug is False.
"""
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        return self.act(self._apply_BN(x))

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map(self, x):
        att_map = torch.tanh(self.att_proj(self._pairwise_mul_nodes(x)))
        return F.softmax(torch.matmul(att_map, self.att_weight) / self.temp, dim=-2)

    def _project(self, x, att_map):
        return self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)

    def _apply_BN(self, x):
        shape = x.size()
        return self.bn(x.view(-1, shape[-1])).view(shape)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.)

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x1, x2 = self.proj_type1(x1), self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, n1, n2)
        master = self._update_master(x, master)
        x = self.act(self._apply_BN(self._project(x, att_map)))
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master

    def _update_master(self, x, master):
        return self._project_master(x, master, self._derive_att_map_master(x, master))

    def _pairwise_mul_nodes(self, x):
        n = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, n, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map_master(self, x, master):
        att_map = torch.tanh(self.att_projM(x * master))
        return F.softmax(torch.matmul(att_map, self.att_weightM) / self.temp, dim=-2)

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = torch.tanh(self.att_proj(self._pairwise_mul_nodes(x)))
        board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        n = num_type1
        board[:, :n, :n, :] = torch.matmul(att_map[:, :n, :n, :], self.att_weight11)
        board[:, n:, n:, :] = torch.matmul(att_map[:, n:, n:, :], self.att_weight22)
        board[:, :n, n:, :] = torch.matmul(att_map[:, :n, n:, :], self.att_weight12)
        board[:, n:, :n, :] = torch.matmul(att_map[:, n:, :n, :], self.att_weight12)
        return F.softmax(board / self.temp, dim=-2)

    def _project(self, x, att_map):
        return self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)

    def _project_master(self, x, master, att_map):
        return self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x)) + self.proj_without_attM(master)

    def _apply_BN(self, x):
        shape = x.size()
        return self.bn(x.view(-1, shape[-1])).view(shape)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        return self.top_k_graph(self.sigmoid(self.proj(self.drop(h))), h, self.k)

    def top_k_graph(self, scores, h, k):
        _, nodes, features = h.size()
        nodes = max(int(nodes * k), 1)
        _, idx = torch.topk(scores, nodes, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, features))


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 in_channels=1, stride=1, padding=0, dilation=1,
                 bias=False, groups=1, mask=False):
        super().__init__()
        if in_channels != 1 or bias or groups != 1:
            raise ValueError("AASIST sinc convolution requires mono, bias=False, groups=1.")
        self.out_channels = out_channels
        self.kernel_size = kernel_size + int(kernel_size % 2 == 0)
        self.sample_rate = sample_rate
        self.stride, self.padding, self.dilation, self.mask = stride, padding, dilation, mask
        f = int(sample_rate / 2) * np.linspace(0, 1, 257)
        fmel = self.to_mel(f)
        self.mel = self.to_hz(np.linspace(np.min(fmel), np.max(fmel), out_channels + 1))
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        # Fixed front-end filters are not learned parameters in the official model.
        self.band_pass = torch.zeros(out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin, fmax = self.mel[i], self.mel[i + 1]
            # Explicit conversion avoids passing a Torch tensor through NumPy ufuncs.
            high = np.sinc((2 * fmax * self.hsupp / sample_rate).numpy())
            low = np.sinc((2 * fmin * self.hsupp / sample_rate).numpy())
            ideal = (2 * fmax / sample_rate) * high - (2 * fmin / sample_rate) * low
            self.band_pass[i] = torch.tensor(np.hamming(self.kernel_size), dtype=torch.float32) * torch.tensor(ideal, dtype=torch.float32)

    def forward(self, x, mask=False):
        bank = self.band_pass.clone().to(x.device)
        if mask:
            count = int(np.random.uniform(0, 20))
            start = random.randint(0, bank.shape[0] - count)
            bank[start:start + count] = 0
        self.filters = bank.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, self.filters, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], (2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], (2, 3), padding=(0, 1))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], (1, 3), padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.selu(self.bn1(x))
        else:
            out = x
        # The upstream implementation applies conv1 to x, not out. Keep it.
        out = self.conv1(x)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return self.mp(out)


class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        filts, dims = d_args['filts'], d_args['gat_dims']
        ratios, temperatures = d_args['pool_ratios'], d_args['temperatures']
        self.conv_time = CONV(filts[0], d_args['first_conv'], in_channels=1)
        self.first_bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(filts[1], first=True)),
            nn.Sequential(Residual_block(filts[2])),
            nn.Sequential(Residual_block(filts[3])),
            nn.Sequential(Residual_block(filts[4])),
            nn.Sequential(Residual_block(filts[4])),
            nn.Sequential(Residual_block(filts[4])))
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(dims[0], dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(dims[1], dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(dims[0], dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(dims[1], dims[1], temperature=temperatures[2])
        self.pool_S = GraphPool(ratios[0], dims[0], 0.3)
        self.pool_T = GraphPool(ratios[1], dims[0], 0.3)
        self.pool_hS1 = GraphPool(ratios[2], dims[1], 0.3)
        self.pool_hT1 = GraphPool(ratios[2], dims[1], 0.3)
        self.pool_hS2 = GraphPool(ratios[2], dims[1], 0.3)
        self.pool_hT2 = GraphPool(ratios[2], dims[1], 0.3)
        self.out_layer = nn.Linear(5 * dims[1], 2)

    def forward(self, x, Freq_aug=False):
        x = self.conv_time(x.unsqueeze(1), mask=Freq_aug).unsqueeze(1)
        x = self.selu(self.first_bn(F.max_pool2d(torch.abs(x), (3, 3))))
        e = self.encoder(x)
        e_S, _ = torch.max(torch.abs(e), dim=3)
        out_S = self.pool_S(self.GAT_layer_S(e_S.transpose(1, 2) + self.pos_S))
        e_T, _ = torch.max(torch.abs(e), dim=2)
        out_T = self.pool_T(self.GAT_layer_T(e_T.transpose(1, 2)))
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        t, s, m = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + t, out_S1 + s, master1 + m
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        t, s, m = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + t, out_S2 + s, master2 + m
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        out_T, out_S = torch.max(out_T1, out_T2), torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        last_hidden = torch.cat([T_max, torch.mean(out_T, dim=1), S_max,
                                 torch.mean(out_S, dim=1), master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        return last_hidden, self.out_layer(last_hidden)
