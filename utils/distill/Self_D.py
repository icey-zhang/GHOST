# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def _gumbel_sigmoid(
    logits, tau=1, hard=True, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # print(gumbels1)
        # gumbels2 = (
        #     -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        #     .exponential_()
        #     .log()
        # )
        # print(gumbels2)
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1) / tau #- gumbels2
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class Self_D(nn.Module):
    def __init__(self, args):
        super(Self_D, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.attention = Attention(args)

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]
        loss = self.attention(g_s, g_t)
        return sum(loss)


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim
        # self.n_t = args.n_t
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        # self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        # self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        # torch.nn.init.xavier_normal_(self.p_t)
        # torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        query, h_t_all = self.linear_trans_t(g_t)
        key, h_hat_s_all = self.linear_trans_s(g_s)
        

        # p_logit = torch.matmul(self.p_t, self.p_s.t())

        # logit = torch.add(torch.einsum('bsq,btq->bts', key, query), p_logit) / np.sqrt(self.qk_dim)
        logit = torch.einsum('bsq,btq->bts', key, query) / np.sqrt(self.qk_dim)
        # atts = F.softmax(logit, dim=2)  # b x t x s
        atts = _gumbel_sigmoid(logit,hard=True)
        # atts = F.gumbel_softmax(logit, tau=1, hard=True, eps=1e-10)
        # print(atts1.max())
        # print(atts1.min())
        # print(atts1)
        # atts = logit
        atts = torch.stack([torch.diag(atts[i,:]) for i in range(atts.shape[0])],dim=0)
        loss = []

        for i, h_t in enumerate(h_t_all):
            h_hat_s = h_hat_s_all[i]
            diff = self.cal_diff(h_hat_s, h_t, atts[:,i])
            loss.append(diff)
            # for i in range(len(h_hat_s_all)):
            #     print(np.sqrt(h_hat_s_all[i].shape[1]),np.sqrt(h_t_all[i].shape[1]))
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t).pow(2).mean(1)#.unsqueeze(1)
        diff = torch.mul(diff, att).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        # self.t = len(args.t_shapes)
        # self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)
        # self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes])
        # self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [f_s.pow(2).mean(1).view(bs, -1) for f_s in g_s]#[sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1)#.view(bs * self.s, -1)  # Bs x h
        # bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=1) for s_m in spatial_mean]
        return key, value


# class Sample(nn.Module):
#     def __init__(self, t_shape):
#         super(Sample, self).__init__()
#         t_N, t_C, t_H, t_W = t_shape
#         self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

#     def forward(self, g_s, bs):
#         g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
#         return g_s
