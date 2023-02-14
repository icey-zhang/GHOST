import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# ==============================蒸馏损失=============================== 
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=1):
        super(DistillKL, self).__init__()
        self.T = T

    def cal_loss(self, y_s, y_t):
        # student网络输出软化后结果
        # log_softmax与softmax没有本质的区别，只不过log_softmax会得到一个正值的loss结果。
        p_s = F.log_softmax(y_s/self.T, dim=1)

        # # teacher网络输出软化后结果
        p_t = F.softmax(y_t/self.T, dim=1)

        # 蒸馏损失采用的是KL散度损失函数
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


    def forward(self, g_s, g_t):
        if isinstance(g_s,list):
            if len(g_s)==3:
                g_s = (g_s[0],g_s[1],g_s[2])
                g_t = (g_t[0],g_t[1],g_t[2])
            elif len(g_s)==1:
                g_s = (g_s[0])
                g_t = (g_t[0])
        return sum(self.cal_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))
        
class DistillMSE(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=1):
        super(DistillMSE, self).__init__()
        self.T = T

    def cal_loss(self, y_s, y_t):
        loss = torch.nn.MSELoss()(y_s,y_t)
        return loss

    def forward(self, g_s, g_t):
        if isinstance(g_s,list):
            if len(g_s)==3:
                g_s = (g_s[0],g_s[1],g_s[2])
                g_t = (g_t[0],g_t[1],g_t[2])
            elif len(g_s)==1:
                g_s = (g_s[0])
                g_t = (g_t[0])

        return sum(self.cal_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))