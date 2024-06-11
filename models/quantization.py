from ast import arg
import torch
import torch.nn as nn
from torch.autograd import Function

import copy
# from .quant_utils.quant_modules import *
from .quant_modules_dorefa import *
# from .quant_utils.quant_modules_uniform import *
# import torch
# import time
# import math
# import numpy as np
import torch.nn.functional as F
# from torch.nn import Module, Parameter
# from .quant_utils import *
# import sys
# def uniform_Q_fn(k):
#     class Uniform_Quantize(torch.autograd.Function):

#         @staticmethod
#         def forward(self, input):
#             if k == 32:
#                 out = input
#             elif k == 1:
#                 out = torch.sign(input)
#             else:
#                 n = float(2 ** k  - 1)
#                 out = torch.round(input * n) / n  # 只有八种离散结果 0, 1/7, ......, 7/7
#             return out
        
#         @staticmethod
#         def backward(self, grad_output):
#             grad_input = grad_output.clone()
#             return grad_input
    
#     return Uniform_Quantize().apply



# class Qout_Activation_Quantize(nn.Module):
#     def __init__(self, bn_bit):
#         super(Qout_Activation_Quantize, self).__init__()
#         assert bn_bit <= 31 or bn_bit == 32
#         self.bn_bit = bn_bit
#         # self.uniform_Q = Uniform_Quantize(bn_bit-1) #这块减一
#     def forward(self, x):
#         if self.bn_bit == 32:
#             bn_Q = x
#         else:
#             bn_Q = Uniform_Quantize.apply(torch.clamp(x, -1, 1),self.bn_bit)
#             # print(np.unique(bn_Q.detach().numpy()))
#         return bn_Q


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None



# def quantize_model(model, args, flag):
#     """
#     Recursively quantize a pretrained single-precision model to int8 quantized model
#     model: pretrained single-precision model
#     """
#     # flag = 0
#     # quantize convolutional and linear layers to 8-bit
#     if type(model) == nn.Conv2d:
#         if flag == 1: #第一个卷积不做量化
#             flag = flag + 1
#             return model,flag
#         else:
#             flag = flag + 1
#             quant_mod = Quant_Conv2d(weight_bit=args.weight_bit)
#             quant_mod.set_param(model)
#             return quant_mod,flag
#     elif type(model) == nn.Linear:
#         quant_mod = Quant_Linear(weight_bit=args.weight_bit)
#         quant_mod.set_param(model)
#         return quant_mod,flag

#     # quantize all the activation to 8-bit
#     elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.SiLU: ##zjq
#         if flag == 1:
#             return model,flag
#         else:
#             return nn.Sequential(*[model, QuantAct(activation_bit=args.activation_bit)]),flag

#     # recursively use the quantized module to replace the single-precision module
#     elif type(model) == nn.Sequential:
#         mods = []
#         for n, m in model.named_children():
#             mo, flag = quantize_model(m, args, flag)
#             mods.append(mo)
#         return nn.Sequential(*mods),flag
#     else:
#         q_model = copy.deepcopy(model)
#         for attr in dir(model):
#             mod = getattr(model, attr)
#             # if 'model_up' in attr:
#             #     print('a')
#             # if 'SAM' in attr:
#             #     print('b')
#             if isinstance(mod, nn.Module) and 'norm' not in attr and 'model_up' not in attr and 'b1' not in attr and 'b2' not in attr and 'b3' not in attr and 'b4' not in attr and 'rb' not in attr and 'softmax' not in attr and 'bottleneck' not in attr: #and 'SAM' not in attr: #SAM,超分不做量化
#                 mo, flag = quantize_model(mod, args, flag)
#                 setattr(q_model, attr, mo)

#         return q_model,flag

def quantize_model_relu(model, args, flag):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # flag = 0
    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        if flag == 2: #第一个卷积不做量化
            flag = flag + 1
            return model,flag
        else:
            flag = flag + 1
            quant_mod = Quant_Conv2d(weight_bit=args.weight_bit)
            quant_mod.set_param(model)
            return quant_mod,flag
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=args.weight_bit)
        quant_mod.set_param(model)
        return quant_mod,flag

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.SiLU: ##zjq
        # if type(model) == nn.SiLU:
        #     model = nn.ReLU() #替换激活函数
        if flag == 2:
            return model,flag
        else:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=args.activation_bit)]),flag

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mo, flag = quantize_model_relu(m, args, flag)
            mods.append(mo)
        return nn.Sequential(*mods),flag
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            # if 'model_up' in attr:
            #     print('a')
            # if 'SAM' in attr:
            #     print('b')
            if isinstance(mod, nn.Module) and 'model_up' not in attr and 'b1' not in attr and 'b2' not in attr and 'b3' not in attr and 'b4' not in attr and 'b5' not in attr and 'b6' not in attr and 'b7' not in attr:
                #and 'rb' not in attr and 'softmax' not in attr and 'bottleneck' not in attr: #and 'SAM' not in attr: #SAM,超分不做量化
                # and 'model_up' not in attr #and 'norm' not in attr
                mo, flag = quantize_model_relu(mod, args, flag)
                setattr(q_model, attr, mo)

        return q_model,flag

def quantize_model_relu_noflag(model, args):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # flag = 0
    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=args.weight_bit)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=args.weight_bit)
        quant_mod.set_param(model)
        return quant_mod

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.SiLU: ##zjq
        # if type(model) == nn.SiLU:
        #     model = nn.ReLU() #替换激活函数
        return nn.Sequential(*[model, QuantAct_relu(activation_bit=args.activation_bit)])

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mo = quantize_model_relu_noflag(m, args)
            mods.append(mo)
        return nn.Sequential(*mods)
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            # if 'model_up' in attr:
            #     print('a')
            # if 'SAM' in attr:
            #     print('b')
            if isinstance(mod, nn.Module) and 'model_up' not in attr and 'b1' not in attr and 'b2' not in attr and 'b3' not in attr and 'b4' not in attr and 'b5' not in attr and 'b6' not in attr and 'b7' not in attr:
                #and 'rb' not in attr and 'softmax' not in attr and 'bottleneck' not in attr: #and 'SAM' not in attr: #SAM,超分不做量化
                # and 'model_up' not in attr #and 'norm' not in attr
                mo = quantize_model_relu_noflag(mod, args)
                setattr(q_model, attr, mo)

        return q_model


def quantize_model_relu_mix(model, args, flag, ac_flag):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # flag = 0
    # quantize convolutional and linear layers to 8-bit
    if isinstance(args.weight_bit,list):
        firstbit = args.weight_bit[0]
        secondbit = args.weight_bit[1]
        thirdbit = args.weight_bit[2]
    else:
        firstbit = args.weight_bit
        secondbit = args.weight_bit/2
        thirdbit = args.weight_bit/4
    if type(model) == nn.Conv2d:
        # print("model_w:",model)
        flag = flag + 1
        if flag == 1: #第一个卷积不做量化
            if args.firstlayer_quan == False:
                return model, flag, ac_flag
            else:
                quant_mod = Quant_Conv2d(weight_bit=firstbit)
                quant_mod.set_param(model)
                return quant_mod, flag, ac_flag
        elif flag > 1 and flag <= 20:
            quant_mod = Quant_Conv2d(weight_bit=firstbit)
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag
        elif flag > 20 and flag <= 40:
            quant_mod = Quant_Conv2d(weight_bit=secondbit)
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag
        else: 
            quant_mod = Quant_Conv2d(weight_bit=thirdbit)
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=firstbit)
        quant_mod.set_param(model)
        return quant_mod, flag, ac_flag

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.SiLU: ##zjq
        # print("model_ac:",model)
        # if type(model) == nn.SiLU:
        #     model = nn.ReLU() #替换激活函数
        ac_flag = ac_flag + 1
        if ac_flag == 1:
            if args.firstlayer_quan == False:
                return model, flag, ac_flag
            else:
                return nn.Sequential(*[model, QuantAct_relu(activation_bit=firstbit)]), flag, ac_flag
        elif ac_flag > 1 and ac_flag <= 20:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=firstbit)]), flag, ac_flag
        elif ac_flag > 20 and ac_flag <= 40:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=secondbit)]), flag, ac_flag
        else:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=thirdbit)]), flag, ac_flag

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        # for n, m in model.named_children():
        #     print(m)
        for n, m in model.named_children():
            # print("m:",m)
            mo, flag, ac_flag = quantize_model_relu_mix(m, args, flag, ac_flag)
            mods.append(mo)
        return nn.Sequential(*mods), flag, ac_flag
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            # if 'model_up' in attr:
            #     print('a')
            # if 'SAM' in attr:
            #     print('b')
            if isinstance(mod, nn.Module) and 'model_up' not in attr:# and 'b1' not in attr and 'b2' not in attr and 'b3' not in attr and 'b4' not in attr and 'b5' not in attr and 'b6' not in attr and 'b7' not in attr:
                #and 'rb' not in attr and 'softmax' not in attr and 'bottleneck' not in attr: #and 'SAM' not in attr: #SAM,超分不做量化
                # and 'model_up' not in attr #and 'norm' not in attr
                # print("mod:",mod)
                mo, flag, ac_flag = quantize_model_relu_mix(mod, args, flag, ac_flag)
                setattr(q_model, attr, mo)

        return q_model, flag, ac_flag

def quantize_model_relu_mixv2(model, args, flag, ac_flag):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # flag = 0
    # quantize convolutional and linear layers to 8-bit
    if isinstance(args.weight_bit,list):
        firstbit = args.weight_bit[0]
        secondbit = args.weight_bit[1]
        thirdbit = args.weight_bit[2]
    else:
        firstbit = args.weight_bit
        secondbit = args.weight_bit/2
        thirdbit = args.weight_bit/4
    if type(model) == nn.Conv2d:
        # print("model_w:",model)
        flag = flag + 1
        if flag == 1: #第一个卷积不做量化
            if args.firstlayer_quan == False:
                return model, flag, ac_flag
            else:
                quant_mod = Quant_Conv2d(weight_bit=firstbit)
                quant_mod.set_param(model)
                return quant_mod, flag, ac_flag
        elif flag > 1 and flag <= 30:
            quant_mod = Quant_Conv2d(weight_bit=firstbit)
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag
        elif flag > 30 and flag <= 50:
            quant_mod = Quant_Conv2d(weight_bit=secondbit)
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag
        else: 
            quant_mod = Quant_Conv2d(weight_bit=thirdbit)
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag
    elif type(model) == nn.Linear:
        quant_mod = Quant_Linear(weight_bit=firstbit)
        quant_mod.set_param(model)
        return quant_mod, flag, ac_flag

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.SiLU: ##zjq
        # print("model_ac:",model)
        # if type(model) == nn.SiLU:
        #     model = nn.ReLU() #替换激活函数
        ac_flag = ac_flag + 1
        if ac_flag == 1:
            if args.firstlayer_quan == False:
                return model, flag, ac_flag
            else:
                return nn.Sequential(*[model, QuantAct_relu(activation_bit=firstbit)]), flag, ac_flag
        elif ac_flag > 1 and ac_flag <= 30:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=firstbit)]), flag, ac_flag
        elif ac_flag > 30 and ac_flag <= 50:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=secondbit)]), flag, ac_flag
        else:
            return nn.Sequential(*[model, QuantAct_relu(activation_bit=thirdbit)]), flag, ac_flag

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        # for n, m in model.named_children():
        #     print(m)
        for n, m in model.named_children():
            # print("m:",m)
            mo, flag, ac_flag = quantize_model_relu_mixv2(m, args, flag, ac_flag)
            mods.append(mo)
        return nn.Sequential(*mods), flag, ac_flag
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            # if 'model_up' in attr:
            #     print('a')
            # if 'SAM' in attr:
            #     print('b')
            if isinstance(mod, nn.Module) and 'model_up' not in attr:# and 'b1' not in attr and 'b2' not in attr and 'b3' not in attr and 'b4' not in attr and 'b5' not in attr and 'b6' not in attr and 'b7' not in attr:
                #and 'rb' not in attr and 'softmax' not in attr and 'bottleneck' not in attr: #and 'SAM' not in attr: #SAM,超分不做量化
                # and 'model_up' not in attr #and 'norm' not in attr
                # print("mod:",mod)
                mo, flag, ac_flag = quantize_model_relu_mixv2(mod, args, flag, ac_flag)
                setattr(q_model, attr, mo)

        return q_model, flag, ac_flag

def quantize_model_relu_automix(model, args, flag, ac_flag):
    """
    Recursively quantize a pretrained single-precision model to quantized model
    model: pretrained single-precision model
    """
    if type(model) == nn.Conv2d:
        flag = flag + 1
        if flag == 1: #第一个卷积不做量化
            if args.firstlayer_quan == False:
                return model, flag, ac_flag
            else:
                quant_mod = Quant_Conv2d(weight_bit=args.weight_bit[flag-1])
                quant_mod.set_param(model)
                return quant_mod, flag, ac_flag
        else:
            quant_mod = Quant_Conv2d(weight_bit=args.weight_bit[flag-1])
            quant_mod.set_param(model)
            return quant_mod, flag, ac_flag

    # elif type(model) == nn.Linear:
    #     quant_mod = Quant_Linear(weight_bit=firstbit)
    #     quant_mod.set_param(model)
    #     return quant_mod, flag, ac_flag

    elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.SiLU: ##zjq
        ac_flag = ac_flag + 1
        if ac_flag == 1:
            if args.firstlayer_quan == False:
                return model, flag, ac_flag
            else:
                return nn.Sequential(*[model, QuantAct_relu(activation_bit=args.weight_bit[ac_flag-1])]), flag, ac_flag
        else:
            return nn.Sequential(*[model,  QuantAct_relu(activation_bit=args.weight_bit[ac_flag-1])]), flag, ac_flag

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mo, flag, ac_flag = quantize_model_relu_automix(m, args, flag, ac_flag)
            mods.append(mo)
        return nn.Sequential(*mods), flag, ac_flag
    else:
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'model_up' not in attr:
                mo, flag, ac_flag = quantize_model_relu_automix(mod, args, flag, ac_flag)
                setattr(q_model, attr, mo)

        return q_model, flag, ac_flag

