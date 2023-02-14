import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# def uniform_Q_fn(k):
class Uniform_Quantize(Function):
    # def __init__(self, k):
    #     super(Uniform_Quantize, self).__init__()
    #     self.k = k
    @staticmethod
    def forward(ctx,input,k):
        if k == 32:
            out = input
        elif k == 1:
            out = torch.sign(input)
        else:
            n = float(2 ** k  - 1)
            out = torch.round(input * n) / n  # 只有八种离散结果 0, 1/7, ......, 7/7
        return out
    
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_output.clone()
        return grad_input,None
    
    # return Uniform_Quantize().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = Uniform_Quantize.apply

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E,self.w_bit) * E
    else:
      weight = torch.tanh(x) #[-1,1]
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = 2 * self.uniform_q(weight,self.w_bit) - 1 #论文中没有乘以max_weight
      #weight_q = max_w * (2 * self.uniform_q(weight,self.w_bit) - 1)
    #   print('weight_q',weight_q.max(),weight_q.min())
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = Uniform_Quantize.apply

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1),self.a_bit) #激活用relu
    #   activation_q = self.uniform_q(torch.clamp(x, -1, 1),self.a_bit-1)
    #   activation_q = self.uniform_q(x,self.a_bit)

      # print(np.unique(activation_q.detach().numpy()))
    return activation_q

class activation_quantize_fn_relu(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn_relu, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = Uniform_Quantize.apply

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1),self.a_bit) #激活用relu
    #   activation_q = self.uniform_q(torch.clamp(x, -1, 1),self.a_bit-1)
    #   activation_q = self.uniform_q(x,self.a_bit)

      # print(np.unique(activation_q.detach().numpy()))
    return activation_q

class QuantAct(nn.Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 ): #running_stat=True
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        self.full_precision_flag = full_precision_flag
        # self.running_stat = running_stat
        # self.register_buffer('x_min', torch.zeros(1).cuda())
        # self.register_buffer('x_max', torch.zeros(1).cuda())
        self.act_function = activation_quantize_fn(activation_bit)#Uniform_Quantize.apply
        # print('activation_bit',activation_bit)

    # def __repr__(self):
    #     return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
    #         self.__class__.__name__, self.activation_bit,
    #         self.full_precision_flag, self.running_stat, self.x_min.item(),
    #         self.x_max.item())

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag)

    # def fix(self):
    #     """
    #     fix the activation range by setting running stat
    #     """
    #     self.running_stat = False

    def forward(self, x):
        """
        quantize given activation x
        """
        # if self.running_stat:
        #     x_min = x.data.min()
        #     x_max = x.data.max()
        #     # print('activation111',x_min,x_max)
        #     # print('activation222',self.x_min,self.x_max)
        #     # in-place operation used on multi-gpus
        #     self.x_min += -self.x_min + min(self.x_min, x_min)
        #     self.x_max += -self.x_max + max(self.x_max, x_max)
            # print('activation',self.x_min,self.x_max)

        if not self.full_precision_flag:
            quant_act = self.act_function(x)#, self.x_min,self.x_max)
            # print('quant_act',quant_act.max(),quant_act.min())
            return quant_act
        else:
            return x

class QuantAct_relu(nn.Module):
    """
    Class to quantize given activations
    """
    def __init__(self,
                 activation_bit,
                 full_precision_flag=False,
                 ): #running_stat=True
        """
        activation_bit: bit-setting for activation
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantAct_relu, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        self.full_precision_flag = full_precision_flag
        # self.running_stat = running_stat
        # self.register_buffer('x_min', torch.zeros(1).cuda())
        # self.register_buffer('x_max', torch.zeros(1).cuda())
        self.act_function = activation_quantize_fn_relu(activation_bit)#Uniform_Quantize.apply
        # print('activation_bit',activation_bit)

    # def __repr__(self):
    #     return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
    #         self.__class__.__name__, self.activation_bit,
    #         self.full_precision_flag, self.running_stat, self.x_min.item(),
    #         self.x_max.item())

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag)

    # def fix(self):
    #     """
    #     fix the activation range by setting running stat
    #     """
    #     self.running_stat = False

    def forward(self, x):
        """
        quantize given activation x
        """
        # if self.running_stat:
        #     x_min = x.data.min()
        #     x_max = x.data.max()
        #     # print('activation111',x_min,x_max)
        #     # print('activation222',self.x_min,self.x_max)
        #     # in-place operation used on multi-gpus
        #     self.x_min += -self.x_min + min(self.x_min, x_min)
        #     self.x_max += -self.x_max + max(self.x_max, x_max)
            # print('activation',self.x_min,self.x_max)

        if not self.full_precision_flag:
            quant_act = self.act_function(x)#, self.x_min,self.x_max)
            # print('quant_act',quant_act.max(),quant_act.min())
            return quant_act
        else:
            return x

class Quant_Linear(nn.Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or froze
        """
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.weight_function = Uniform_Quantize.apply

    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        try:
            self.bias = nn.Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        # w = self.weight
        # x_transform = w.data.detach()
        # w_min = x_transform.min(dim=1).values
        # w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias)


class Quant_Conv2d(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        # print('weight_bit',weight_bit)
        self.weight_function = weight_quantize_fn(weight_bit)#Uniform_Quantize.apply

    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data.clone())
        try:
            self.bias = nn.Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        # w = self.weight
        # x_transform = w.data.contiguous().view(self.out_channels, -1)
        # w_min = x_transform.min(dim=1).values
        # w_max = x_transform.max(dim=1).values
        if not self.full_precision_flag:
            w = self.weight_function(self.weight)
        else:
            w = self.weight
        # print(x.min(),x.max())

        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)