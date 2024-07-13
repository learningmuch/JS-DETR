# import torch
# import torch.nn.functional as F
# import torch.nn as nn
#
#
# class GroupNorm2d(nn.Module):
#
#     def __init__(self, n_groups: int = 16, n_channels: int = 16, eps: float = 1e-10):
#         super(GroupNorm2d, self).__init__()
#         assert n_channels % n_groups == 0
#         self.n_groups = n_groups
#         self.gamma = nn.Parameter(torch.randn(n_channels, 1, 1))  # learnable gamma
#         self.beta = nn.Parameter(torch.zeros(n_channels, 1, 1))  # learnable beta
#         self.eps = eps
#
#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = x.reshape(N, self.n_groups, -1)
#         mean = x.mean(dim=2, keepdim=True)
#         std = x.std(dim=2, keepdim=True)
#         x = (x - mean) / (std + self.eps)
#         x = x.reshape(N, C, H, W)
#         return x * self.gamma + self.beta
#
#     # Spatial and Reconstruct Unit
#
#
# class SRU(nn.Module):
#
#     def __init__(
#             self,
#             n_channels: int,  # in_channels
#             n_groups: int = 16,  # 16
#             gate_treshold: float = 0.5,  # 0.5
#     ):
#         super().__init__()
#
#         # initialize GroupNorm2d
#         self.gn = GroupNorm2d(n_groups=n_groups, n_channels=n_channels)
#         self.gate_treshold = gate_treshold
#         self.sigomid = nn.Sigmoid()
#
#     def forward(self, x):
#         gn_x = self.gn(x)
#         w_gamma = self.gn.gamma / sum(self.gn.gamma)  # cal gamma weight
#         reweights = self.sigomid(gn_x * w_gamma)  # importance
#
#         info_mask = reweights >= self.gate_treshold
#         noninfo_mask = reweights < self.gate_treshold
#         x_1 = info_mask * x
#         x_2 = noninfo_mask * x
#         x = self.reconstruct(x_1, x_2)
#         return x
#
#     def reconstruct(self, x_1, x_2):
#         x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
#         x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
#         return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
#
#
# # Channel Reduction Unit
# class CRU(nn.Module):
#
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, alpha: float = 1 / 2,
#                  squeeze_radio: int = 2, groups: int = 2):
#         super().__init__()
#
#         self.up_channel = up_channel = int(alpha * in_channels)
#         self.low_channel = low_channel = in_channels - up_channel
#         self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
#         self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
#
#         self.GWC = nn.Conv2d(up_channel // squeeze_radio, out_channels, kernel_size=kernel_size, stride=1,
#                              padding=kernel_size // 2, groups=groups)
#         self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, out_channels, kernel_size=1, bias=False)
#
#         if in_channels == out_channels:
#             self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, out_channels - low_channel // squeeze_radio, kernel_size=1,
#                                   bias=False)
#         else:
#             self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, in_channels - out_channels,
#                                   kernel_size=1, bias=False)
#
#         self.pool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
#         up, low = self.squeeze1(up), self.squeeze2(low)
#
#         y1 = self.GWC(up) + self.PWC1(up)
#
#         y2 = torch.cat([self.PWC2(low), low], dim=1)
#
#         s1 = self.pool(y1)
#         s2 = self.pool(y2)
#         s = torch.cat([s1, s2], dim=1)
#         beta = F.softmax(s, dim=1)
#         beta1, beta2 = torch.split(beta, beta.size(1) // 2, dim=1)
#         y = beta1 * y1 + beta2 * y2
#         return y
#
#
# # Squeeze and Channel Reduction Convolution
# class ScConv(nn.Module):
#
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
#                  n_groups: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2, squeeze_radio: int = 2,
#                  groups: int = 16):
#         super().__init__()
#
#         self.SRU = SRU(in_channels, n_groups=n_groups, gate_treshold=gate_treshold)
#         self.CRU = CRU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, alpha=alpha,
#                        squeeze_radio=squeeze_radio, groups=groups)
#
#     def forward(self, x):
#         x = self.SRU(x)
#         x = self.CRU(x)
#         return x
#
#
# if __name__ == '__main__':
#     from thop import profile
#
#     x1 = torch.randn(16, 96, 224, 224)
#     x2 = torch.randn(16, 96, 224, 224)
#     conv2d_model = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3))
#     flops1, params1 = profile(conv2d_model, (x1,))
#     scconv_model = ScConv(96, 64, kernel_size=3, alpha=1 / 2,
#                           squeeze_radio=2)  # out_channels > in_channels * (1-alpha) / squeeze_radio
#     flops2, params2 = profile(scconv_model, (x2,))
#     print(f'model:["Conv2d"], FLOPS: [{flops1}], Params: [{params1}]')
#     print(f'model:["scConv"], FLOPS: [{flops2}], Params: [{params2}]')


'''
Description:
Date: 2023-07-21 14:36:27
LastEditTime: 2023-07-27 18:41:47
FilePath: /chengdongzhou/ScConv.py
'''
# import torch
# import torch.nn.functional as F
# import torch.nn as nn


# class GroupBatchnorm2d(nn.Module):
#     def __init__(self, c_num: int,
#                  group_num: int = 16,
#                  eps: float = 1e-10
#                  ):
#         super(GroupBatchnorm2d, self).__init__()
#         assert c_num >= group_num
#         self.group_num = group_num
#         self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
#         self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
#         self.eps = eps
#
#     def forward(self, x):
#         N, C, H, W = x.size()
#         x = x.view(N, self.group_num, -1)
#         mean = x.mean(dim=2, keepdim=True)
#         std = x.std(dim=2, keepdim=True)
#         x = (x - mean) / (std + self.eps)
#         x = x.view(N, C, H, W)
#         return x * self.weight + self.bias
#
#
# class SRU(nn.Module):
#     def __init__(self,
#                  oup_channels: int,
#                  group_num: int = 16,
#                  gate_treshold: float = 0.5,
#                  torch_gn: bool = True
#                  ):
#         super().__init__()
#
#         self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
#             c_num=oup_channels, group_num=group_num)
#         self.gate_treshold = gate_treshold
#         self.sigomid = nn.Sigmoid()
#
#     def forward(self, x):
#         gn_x = self.gn(x)
#         w_gamma = self.gn.weight / sum(self.gn.weight)
#         w_gamma = w_gamma.view(1, -1, 1, 1)
#         reweigts = self.sigomid(gn_x * w_gamma)
#         # Gate
#         w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
#         w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
#         x_1 = w1 * x
#         x_2 = w2 * x
#         y = self.reconstruct(x_1, x_2)
#         return y
#
#     def reconstruct(self, x_1, x_2):
#         x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
#         x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
#         return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)
#
#
# class CRU(nn.Module):
#     '''
#     alpha: 0<alpha<1
#     '''
#
#     def __init__(self,
#                  op_channel: int,
#                  alpha: float = 1 / 2,
#                  squeeze_radio: int = 2,
#                  group_size: int = 2,
#                  group_kernel_size: int = 3,
#                  ):
#         super().__init__()
#         self.up_channel = up_channel = int(alpha * op_channel)
#         self.low_channel = low_channel = op_channel - up_channel
#         self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
#         self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
#         # up
#         self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
#                              padding=group_kernel_size // 2, groups=group_size)
#         self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
#         # low
#         self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
#                               bias=False)
#         self.advavg = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         # Split
#         up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
#         up, low = self.squeeze1(up), self.squeeze2(low)
#
#         y1 = self.GWC(up) + self.PWC1(up)
#
#         y2 = torch.cat([self.PWC2(low), low], dim=1)
#
#         s1 = self.advavg(y1)
#         s2 = self.advavg(y2)
#         s = torch.cat([s1, s2], dim=1)
#         beta = F.softmax(s, dim=1)
#         beta1, beta2 = torch.split(beta, beta.size(1) // 2, dim=1)
#         y = beta1 * y1 + beta2 * y2
#         return y
#
#
# class ScConv(nn.Module):
#     def __init__(self,
#                  op_channel: int,
#                  group_num: int = 4,
#                  gate_treshold: float = 0.5,
#                  alpha: float = 1 / 2,
#                  squeeze_radio: int = 2,
#                  group_size: int = 2,
#                  group_kernel_size: int = 3,
#                  ):
#         super().__init__()
#         self.SRU = SRU(op_channel,
#                        group_num=group_num,
#                        gate_treshold=gate_treshold)
#         self.CRU = CRU(op_channel,
#                        alpha=alpha,
#                        squeeze_radio=squeeze_radio,
#                        group_size=group_size,
#                        group_kernel_size=group_kernel_size)
#
#     def forward(self, x):
#         x = self.SRU(x)
#         x = self.CRU(x)
#         return x
import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


if __name__ == '__main__':
    from thop import profile

    x1 = torch.randn(16, 64, 224, 224)
    x2 = torch.randn(16, 64, 224, 224)
    conv2d_model = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3))
    flops1, params1 = profile(conv2d_model, (x1,))
    scconv_model = ScConv(64, 2)  # out_channels > in_channels * (1-alpha) / squeeze_radio
    flops2, params2 = profile(scconv_model, (x2,))
    print(f'model:["Conv2d"], FLOPS: [{flops1}], Params: [{params1}]')
    print(f'model:["scConv"], FLOPS: [{flops2}], Params: [{params2}]')
