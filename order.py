# import cv2
# import os
# print(cv2.__version__)
#
# path = r'E:\各种版本的detr修改\output\DA\001000.jpg'
# print(os.path.exists(path))
# print(path)
# image = cv2.imread(path)
# print(image)
import torch
import torch.nn as nn

class channel_attention(nn.Module):
    def __init__(self, in_channels):
        super(channel_attention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // 8)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)     # (2,64,1,1).
        y = self.conv(y)        # (2,8,1,1).tensor([[[[-0.2176]],  [[-0.1464]],   [[0.2135]],    ]]
        y = self.bn(y)          # (2,8,1,1).tensor([[[[-0.9320]],  [[0.8075]],   [[-0.1116]],
        y = self.relu(y)        # (2,8,1,1).
        y = self.conv1(y)       # (2,64,1,1)
        y = self.sigmoid(y)     # (2,64,1,1)
        z = y.expand_as(x)
        h = x * y.expand_as(x)
        return x * y.expand_as(x)              # element-wise multiplication


class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel // gamma, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // gamma, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.b = b

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * (self.b + y)

class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()

        self.conv_row1a = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.conv_row1b = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.conv_row2b = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False)
        self.conv_out = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()   # （2,64,32,32）

        # 计算 row1a、row1b 和 g
        row1a = self.conv_row1a(x)      # （2,8,32,32）
        row1b = self.conv_row1b(x)      # （2,8,32,32）
        row2b = self.conv_row2b(x)      # （2,32,32,32）

        # 重塑维度
        row1a = row1a.view(b, c // 8, h * w).permute(0, 2, 1)       # （2,8,1024）--->（2,1024,8）
        row1b = row1b.view(b, c // 8, h * w)                        # （2,8,1024）
        row2b = row2b.view(b, c // 2, h * w).permute(0, 2, 1)       # (2, 32, 1024)  ---> (2, 1024, 32)

        # 计算关联强度矩阵
        attention = torch.matmul(row1a, row1b)  # （2,1024,1024）
        attention = self.softmax(attention)      # （2,1024,1024）

        # 加权融合
        out = torch.matmul(attention, row2b)         # （2,1024,32）
        out = out.permute(0, 2, 1).contiguous()     #（2,32,1024) contiguous() 是一个用于返回一个具有相同数据但存储方式可能不同的张量的方法,就是让转置后的张量变连续
        out = out.view(b, c // 2, h, w)             # （2,32,32,32)  ,又回到下面那一行去了的初始情况去了
        out = self.conv_out(out)                    # （2,64,32,32)  通过一个卷积回归初始情况

        return out

class DA1net(nn.Module):
    def __init__(self, in_channels):
        super(DA1net, self).__init__()

        self.position_attention = PositionAttentionModule(in_channels)
        self.channel_attention = channel_attention(in_channels)

    def forward(self, x):
        out_position = self.position_attention(x)
        out_channel = self.channel_attention(x)

        out = out_position + out_channel

        return out
# 随机生成输入数据
# input_data = torch.rand((2, 64, 32, 32))
# # 假设输入通道数为64，空间维度为32x32
#
# # 创建DANet模型
# danet_model = PositionAttentionModule(in_channels=64)
# # danet_model = ECABlock(channel=64)
# # danet_model = DA1Net(in_channels=64)
# # danet_model = channel_attention(in_channels=64)
#
# # 执行前向传播
# output = danet_model(input_data)
#
# # 打印输出结果
# print("Input Shape:", input_data.shape)
# print("Output Shape:", output.shape)



# import torch
#
# # 创建两个矩阵
# matrix1 = torch.tensor([[1, 2], [3, 4]])
# matrix2 = torch.tensor([[5, 6], [7, 8]])
#
# # 使用 torch.matmul 进行矩阵乘法
# result = torch.matmul(matrix1, matrix2)
#
# # 打印结果
# print(result)

# !/usr/bin/python3

# for i  in range(5):
#     print(i)
#     if (i == 2):
#         break
#