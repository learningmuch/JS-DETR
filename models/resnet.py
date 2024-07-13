from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from .SCConv import ScConv


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class SCconv_Bottleneck(nn.Module):
#     expansion = 4  # 一个残差结构中的通道数变化，ResNet18和ResNet34应该是1，即不变。50及后面的都是4。
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
#         super(SCconv_Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=1, stride=1, bias=False)  # squeeze channels
#         self.bn1 = norm_layer(out_channel)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = norm_layer(out_channel)
#         # -----------------------------------------
#         self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
#                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
#         self.sc_2 = ScConv(out_channel * self.expansion)
#         self.bn3 = norm_layer(out_channel * self.expansion)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)  # x:(4,1024,50,50)      out:(4,512,50,50)
#         out = self.bn1(out)  # out:(4,512,50,50)
#         out = self.relu(out)  # out:(4,512,50,50)
#         # print(out.shape)
#         out = self.conv2(out)  # out:(4,512,25,25)
#         # print(out.shape)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)  # out:(4,2048,25,25)
#         out = self.sc_2(out)  # out:(4,2048,25,25)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class SCconv_Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.sc_2 = ScConv(planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.sc_2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DANet(nn.Module):
    def __init__(self, in_channels):
        super(DANet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1) # 自适应平均池化，将空间维度降为1x1
        self.conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False) # 第一个全连接层，用于特征变换
        self.bn = nn.BatchNorm2d(in_channels // 8)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, bias=False)# 第二个全连接层，将特征变换后的通道数恢复为原始通道数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # 计算通道维度的特征
        y = self.avgpool(x)
        y = self.conv(y)
        y = self.bn(y)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()

        self.conv_theta = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.conv_phi = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, bias=False)
        self.conv_g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False)
        self.conv_out = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()

        # 计算 theta、phi 和 g
        theta = self.conv_theta(x)
        phi = self.conv_phi(x)
        g = self.conv_g(x)

        # 重塑维度
        theta = theta.view(b, c // 8, h * w).permute(0, 2, 1)
        phi = phi.view(b, c // 8, h * w)
        g = g.view(b, c // 2, h * w).permute(0, 2, 1)

        # 计算关联强度矩阵
        attention = torch.matmul(theta, phi)
        attention = self.softmax(attention)

        # 加权融合
        out = torch.matmul(attention, g)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(b, c // 2, h, w)
        out = self.conv_out(out)

        return out


class DA1Net(nn.Module):
    def __init__(self, in_channels):
        super(DA1Net, self).__init__()

        self.position_attention = PositionAttentionModule(in_channels)
        self.channel_attention = DANet(in_channels)

    def forward(self, x):
        out_position = self.position_attention(x)
        out_channel = self.channel_attention(x)

        out = out_position + out_channel

        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], scconv=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        scconv: bool = False,
    ) -> nn.Sequential:
        if scconv == True:
            block = SCconv_Bottleneck
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



class ResNet_DA1(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], scconv=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        # 将DANet模块与ResNet50集成
        self.danet = DA1Net(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化DANet模块的所有参数 ,这样在修改resnet网络结构之后仍然可以用之前的预训练权重进行训练
        for m in self.modules():
            if isinstance(m, DANet):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(mm, nn.BatchNorm2d):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)
        # 网络中的权重参数进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        scconv : bool = False
    ) -> nn.Sequential:
        if scconv == True:
            block = SCconv_Bottleneck
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if scconv:
            layers.append(
                SCconv_Bottleneck(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                    norm_layer
                )
            )
        else:
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
                )
            )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 将DANet模块与ResNet50集成
        x = self.danet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def resnet50(num_classes=1000, pretrain_path=''):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrain_path != '':
        model.load_state_dict(torch.load(pretrain_path), strict=False)
    return model


def resnet50_da1(num_classes=1000, pretrain_path=''):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth

    model = ResNet_DA1(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrain_path != '':
        model.load_state_dict(torch.load(pretrain_path), strict=False)
    return model