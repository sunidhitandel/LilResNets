import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        conv_kernel_size=3,
        shortcut_kernel_size=1,
        drop=0.4,
    ):
        """
        Convolutional Layer kernel size Fi
        Skip connection (shortcut) kernel size Ki
        """
        super(BasicBlock, self).__init__()
        self.drop = drop

        # Pre-activation setup: BN -> GLEU -> Conv2d
        # 1st block
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=int(conv_kernel_size / 2),
            bias=False,
        )

        # 2nd block:
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=int(conv_kernel_size / 2),
            bias=False,
        )

        # Shortcut --> connect the input back to 2nd block
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=self.expansion * planes,
                    kernel_size=shortcut_kernel_size,
                    stride=stride,
                    padding=int(shortcut_kernel_size / 2),
                    bias=False,
                ),
            )

        # Spatial Dropout
        if self.drop:
            self.dropout = nn.Dropout2d(p=self.drop)

    def forward(self, x):
        shortcut = self.shortcut(x)

        # Based on pre-activation setup
        # BN -> GLEU -> Conv2d
        out = self.conv1(F.gelu(self.bn1(x)))
        out = self.conv2(F.gelu(self.bn2(out)))

        # Add shortcut
        out += shortcut

        if self.drop:
            out = self.dropout(out)

        return out


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """
    Convolution 1x1 layer.
    Parameters:
    ----------
    in_channels : Number of input channels.
    out_channels : Number of output channels.
    stride : Strides of the convolution.
    groups : Number of groups.
    bias : Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias,
    )


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : Number of channels.
    reduction : Squeeze reduction value.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_cannels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels, out_channels=mid_cannels, bias=True)
        self.activ = nn.GELU()

        self.conv2 = conv1x1(in_channels=mid_cannels, out_channels=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class PreResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        conv_kernel_sizes=None,
        shortcut_kernel_sizes=None,
        num_classes=10,
        num_channels=32,
        avg_pool_kernel_size=4,
        drop=None,
        squeeze_and_excitation=None,
    ):
        super(PreResNet, self).__init__()
        self.in_planes = num_channels
        # self.avg_pool_kernel_size = avg_pool_kernel_size
        self.avg_pool_kernel_size = int(32 / (2 ** (len(num_blocks) - 1)))

        """
        # of channels Ci
        """
        self.num_channels = num_channels
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation

        if self.squeeze_and_excitation:
            self.seblock = SEBlock(channels=self.num_channels)

        """
        # of Residual Layers N
        # of Residual Blocks Bi
        """
        self.residual_layers = []
        for n in range(len(num_blocks)):
            stride = (
                1 if n == 0 else 2
            )  # stride=1 for first residual layer, and stride=2 for the remaining layers
            conv_kernel_size = (
                conv_kernel_sizes[n] if conv_kernel_sizes else 3
            )  # setting default kernel size of block's convolutional layers
            shortcut_kernel_size = (
                shortcut_kernel_sizes[n] if shortcut_kernel_sizes else 1
            )  # setting default kernel size of block's skip connection (shortcut) layers
            self.residual_layers.append(
                self._make_layer(
                    block,
                    self.num_channels * (2**n),
                    num_blocks[n],
                    stride=stride,
                    conv_kernel_size=conv_kernel_size,
                    shortcut_kernel_size=shortcut_kernel_size,
                )
            )

        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.linear = nn.Linear(
            self.num_channels * (2**n) * block.expansion, num_classes
        )
        """
        Dropout layer
        """
        if self.drop:
            self.dropout = nn.Dropout(
                self.drop
            )  # Define proportion or neurons to dropout

    def _make_layer(
        self, block, planes, num_blocks, stride, conv_kernel_size, shortcut_kernel_size
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    conv_kernel_size,
                    shortcut_kernel_size,
                    drop=self.drop,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(F.gelu(self.bn1(x)))
        # out = F.relu(self.bn1(self.conv1(x)))
        if self.squeeze_and_excitation:
            out = self.seblock(out)
        for layer in self.residual_layers:
            out = layer(out)
        """
        Average pool kernel size
        """
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)
        if self.drop:
            out = self.dropout(out)
        out = self.linear(out)
        return out

    @classmethod
    def from_config(cls, config_dict: dict) -> "PreResNet":
        # Initialize model
        return cls(
            block=BasicBlock,
            num_blocks=config_dict["num_blocks"],
            conv_kernel_sizes=config_dict["conv_kernel_sizes"],
            shortcut_kernel_sizes=config_dict["shortcut_kernel_sizes"],
            num_channels=config_dict["num_channels"],
            avg_pool_kernel_size=config_dict["avg_pool_kernel_size"],
            drop=config_dict["drop"],
            squeeze_and_excitation=config_dict["squeeze_and_excitation"],
        )
