import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    """
    Convolution 1x1 layer.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias,
    )


def conv2x2(in_channels, out_channels, stride=1, bias=False):
    """
    Convolution 2x2 layer with padding to maintain spatial dimensions.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=stride,
        padding=1 if stride == 1 else 0,  # Padding to maintain dimensions when stride=1
        bias=bias,
    )


def conv3x3(in_channels, out_channels, stride=1, bias=False):
    """
    Convolution 3x3 layer.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
    )


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block using 1x1 convolutions.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = max(channels // reduction, 8)  # Ensure at least 8 channels

        # Keep pooling output at (1, 1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        # Use conv1x1 for both reduction and expansion
        self.conv1 = conv1x1(channels, mid_channels, bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(mid_channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return x * w.expand_as(x)


class HybridBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        conv_type="3x3",
        shortcut_type="1x1",
        drop=0.0,
    ):
        super(HybridBasicBlock, self).__init__()
        self.drop = drop

        # First convolution
        if conv_type == "1x1":
            self.conv1 = conv1x1(in_planes, planes, stride=stride)
        elif conv_type == "2x2":
            self.conv1 = conv2x2(in_planes, planes, stride=stride)
        else:  # Default to 3x3
            self.conv1 = conv3x3(in_planes, planes, stride=stride)

        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolution
        if conv_type == "1x1":
            self.conv2 = conv1x1(planes, planes, stride=1)
        elif conv_type == "2x2":
            self.conv2 = conv2x2(planes, planes, stride=1)
        else:  # Default to 3x3
            self.conv2 = conv3x3(planes, planes, stride=1)

        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut_type == "1x1":
                self.shortcut = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride=stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            elif shortcut_type == "2x2":
                self.shortcut = nn.Sequential(
                    conv2x2(in_planes, planes * self.expansion, stride=stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            else:  # Default to 3x3
                self.shortcut = nn.Sequential(
                    conv3x3(in_planes, planes * self.expansion, stride=stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )

        if self.drop > 0:
            self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)

        if self.drop > 0:
            out = self.dropout(out)

        return out


class HybridBottleneck(nn.Module):
    """
    Bottleneck block with 1x1 -> 3x3 -> 1x1 convolutions.
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        conv_type="3x3",
        shortcut_type="1x1",
        drop=0.0,
    ):
        super(HybridBottleneck, self).__init__()
        self.drop = drop

        # First 1x1 convolution for dimensionality reduction
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        # Middle convolution with the specified type
        if conv_type == "1x1":
            self.conv2 = conv1x1(planes, planes, stride=stride)
        elif conv_type == "2x2":
            self.conv2 = conv2x2(planes, planes, stride=stride)
        else:  # Default to 3x3
            self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)

        # Last 1x1 convolution for expansion
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut_type == "1x1":
                self.shortcut = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride=stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            elif shortcut_type == "2x2":
                self.shortcut = nn.Sequential(
                    conv2x2(in_planes, planes * self.expansion, stride=stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )
            else:  # Default to 3x3
                self.shortcut = nn.Sequential(
                    conv3x3(in_planes, planes * self.expansion, stride=stride),
                    nn.BatchNorm2d(planes * self.expansion),
                )

        if self.drop > 0:
            self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        if self.drop > 0:
            out = self.dropout(out)

        return out


class HybridResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        conv_types=None,
        shortcut_types=None,
        num_classes=10,
        num_channels=64,
        avg_pool_kernel_size=4,
        drop=0.1,
        squeeze_and_excitation=True,
        se_reduction=16,
    ):
        super(HybridResNet, self).__init__()

        self.in_planes = num_channels
        self.num_channels = num_channels

        # Initial convolution layer - use 3x3 for better feature extraction at input
        self.conv1 = conv3x3(3, self.num_channels)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation

        # Add SE block after initial convolution
        if self.squeeze_and_excitation:
            self.se_blocks = nn.ModuleList(
                [SEBlock(channels=self.num_channels, reduction=se_reduction)]
            )

        # Residual layers
        self.layer1 = self._make_layer(
            block,
            self.num_channels,
            num_blocks[0],
            stride=1,
            conv_type=conv_types[0],
            shortcut_type=shortcut_types[0],
        )
        self.layer2 = self._make_layer(
            block,
            self.num_channels * 2,
            num_blocks[1],
            stride=2,
            conv_type=conv_types[1],
            shortcut_type=shortcut_types[1],
        )
        self.layer3 = self._make_layer(
            block,
            self.num_channels * 4,
            num_blocks[2],
            stride=2,
            conv_type=conv_types[2],
            shortcut_type=shortcut_types[2],
        )

        # Add SE blocks after each residual layer
        if self.squeeze_and_excitation:
            self.se_blocks.append(
                SEBlock(
                    channels=self.num_channels * block.expansion, reduction=se_reduction
                )
            )
            self.se_blocks.append(
                SEBlock(
                    channels=self.num_channels * 2 * block.expansion,
                    reduction=se_reduction,
                )
            )
            self.se_blocks.append(
                SEBlock(
                    channels=self.num_channels * 4 * block.expansion,
                    reduction=se_reduction,
                )
            )

        # Final classifier layer
        final_channels = self.num_channels * 4 * block.expansion
        self.linear = nn.Linear(final_channels, num_classes)

        if self.drop:
            self.dropout = nn.Dropout(self.drop)

        self.avg_pool_kernel_size = avg_pool_kernel_size

    def _make_layer(self, block, planes, num_blocks, stride, conv_type, shortcut_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    conv_type=conv_type,
                    shortcut_type=shortcut_type,
                    drop=self.drop,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        if self.squeeze_and_excitation:
            out = self.se_blocks[0](out)

        out = self.layer1(out)
        if self.squeeze_and_excitation:
            out = self.se_blocks[1](out)

        out = self.layer2(out)
        if self.squeeze_and_excitation:
            out = self.se_blocks[2](out)

        out = self.layer3(out)
        if self.squeeze_and_excitation:
            out = self.se_blocks[3](out)

        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)

        if self.drop:
            out = self.dropout(out)

        out = self.linear(out)
        return out

    @classmethod
    def from_config(cls, config_dict: dict) -> "HybridResNet":

        # Choose block type based on config
        block = (
            HybridBottleneck
            if config_dict.get("use_bottleneck", False)
            else HybridBasicBlock
        )

        # Initialize model
        return cls(
            block=block,
            num_blocks=config_dict["num_blocks"],
            conv_types=config_dict["conv_types"],
            shortcut_types=config_dict["shortcut_types"],
            num_classes=config_dict.get("num_classes", 10),  # Default: 10 classes
            num_channels=config_dict["num_channels"],
            avg_pool_kernel_size=config_dict["avg_pool_kernel_size"],
            drop=config_dict["drop"],
            squeeze_and_excitation=config_dict["squeeze_and_excitation"],
            se_reduction=config_dict.get("se_reduction", 16),  # Default: 16
        )
