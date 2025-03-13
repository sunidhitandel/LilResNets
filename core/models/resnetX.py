import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias,
    )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels, out_channels=mid_channels, bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(in_channels=mid_channels, out_channels=channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return x * w


class ResNeXtBlock(nn.Module):
    expansion = 4  # Like ResNet-50+, output channels expand by 4

    def __init__(
        self,
        in_planes,
        planes,
        cardinality=32,  # Number of parallel paths
        stride=1,
        conv_kernel_size=3,
        shortcut_kernel_size=1,
        drop=0.0,
    ):
        super(ResNeXtBlock, self).__init__()
        self.drop = drop
        self.cardinality = cardinality

        # Width per group (e.g., 4 channels per path)
        width = planes  # Intermediate planes before expansion

        # 1x1 conv to reduce channels
        self.conv1 = conv1x1(in_planes, width, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3x3 conv with cardinality (grouped convolution)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=int(conv_kernel_size / 2),
            groups=cardinality,  # Split into cardinality groups
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)

        # 1x1 conv to expand channels
        self.conv3 = conv1x1(width, planes * self.expansion, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(
                    in_planes,
                    planes * self.expansion,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        # Main path: 1x1 → 3x3 (grouped) → 1x1
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Add shortcut
        out += self.shortcut(x)
        out = F.relu(out)

        if self.drop > 0:
            out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        conv_kernel_sizes=None,
        shortcut_kernel_sizes=None,
        num_classes=10,
        num_channels=32,
        avg_pool_kernel_size=None,
        drop=None,
        squeeze_and_excitation=None,
        cardinality=1,
    ):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        self.avg_pool_kernel_size = (
            avg_pool_kernel_size
            if avg_pool_kernel_size
            else int(32 / (2 ** (len(num_blocks) - 1)))
        )

        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(
            3, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation
        self.cardinality = cardinality
        if self.squeeze_and_excitation:
            self.seblock = SEBlock(channels=self.num_channels)

        # Build residual layers
        self.residual_layers = []
        for n in range(len(num_blocks)):
            stride = 1 if n == 0 else 2
            conv_kernel_size = conv_kernel_sizes[n] if conv_kernel_sizes else 3
            shortcut_kernel_size = (
                shortcut_kernel_sizes[n] if shortcut_kernel_sizes else 1
            )
            self.residual_layers.append(
                self._make_layer(
                    block,
                    self.num_channels * (2**n),
                    num_blocks[n],
                    stride=stride,
                    conv_kernel_size=conv_kernel_size,
                    shortcut_kernel_size=shortcut_kernel_size,
                    cardinality=self.cardinality,
                )
            )

        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.linear = nn.Linear(
            self.num_channels * (2**n) * block.expansion, num_classes
        )
        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        conv_kernel_size,
        shortcut_kernel_size,
        cardinality,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    cardinality=cardinality,
                    stride=stride,
                    conv_kernel_size=conv_kernel_size,
                    shortcut_kernel_size=shortcut_kernel_size,
                    drop=self.drop,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.squeeze_and_excitation:
            out = self.seblock(out)
        for layer in self.residual_layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)
        if self.drop:
            out = self.dropout(out)
        out = self.linear(out)
        return out

    @classmethod
    def from_config(cls, config_dict: dict) -> "ResNet":
        return cls(
            block=ResNeXtBlock,
            num_blocks=config_dict["num_blocks"],
            conv_kernel_sizes=config_dict["conv_kernel_sizes"],
            shortcut_kernel_sizes=config_dict["shortcut_kernel_sizes"],
            num_channels=config_dict["num_channels"],
            avg_pool_kernel_size=config_dict["avg_pool_kernel_size"],
            drop=config_dict["drop"],
            squeeze_and_excitation=config_dict["squeeze_and_excitation"],
            cardinality=config_dict.get("cardinality", 32),  # Default to 32
        )
