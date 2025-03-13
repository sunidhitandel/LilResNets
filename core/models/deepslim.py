import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """
    Convolution 1x1 layer.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=channels,
            bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, conv_kernel_size=3, shortcut_kernel_size=1, drop=0.0):
        """
        Convolutional Layer kernel size Fi
        Skip connection (shortcut) kernel size Ki
        """
        super(BasicBlock, self).__init__()
        self.drop = drop
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_kernel_size, stride=stride, padding=int(conv_kernel_size/2), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size,stride=1, padding=int(conv_kernel_size/2), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,kernel_size=shortcut_kernel_size, stride=stride, padding=int(shortcut_kernel_size/2), bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if self.drop: self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.drop > 0:
            out = self.dropout(out)
        return out


class ResNetDeepSlim(nn.Module):
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
            squeeze_and_excitation=None):
        super(ResNetDeepSlim, self).__init__()
        self.in_planes = num_channels
        self.avg_pool_kernel_size = int(32 / (2**(len(num_blocks)-1)))

        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation
        if self.squeeze_and_excitation:
            self.seblock = SEBlock(channels=self.num_channels)

        self.residual_layers = []
        for n in range(len(num_blocks)):
            stride = 1 if n==0 else 2
            conv_kernel_size = conv_kernel_sizes[n] if conv_kernel_sizes else 3
            shortcut_kernel_size = shortcut_kernel_sizes[n] if shortcut_kernel_sizes else 1
            self.residual_layers.append(self._make_layer(
                                                    block,
                                                    self.num_channels*(2**n),
                                                    num_blocks[n],
                                                    stride=stride,
                                                    conv_kernel_size=conv_kernel_size,
                                                    shortcut_kernel_size=shortcut_kernel_size))

        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.linear = nn.Linear(self.num_channels*(2**n)*block.expansion, num_classes)
        
        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def _make_layer(self, block, planes, num_blocks, stride, conv_kernel_size, shortcut_kernel_size):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_kernel_size, shortcut_kernel_size, drop=self.drop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.squeeze_and_excitation: out = self.seblock(out)
        for layer in self.residual_layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)
        if self.drop: out = self.dropout(out)
        out = self.linear(out)
        return out

# Function to get the DeepSlim model
def get_custom_model(config_dict):
    """
    Factory function that returns the DeepSlim model and its total parameters
    """
    model = ResNetDeepSlim(
            block=BasicBlock,
            num_blocks=config_dict['num_blocks'],
            conv_kernel_sizes=config_dict['conv_kernel_sizes'],
            shortcut_kernel_sizes=config_dict['shortcut_kernel_sizes'],
            num_channels=config_dict['num_channels'],
            avg_pool_kernel_size=config_dict['avg_pool_kernel_size'],
            drop=config_dict['drop'],
            squeeze_and_excitation=config_dict['squeeze_and_excitation']
        )

    total_params = 0
    for x in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    return model, total_params