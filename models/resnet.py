import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import BasicBlock

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=64, num_classes=10, 
                 kernel_sizes=None, shortcut_kernel_sizes=None, 
                 use_se=False, dropout_rate=0, avg_pool_size=8):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        
        if kernel_sizes is None:
            kernel_sizes = [3] * len(num_blocks)
        if shortcut_kernel_sizes is None:
            shortcut_kernel_sizes = [1] * len(num_blocks)
            
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        layers = []
        for i, num_block in enumerate(num_blocks):
            layers.append(
                self._make_layer(block, num_channels * (2**i), num_block, stride=1 if i == 0 else 2,
                                kernel_size=kernel_sizes[i], shortcut_kernel_size=shortcut_kernel_sizes[i])
            )
        self.layers = nn.Sequential(*layers)
        
        final_channels = num_channels * (2 ** (len(num_blocks) - 1))
        self.avg_pool = nn.AvgPool2d(avg_pool_size)
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
            
        self.linear = nn.Linear(final_channels, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size=3, shortcut_kernel_size=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, kernel_size, shortcut_kernel_size, self.use_se)
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        if self.dropout_rate > 0:
            out = self.dropout(out)
            
        out = self.linear(out)
        return out
