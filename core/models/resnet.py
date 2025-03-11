import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_ks=3, shortcut_ks=1, use_se=False, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        
        # Calculate padding to maintain size
        padding = (conv_ks - 1) // 2
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_ks, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_ks, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if self.use_se:
            self.se = SELayer(planes, reduction=16)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            shortcut_padding = (shortcut_ks - 1) // 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=shortcut_ks, stride=stride, padding=shortcut_padding, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=64, num_classes=10, 
                 conv_kernel_sizes=None, shortcut_kernel_sizes=None, 
                 avg_pool_kernel_size=8, dropout_rate=0, use_se=False):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3] * len(num_blocks)
        if shortcut_kernel_sizes is None:
            shortcut_kernel_sizes = [1] * len(num_blocks)
        
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        layers = []
        for i, num_block in enumerate(num_blocks):
            stride = 2 if i > 0 else 1
            layers.append(
                self._make_layer(block, num_channels * (2**i), num_block, stride, 
                               conv_kernel_sizes[i], shortcut_kernel_sizes[i], use_se, dropout_rate)
            )
        self.layers = nn.Sequential(*layers)
        
        self.avg_pool = nn.AvgPool2d(avg_pool_kernel_size)
        last_channel_size = num_channels * (2**(len(num_blocks)-1))
        self.linear = nn.Linear(last_channel_size * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, conv_ks, shortcut_ks, use_se, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, conv_ks, shortcut_ks, use_se, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(config):
    return ResNet(BasicBlock, [3, 3, 3], 
                 num_channels=config.get('num_channels', 64),
                 num_classes=10,
                 conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
                 shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
                 avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
                 dropout_rate=config.get('drop', 0),
                 use_se=config.get('squeeze_and_excitation', False))

def resnet18(config):
    return ResNet(BasicBlock,  [4, 4, 3], 
                 num_channels=config.get('num_channels', 64),
                 num_classes=10,
                 conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
                 shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
                 avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
                 dropout_rate=config.get('drop', 0),
                 use_se=config.get('squeeze_and_excitation', True))

def resnet56(config):
    return ResNet(BasicBlock, [9, 9, 9], 
                 num_channels=config.get('num_channels', 64),
                 num_classes=10,
                 conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
                 shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
                 avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
                 dropout_rate=config.get('drop', 0),
                 use_se=config.get('squeeze_and_excitation', False))

def resnet156(config):
    return ResNet(BasicBlock, [25, 25, 25], 
                 num_channels=config.get('num_channels', 64),
                 num_classes=10,
                 conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
                 shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
                 avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
                 dropout_rate=config.get('drop', 0),
                 use_se=config.get('squeeze_and_excitation', False))

# Model registry
model_registry = {
    'resnet20': resnet20,
    'resnet18': resnet18,
    'resnet56': resnet56,
    'resnet156': resnet156
}

def get_model(config):
    model_name = config.get('model', 'resnet18')
    if model_name in model_registry:
        return model_registry[model_name](config)
    else:
        raise ValueError(f"Model {model_name} not found in registry")