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
        padding = (conv_kernel_size - 1) // 2
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_kernel_size, 
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size,
                              stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            shortcut_padding = (shortcut_kernel_size - 1) // 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=shortcut_kernel_size, 
                          stride=stride, padding=shortcut_padding, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            
        if self.drop > 0:
            self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
            squeeze_and_excitation=None):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        
        # Set default kernel sizes if not provided
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3] * len(num_blocks)
        if shortcut_kernel_sizes is None:
            shortcut_kernel_sizes = [1] * len(num_blocks)
        
        # Calculate adaptive pool kernel size based on num_blocks if not provided
        if avg_pool_kernel_size is None:
            self.avg_pool_kernel_size = int(32 / (2**(len(num_blocks)-1)))
        else:
            self.avg_pool_kernel_size = avg_pool_kernel_size

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation

        # Add SE block after initial convolution if specified
        if self.squeeze_and_excitation:
            self.seblock = SEBlock(channels=num_channels)

        # Create residual layers
        self.residual_layers = []
        for n in range(len(num_blocks)):
            stride = 1 if n == 0 else 2  # stride=1 for first layer, stride=2 for others
            conv_kernel_size = conv_kernel_sizes[n]
            shortcut_kernel_size = shortcut_kernel_sizes[n]
            self.residual_layers.append(self._make_layer(
                block,
                num_channels * (2**n),
                num_blocks[n],
                stride=stride,
                conv_kernel_size=conv_kernel_size,
                shortcut_kernel_size=shortcut_kernel_size))

        # Register layers as ModuleList so they're properly recognized
        self.residual_layers = nn.ModuleList(self.residual_layers)
        
        # Output projection
        final_channels = num_channels * (2**(len(num_blocks)-1))
        self.linear = nn.Linear(final_channels * block.expansion, num_classes)
        
        # Final dropout
        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def _make_layer(self, block, planes, num_blocks, stride, conv_kernel_size, shortcut_kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, 
                planes, 
                stride, 
                conv_kernel_size, 
                shortcut_kernel_size, 
                drop=self.drop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Apply SE block if specified
        if self.squeeze_and_excitation:
            out = self.seblock(out)
            
        # Apply residual layers
        for layer in self.residual_layers:
            out = layer(out)
            
        # Global pooling
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = out.view(out.size(0), -1)
        
        # Final dropout before classification
        if self.drop:
            out = self.dropout(out)
            
        out = self.linear(out)
        return out

# Model factory functions
def resnet18(config):
    return ResNet(
        block=BasicBlock,
        num_blocks=[3, 3, 3],
        conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
        shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
        num_classes=10,
        num_channels=config.get('num_channels', 64),
        avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
        drop=config.get('drop', 0),
        squeeze_and_excitation=config.get('squeeze_and_excitation', True)
    )

def best_model(config):
    return ResNet(
        block=BasicBlock,
        num_blocks=config.get('num_blocks',  [4, 4, 3]),
        conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
        shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
        num_classes=10,
        num_channels=config.get('num_channels', 64),
        avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
        drop=config.get('drop', 0),
        squeeze_and_excitation=config.get('squeeze_and_excitation', True)
    )

def resnet56(config):
    return ResNet(
        block=BasicBlock,
        num_blocks=[9, 9, 9],
        conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
        shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
        num_classes=10,
        num_channels=config.get('num_channels', 16),
        avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
        drop=config.get('drop', 0.1),
        squeeze_and_excitation=config.get('squeeze_and_excitation', False)
    )

def resnet156(config):
    return ResNet(
        block=BasicBlock,
        num_blocks=[25, 25, 25],
        conv_kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
        shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
        num_classes=10,
        num_channels=config.get('num_channels', 8),
        avg_pool_kernel_size=config.get('avg_pool_kernel_size', 8),
        drop=config.get('drop', 0.1),
        squeeze_and_excitation=config.get('squeeze_and_excitation', False)
    )

# Model registry
model_registry = {
    'best_model': best_model,
    'resnet18': resnet18,
    'resnet56': resnet56,
    'resnet156': resnet156
}

def get_model(config_dict):
    """
    Factory function that returns a model and its total parameters
    """
    model_name = config_dict.get('model', 'best_model')
    if model_name in model_registry:
        model = model_registry[model_name](config_dict)
    else:
        raise ValueError(f"Model {model_name} not found in registry")
    return model

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False