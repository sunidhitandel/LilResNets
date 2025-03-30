import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
from pathlib import Path


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=bias)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
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
        return x * w

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv_kernel_size=3, shortcut_kernel_size=1, drop=0.0):
        super(BasicBlock, self).__init__()
        self.drop = drop
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=conv_kernel_size, stride=stride, padding=conv_kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=conv_kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=shortcut_kernel_size, stride=stride, padding=shortcut_kernel_size//2, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        if self.drop:
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
    def __init__(self, block, num_blocks, num_channels=32, num_classes=10, drop=None, squeeze_and_excitation=None):
        super(ResNet, self).__init__()
        self.in_planes = num_channels
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.drop = drop
        self.squeeze_and_excitation = squeeze_and_excitation
        if self.squeeze_and_excitation:
            self.seblock = SEBlock(channels=self.num_channels)

        self.residual_layers = []
        for n, num_block in enumerate(num_blocks):
            stride = 1 if n == 0 else 2
            self.residual_layers.append(self._make_layer(block, self.num_channels * (2**n), num_block, stride=stride))
        self.residual_layers = nn.ModuleList(self.residual_layers)
        self.avg_pool_kernel_size = 8  # Fixed for 32x32 inputs
        self.linear = nn.Linear(self.num_channels * (2**(len(num_blocks) - 1)), num_classes)
        if self.drop:
            self.dropout = nn.Dropout(self.drop)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.squeeze_and_excitation:
            out = self.seblock(out)
        for layer in self.residual_layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.avg_pool_kernel_size)
        out = torch.flatten(out, 1)
        if self.drop:
            out = self.dropout(out)
        return self.linear(out)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_configs(configs):
    print("Model Configurations and Trainable Parameters:")
    print("-" * 60)
    print(f"{'Model':<10} {'Num Blocks':<15} {'Channels':<10} {'Dropout':<10} {'SE Block':<10} {'Params':<10}")
    print("-" * 60)
    for name in configs:
        config = configs[name]
        model = ResNet(
            block=BasicBlock,
            num_blocks=config["num_blocks"],
            num_channels=config["num_channels"],
            drop=config["drop"],
            squeeze_and_excitation=config["squeeze_and_excitation"],
        )
        params = count_parameters(model)
        print(f"{str(name):<15} {str(config['num_blocks']):<15} {config['num_channels']:<10} {config['drop']:<10} {config['squeeze_and_excitation']:<10} {params:<10}")



def load_configs_from_yaml_folder(configs_dir='configs'):
    configs = {}
    config_path = Path(configs_dir)
    
    if not config_path.exists() or not config_path.is_dir():
        print(f"Warning: Config directory '{configs_dir}' not found")
        return configs
    
    for file_path in config_path.glob('*.yaml'):
        model_name = file_path.stem
        
        with open(file_path, 'r') as file:
            try:
                config_data = yaml.safe_load(file)
                configs[model_name] = config_data
            except yaml.YAMLError as e:
                print(f"Error parsing {file_path.name}: {e}")
    
    return configs
configs = load_configs_from_yaml_folder()

print_model_configs(configs)