import torch
from .resnet import ResNet
from .blocks import BasicBlock

def create_model(config, num_classes=10):
    """Factory function to create models based on configuration"""
    model_type = config.get('model_type', 'custom')
    
    if model_type == 'resnet18':
        model = ResNet(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            num_channels=config.get('num_channels', 64),
            num_classes=num_classes,
            use_se=config.get('squeeze_and_excitation', 0) == 1,
            dropout_rate=config.get('drop', 0)
        )
    elif model_type == 'resnet50':
        # For simplicity, still using BasicBlock instead of Bottleneck
        model = ResNet(
            block=BasicBlock,
            num_blocks=[3, 4, 6, 3],
            num_channels=config.get('num_channels', 64),
            num_classes=num_classes,
            use_se=config.get('squeeze_and_excitation', 0) == 1,
            dropout_rate=config.get('drop', 0)
        )
    else:  # Custom architecture
        model = ResNet(
            block=BasicBlock,
            num_blocks=config.get('num_blocks', [4, 4, 3]),
            num_channels=config.get('num_channels', 64),
            kernel_sizes=config.get('conv_kernel_sizes', [3, 3, 3]),
            shortcut_kernel_sizes=config.get('shortcut_kernel_sizes', [1, 1, 1]),
            use_se=config.get('squeeze_and_excitation', 0) == 1,
            dropout_rate=config.get('drop', 0),
            avg_pool_size=config.get('avg_pool_kernel_size', 8),
            num_classes=num_classes
        )
    
    return model