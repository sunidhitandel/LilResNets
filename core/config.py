import dataclasses
import yaml
import torch
import os
from typing import List, Optional, Union, Dict, Any

@dataclasses.dataclass
class Config:
    # Model parameters
    num_blocks: List[int] = dataclasses.field(default_factory=lambda: [4, 4, 3])
    conv_kernel_sizes: List[int] = dataclasses.field(default_factory=lambda: [3, 3, 3])
    shortcut_kernel_sizes: List[int] = dataclasses.field(default_factory=lambda: [1, 1, 1])
    num_channels: int = 64
    avg_pool_kernel_size: int = 8
    drop: float = 0.0
    squeeze_and_excitation: int = 1
    model_type: str = "custom"
    
    # Training parameters
    grad_clip: float = 0.1
    max_epochs: int = 200
    optim: str = "sgd"
    lr_sched: str = "CosineAnnealingLR"
    momentum: float = 0.9
    lr: float = 0.01
    weight_decay: float = 0.0005
    batch_size: int = 128
    num_workers: int = 16
    data_augmentation: int = 1
    data_normalize: int = 1
    lookahead: int = 1
    label_smoothing: float = 0.1
    use_amp: bool = True
    
    # Data paths
    dataset_dir: str = ""
    val_dataset_dir: str = ""
    
    # Experiment parameters
    experiment_name: str = "default_experiment"
    save_dir: str = "./experiments"
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'Config':
        """Load configuration from a YAML file"""
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_file: str) -> None:
        """Save configuration to a YAML file"""
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        with open(yaml_file, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return dataclasses.asdict(self)
    
    def update(self, **kwargs) -> 'Config':
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get(self, key, default=None):
        """Get attribute with default fallback"""
        return getattr(self, key, default)