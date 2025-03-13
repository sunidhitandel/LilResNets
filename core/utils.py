import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from core.logger import Logger
from core.models import *

MODEL_MASTER_MAPPING = {
    "ResDrop": "resdrop.py",
    "ResReg": "resreg.py",
    "ResWide": "reswide.py",
    "ResMix": "resmix.py",
    "DeepSlim": "deepslim.py",
    "MyResNet": "myresnet.py"
}

def get_device():
    """Determine the available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment(config):
    """Set up experiment directory and logger"""
    experiment_name = config.get('model_name', 'best_model')
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    experiment_dir = os.path.join('experiments', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    logger = Logger(experiment_dir)
    return experiment_dir, logger


def get_model(model_name, config_dict):
    """Retrieve the model based on model_name and config_dict."""
    module_name = MODEL_MASTER_MAPPING.get(model_name, "myresnet.py")
    model_module = __import__(module_name.replace(".py", ""))
    if hasattr(model_module, "get_custom_model"):
        return model_module.get_custom_model(config_dict)
    return model_module.MyResNet(config_dict)

def save_checkpoint(state, is_best, filename, best_filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def plot_metrics(metrics_df, experiment_dir):
    """Plot and save training metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot accuracy
    ax1.plot(metrics_df.index, metrics_df['train_acc'], label='Train Accuracy')
    ax1.plot(metrics_df.index, metrics_df['val_acc'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss and learning rate
    ax2.plot(metrics_df.index, metrics_df['train_loss'], label='Train Loss')
    ax2.plot(metrics_df.index, metrics_df['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate on secondary y-axis
    ax3 = ax2.twinx()
    ax3.plot(metrics_df.index, metrics_df['lr'], 'g--', label='Learning Rate')
    ax3.set_ylabel('Learning Rate')
    ax3.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f"{experiment_dir}/training_metrics.jpg", dpi=300)
    plt.close()