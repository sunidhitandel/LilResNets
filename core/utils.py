import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from core.logger import Logger
from core.models import deepslim
from core.models import myresnet
from core.models import prenet
from core.models import resdrop
from core.models import resmix
from core.models import resreg
from core.models import reswide


def get_device():
    """Determine the available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def unpickle(file):
    """Unpickle the CIFAR-10 data files"""
    with open(file, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def load_config(config_path: str):
    """Load YAML configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(experiment_name: str) -> str:
    """Set up experiment directory and logger"""
    experiment_dir = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


MODEL_MASTER_MAPPING = {
    "ResDrop": "resdrop",
    "ResReg": "resreg",
    "ResWide": "reswide",
    "ResMix": "resmix",
    "DeepSlim": "deepslim",
    "MyResNet": "myresnet",
}


def get_model(config_dict: dict, logger: Logger):
    """Retrieve the model based on model_name and config_dict."""
    model_name = config_dict["model_name"].lower()

    # Mapping model names to their respective classes
    model_map = {
        "deepslim": deepslim.HybridResNet,
        "resdrop": resdrop.ResNetDeepSlim,
        "resreg": resreg.ResNetDeepSlim,
        "reswide": reswide.ResNetDeepSlim,
        "resmix": resmix.ResNetDeepSlim,
        "prenet": prenet.PreResNet,
    }

    # Get the model class, default to MyResNet if not found
    model_class = model_map.get(model_name, myresnet.ResNet)
    return model_class.from_config(config_dict)


def save_checkpoint(state, is_best, filename, best_filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def plot_metrics(metrics_df, experiment_dir):
    """Plot and save training metrics"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot accuracy
    ax1.plot(metrics_df.index, metrics_df["train_acc"], label="Train Accuracy")
    ax1.plot(metrics_df.index, metrics_df["val_acc"], label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Plot loss and learning rate
    ax2.plot(metrics_df.index, metrics_df["train_loss"], label="Train Loss")
    ax2.plot(metrics_df.index, metrics_df["val_loss"], label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.legend()
    ax2.grid(True)

    # Plot learning rate on secondary y-axis
    ax3 = ax2.twinx()
    ax3.plot(metrics_df.index, metrics_df["lr"], "g--", label="Learning Rate")
    ax3.set_ylabel("Learning Rate")
    ax3.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{experiment_dir}/training_metrics.jpg", dpi=300)
    plt.close()
