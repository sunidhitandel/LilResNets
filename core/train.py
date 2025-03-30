import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from core import logger
from core.dataset import get_data_loaders
from core.utils import get_device
from core.utils import get_model
from core.utils import save_checkpoint



def create_optimizer(config, model):
    """Create optimizer based on config"""
    optimizers = {
        "sgd": lambda: optim.SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0),
            weight_decay=config.get("weight_decay", 0),
        ),
        "adam": lambda: optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0)
        ),
        "adagrad": lambda: optim.Adagrad(
            model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0)
        ),
        "rmsprop": lambda: optim.RMSprop(
            model.parameters(),
            lr=config["lr"],
            momentum=config.get("momentum", 0),
            weight_decay=config.get("weight_decay", 0),
        ),
        "adamw": lambda: optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0)
        ),
        "radam": lambda: optim.RAdam(
            model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0)
        ),
    }
    
    optim_name = config["optim"].lower()
    if optim_name not in optimizers:
        raise ValueError(f"Optimizer {config['optim']} not supported")
    
    return optimizers[optim_name]()

def create_scheduler(config, optimizer):
    """Create learning rate scheduler based on config"""
    sched_type = config["lr_sched"].lower()
    if sched_type == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    elif sched_type == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.get("T_0", 10), T_mult=config.get("T_mult", 2)
        )
    elif sched_type == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
        )
    elif sched_type == "reduceonplateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    elif sched_type == "onecyclelr":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config["max_lr"],
            epochs=config["max_epochs"],
            steps_per_epoch=config["steps_per_epoch"],
        )
    else:
        raise ValueError(f"Scheduler {config['lr_sched']} not supported")
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # Enable automatic mixed precision training if specified
    scaler = None
    if config.get("use_amp", False) and device == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # Create progress bar for training
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        # Reset gradients
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if config.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
        # Update statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # Update progress bar
        pbar.set_postfix(
            {"loss": train_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
        )

    return train_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return val_loss / len(val_loader), 100.0 * correct / total


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0009):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
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


def train(config: str, experiment_dir: str, logger: logger.Logger):
    """Main training function"""
    device = get_device()
    logger.log_message(f"Using device: {device}")
    train_loader, val_loader = get_data_loaders(config, logger)

    # Create model
    model = get_model(config, logger)
    model = model.to(device)
    logger.log_model_state(device, model)

    # Check if old checkpoint exists
    ckpt_path = os.path.join(experiment_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # Create loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=(
            config["label_smoothing"] if config.get("label_smoothing", None) else 0
        )
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)

    # Initialize training variables
    start_epoch = 0
    best_acc = 0
    if config.get("early_stopping_patience", None):
        early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"], min_delta=config["min_delta"]
        )

    # Training loop
    for epoch in range(start_epoch, config["max_epochs"]):
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # Log metrics
        logger.log_metrics(
            epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr
        )
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "acc": val_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            f"{experiment_dir}/checkpoint.pt",
            f"{experiment_dir}/best_model.pt",
        )
        # Print progress
        print(
            f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}"
        )
        if "early_stopping_patience" in config:
            if early_stopping(val_loss):
                logger.log_message("Early stopping triggered!!!")
                break

    logger.log_message(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    return best_acc
