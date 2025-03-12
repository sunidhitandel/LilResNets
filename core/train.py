import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.dataset import get_data_loaders
from core.models.resnet import get_model
from core.utils import get_device, save_checkpoint, plot_metrics

def create_optimizer(config, model):
    """Create optimizer based on config"""
    if config['optim'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4)
        )
    elif config['optim'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 5e-4)
        )
    elif config['optim'].lower() == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 5e-4)
        )
    elif config['optim'].lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config['lr'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4)
        )
    else:
        raise ValueError(f"Optimizer {config['optim']} not supported")
    
    return optimizer

def create_scheduler(config, optimizer):
    """Create learning rate scheduler based on config"""
    if config['lr_sched'].lower() == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
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
    scaler = torch.amp.GradScaler("cuda") if config.get('use_amp', False) else None
    
    # Create progress bar for training
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        if scaler is not None:
            # Use AMP for forward pass
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            
            # Apply gradient clipping if specified
            if config.get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping if specified
            if config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
        
        # Update statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return train_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

def train(config, experiment_dir, logger):
    """Main training function"""
    device = get_device()
    logger.log_message(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(config)
    
    # Create model
    model = get_model(config)
    model = model.to(device)
    
    # Log model parameters
    total_params = logger.log_params(model)
    logger.log_device(device, model)
    
    # Create loss function
    if config.get('label_smoothing', 0) > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    
    # Initialize training variables
    start_epoch = 0
    best_acc = 0
    metrics_df = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    # Training loop
    for epoch in range(start_epoch, config['max_epochs']):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log metrics
        logger.log_metrics(epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Update metrics dataframe
        metrics_df.loc[epoch] = [train_loss, train_acc, val_loss, val_acc, current_lr]
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, 
        f"{experiment_dir}/checkpoint.pt", 
        f"{experiment_dir}/best_model.pt")
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
    
    # Save final metrics
    metrics_df.to_csv(f"{experiment_dir}/training_metrics.csv")
    
    # Plot and save training curves
    plot_metrics(metrics_df, experiment_dir)
    
    # Log completion
    logger.log_message(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    return best_acc, metrics_df