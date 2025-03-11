import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from datetime import datetime
import pandas as pd

from .metrics import AverageMeter, calculate_accuracy
from .logger import Logger

def train_epoch(model, train_loader, optimizer, criterion, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if config.use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            # Scale gradients and perform backward pass
            scaler.scale(loss).backward()
            
            # Optional gradient clipping
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Optional gradient clipping
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
            optimizer.step()
        
        # Compute accuracy and update metrics
        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accs.avg:.4f}'
        })
    
    return losses.avg, accs.avg

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, config, logger):
    """Train the model for multiple epochs"""
    device = next(model.parameters()).device
    best_acc = 0
    start_epoch = 0
    metrics_df = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    # Create experiments directory if it doesn't exist
    os.makedirs(os.path.join(config.save_dir, config.experiment_name), exist_ok=True)
    
    # Initialize mixed precision scaler if needed
    scaler = torch.cuda.amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None
    
    # Training loop
    for epoch in range(start_epoch, config.max_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config, scaler
        )
        
        # Evaluate on validation set
        val_loss, val_acc = test_epoch(
            model, val_loader, criterion, device, config
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log metrics
        logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Save metrics to dataframe
        metrics_df.loc[epoch] = [train_loss, train_acc, val_loss, val_acc, current_lr]
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.save_dir, config.experiment_name, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.param_groups,
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, save_path)
            
        # Save metrics to CSV
        metrics_df.to_csv(os.path.join(config.save_dir, config.experiment_name, 'training_metrics.csv'), index=True)
        
    return best_acc, metrics_df