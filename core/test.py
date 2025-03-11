import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import os

from .metrics import AverageMeter, calculate_accuracy

def test_epoch(model, val_loader, criterion, device, config):
    """Evaluate model on validation data"""
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            if config.use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Compute accuracy and update metrics
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'val_loss': f'{losses.avg:.4f}',
                'val_acc': f'{accs.avg:.4f}'
            })
    return losses.avg, accs.avg

def evaluate(model, val_loader, criterion, device, config, class_names=None, visualize=True):
    """Full evaluation with optional visualization and confusion matrix"""
    model.eval()
    val_loss, val_acc = test_epoch(model, val_loader, criterion, device, config)
    
    # If visualization is requested
    if visualize:
        # Generate confusion matrix
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        
        # Create a confusion matrix plot
        plt.figure(figsize=(10, 8))
        if class_names is None:
            class_names = [str(i) for i in range(10)]  # Default CIFAR-10 classes
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix
        save_path = os.path.join(config.save_dir, config.experiment_name, 'confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        # Print classification report if sklearn is available
        try:
            from sklearn.metrics import classification_report
            report = classification_report(all_targets, all_preds, target_names=class_names)
            print("\nClassification Report:\n", report)
            
            # Save report to file
            with open(os.path.join(config.save_dir, config.experiment_name, 'classification_report.txt'), 'w') as f:
                f.write(report)
        except ImportError:
            pass
    
    return val_loss, val_acc
