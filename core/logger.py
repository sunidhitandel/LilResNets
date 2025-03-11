import os
import json
import numpy as np
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional

class Logger:
    """Logging utility for training"""
    
    def __init__(self, config, log_dir=None):
        """Initialize logger with config"""
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if log_dir is None:
            log_dir = os.path.join(
                config.save_dir, 
                f"{config.experiment_name}_{timestamp}"
            )
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Save config
        self.save_config()
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def save_config(self):
        """Save configuration to JSON file"""
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics to TensorBoard and internal storage"""
        epoch = metrics.get('epoch', len(self.metrics['train_loss']))
        
        # Log to TensorBoard
        for key, value in metrics.items():
            if key != 'epoch':
                self.writer.add_scalar(key, value, epoch)
        
        # Store metrics internally
        for key in self.metrics.keys():
            if key in metrics:
                self.metrics[key].append(metrics[key])
        
        # Log to console
        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {metrics.get('train_loss', 'N/A'):.4f}, "
              f"Train Acc: {metrics.get('train_acc', 'N/A'):.4f}, "
              f"Val Loss: {metrics.get('val_loss', 'N/A'):.4f}, "
              f"Val Acc: {metrics.get('val_acc', 'N/A'):.4f}, "
              f"LR: {metrics.get('lr', 'N/A'):.6f}")
    
    def log_model(self, model, input_shape=(1, 3, 32, 32)):
        """Log model architecture and graph to TensorBoard"""
        try:
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_shape, device=device)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Failed to log model graph: {e}")
    
    def save_model_summary(self, model_summary):
        """Save model summary to text file"""
        summary_path = os.path.join(self.log_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(str(model_summary))
    
    def plot_metrics(self):
        """Plot training metrics and save to files"""
        epochs = list(range(1, len(self.metrics['train_loss']) + 1))
        
        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.metrics['train_acc'], label='Train Accuracy')
        plt.plot(epochs, self.metrics['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'accuracy_plot.png'))
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.metrics['lr'])
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.savefig(os.path.join(self.log_dir, 'lr_plot.png'))
        plt.close()
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': epochs,
            **{k: v for k, v in self.metrics.items()}
        })
        metrics_df.to_csv(os.path.join(self.log_dir, 'metrics.csv'), index=False)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int = 0):
        """Log images to TensorBoard
        
        Args:
            tag: Tag for the images
            images: Tensor of images with shape [B, C, H, W]
            step: Global step value
        """
        self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int = 0):
        """Log histogram of values to TensorBoard
        
        Args:
            tag: Tag for the histogram
            values: Values to create histogram from
            step: Global step value
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_model_weights(self, model, step: int = 0):
        """Log model weights as histograms
        
        Args:
            model: PyTorch model
            step: Global step value
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: list, epoch: int = 0):
        """Log confusion matrix as an image
        
        Args:
            cm: Confusion matrix as numpy array
            class_names: List of class names
            epoch: Current epoch
        """
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize and print the values in cells
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                         horizontalalignment='center',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        self.writer.add_figure('confusion_matrix', figure, epoch)
        plt.close()
    
    def log_embeddings(self, embeddings: torch.Tensor, metadata: list, tag: str = 'embeddings', step: int = 0):
        """Log embeddings for visualization in TensorBoard
        
        Args:
            embeddings: Tensor of embeddings
            metadata: List of labels for each embedding
            tag: Tag for the embeddings
            step: Global step value
        """
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            tag=tag,
            global_step=step
        )
    
    def create_checkpoint(self, state: Dict[str, Any], is_best: bool = False, filename: str = 'checkpoint.pth.tar'):
        """Save checkpoint of the model and optimizer
        
        Args:
            state: Dictionary containing model state, optimizer state, epoch, etc.
            is_best: Flag indicating if this is the best model so far
            filename: Filename for the checkpoint
        """
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.log_dir, filename)
        torch.save(state, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.log_dir, 'model_best.pth.tar')
            torch.save(state, best_path)
    
    def close(self):
        """Clean up and close the logger"""
        # Generate final plots
        self.plot_metrics()
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save full metrics history as JSON
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"Logger closed. All logs saved to {self.log_dir}")