import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.execution_log_path = os.path.join(log_dir, "execution_log.txt")
        with open(self.execution_log_path, "a") as f:
            f.write(
                f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    def log_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log training and validation metrics"""
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["train_acc"].append(train_acc)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["val_acc"].append(val_acc)
        self.metrics["lr"].append(lr)

        # Log to CSV
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(
            os.path.join(self.log_dir, "training_metrics.csv"), index=False
        )

        # Log to execution log
        with open(self.execution_log_path, "a") as f:
            f.write(
                f"Epoch {epoch}/{self.metrics['epoch'][-1]} Val - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}\n"
            )

    def log_message(self, message):
        """Log a message to the execution log"""
        with open(self.execution_log_path, "a") as f:
            f.write(f"{message}\n")

    def log_model_state(self, device, model):
        """Log model state"""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open(self.execution_log_path, "a") as f:
            f.write(f"Using device: {device}\n")
            f.write(f"Model on GPU: {next(model.parameters()).is_cuda}\n")
            f.write(f"Total Parameters: {total_params}\n")
