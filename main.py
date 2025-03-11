#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.config import parse_config
from core.logger import Logger
from core.train import train
from core.test import test
from models.factory import create_model
from data.dataset import load_train_data, load_test_data

def main():
    try:
        # Argument parsing
        parser = argparse.ArgumentParser(description="Train and evaluate ResNet variants")
        parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
        parser.add_argument("--experiment", type=str, help="Experiment name (overrides config)")
        parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
        parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
        parser.add_argument("--gpu", type=str, help="Specify GPU device ID")
        args = parser.parse_args()

        # Load configuration
        config = parse_config(args.config)

        if args.experiment:
            exp_config_path = f"configs/experiments/{args.experiment}.yaml"
            if os.path.exists(exp_config_path):
                with open(exp_config_path, "r") as f:
                    exp_config = yaml.safe_load(f)
                config.__dict__.update(exp_config)
                config.experiment_name = args.experiment
            else:
                print(f"Warning: Experiment config '{exp_config_path}' not found.")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        print(f"Using device: {device}")

        # Create experiment directory
        os.makedirs(config.save_dir, exist_ok=True)

        # Initialize logger and model
        logger = Logger(config)
        model = create_model(config).to(device)
        logger.save_model_summary(str(model))

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=getattr(config, "label_smoothing", 0))
        optimizer = getattr(optim, config.optim.capitalize())(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Load data
        train_loader = DataLoader(load_train_data(config), batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(load_test_data(config), batch_size=config.batch_size, shuffle=False)

        # Resume from checkpoint
        start_epoch, best_acc = 0, 0
        if args.resume:
            checkpoint_path = os.path.join(config.save_dir, f"{config.experiment_name}_checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch, best_acc = checkpoint["epoch"] + 1, checkpoint.get("best_acc", 0)

        # Evaluation mode
        if args.evaluate:
            test_loss, test_acc = test(model, test_loader, criterion, device, config, logger, evaluate=True)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            return

        # Training loop
        print(f"Starting training for {config.max_epochs} epochs")
        for epoch in range(start_epoch, config.max_epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, config, logger)
            test_loss, test_acc = test(model, test_loader, criterion, device, config, logger, epoch=epoch)

            logger.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": test_loss, "val_acc": test_acc})

            # Save checkpoint
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_acc": best_acc},
                           os.path.join(config.save_dir, f"{config.experiment_name}_checkpoint.pth"))

        print(f"Training completed. Best accuracy: {best_acc:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
