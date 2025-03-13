import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from core import dataset
from core import logger
from core import utils


def test(config: dict, ckpt_path: str, output_dir: str, logger: logger.Logger):
    """Generate predictions for unlabeled test data"""
    device = utils.get_device()
    test_loader = dataset.get_test_data_loader(config)
    model = utils.get_model(config, logger).to(device)

    # Load best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if logger:
        logger.log_message(f"Loaded model from {ckpt_path}")
        if "epoch" in checkpoint:
            logger.log_message(f"Model checkpoint from epoch {checkpoint['epoch']}")

    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_ids.extend(ids.tolist())

    # Create submission CSV (must have exactly two columns: "ID" and "Labels")
    submission_df = pd.DataFrame({"ID": all_ids, "Labels": all_preds})
    os.makedirs(output_dir, exist_ok=True)
    submission_path = f"{output_dir}/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    if logger:
        logger.log_message(f"Predictions saved to {submission_path}")

    return submission_path
