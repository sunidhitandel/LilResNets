import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.dataset import get_test_data_loader
from core.models.resnet import get_model
from core.utils import get_device

def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def test(config, model_path, output_dir, logger=None):
    """Generate predictions for unlabeled test data"""
    device = get_device()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_loader = get_test_data_loader(config)
    model = get_model(config)
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if logger:
        logger.log_message(f"Loaded model from {model_path}")
        if 'epoch' in checkpoint:
            logger.log_message(f"Model checkpoint from epoch {checkpoint['epoch']}")
    
    model.eval()
    all_preds = []
    all_ids = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_ids.extend(ids.tolist())
    submission_df = pd.DataFrame({
        'ID': all_ids,
        'Labels': all_preds
    })
    submission_path = f"{output_dir}/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    if logger:
        logger.log_message(f"Predictions saved to {submission_path}")
    
    return submission_path

