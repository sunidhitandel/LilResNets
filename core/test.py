import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.dataset import get_test_data_loader
from core.utils import get_device, get_model
import pickle  
from torchvision import transforms  
from torch.utils.data import Dataset, DataLoader  

def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict
def get_transforms(normalization=True):
    test_transform_list = []
    test_transform_list.append(transforms.ToPILImage())
    test_transform_list.append(transforms.ToTensor())
    if normalization:
        norm_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        test_transform_list.append(norm_transform)
    test_transforms = transforms.Compose(test_transform_list)
    return test_transforms

class CIFARTestDataset(Dataset):
    def __init__(self, data, ids, transform=None):
        """
        data: numpy array of shape (N, 32, 32, 3)
        ids: array-like of test image IDs
        transform: transforms to apply
        """
        self.data = data
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # If data is stored as (32,32,3)
        img = self.data[idx].astype("uint8")
        img_id = self.ids[idx]
        if self.transform:
            img = self.transform(img)
        return img, img_id

def test(model_name, config, model_path, output_dir, logger=None):
    """Generate predictions for unlabeled test data"""
    device = get_device()
    test_file = "./data/cifar_test_nolabel.pkl"
    test_dict = unpickle(test_file)
    test_images = test_dict[b'data'] 
    test_ids = test_dict[b'ids']
    transform_test = get_transforms()
    test_dataset = CIFARTestDataset(test_images, test_ids, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=int(256/4), shuffle=False, num_workers=16)
    model,_ = get_model(model_name, config)
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
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_ids.extend(ids.tolist())

    # Create submission CSV (must have exactly two columns: "ID" and "Labels")
    submission_df = pd.DataFrame({
        "ID": all_ids,
        "Labels": all_preds
    })
    submission_path = f"{output_dir}/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    if logger:
        logger.log_message(f"Predictions saved to {submission_path}")
    
    return submission_path

