import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import dataclasses
from .transforms import get_train_transforms, get_test_transforms

@dataclasses.dataclass
class DataConfig:
    dataset_dir: str
    val_dataset_dir: str
    add_augmentation: bool = True
    add_jitter: bool = False
    add_normalization: bool = True


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


class CIFARDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # shape (N, 3072)
        self.labels = labels  # shape (N,)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_train_data(config: DataConfig):
    data_list, labels_list = [], []
    for i in range(1, 6):
        batch_file = os.path.join(config.dataset_dir, f"data_batch_{i}")
        batch = unpickle(batch_file)
        data_list.append(batch[b'data'])
        labels_list.extend(batch[b'labels'])
    X = np.concatenate(data_list, axis=0)
    y = np.array(labels_list)
    return CIFARDataset(X, y, transform=get_train_transforms(config))


def load_test_data(config: DataConfig):
    val_batch = unpickle(config.val_dataset_dir)
    return CIFARDataset(val_batch[b'data'], val_batch[b'labels'], transform=get_test_transforms(config.add_normalization))


def create_data_loaders(config: DataConfig, batch_size: int, num_workers: int):
    train_dataset = load_train_data(config)
    test_dataset = load_test_data(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size//2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

