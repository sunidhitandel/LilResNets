import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from core import utils
from core.augmentation import get_augmentations

dataset_dir = "./data/cifar-10-python/cifar-10-batches-py/"
val_dataset_dir = "./data/cifar-10-python/cifar-10-batches-py/test_batch"
test_dataset_dir = "./data/cifar_test_nolabel.pkl"

class CIFARDataset(Dataset):
    """CIFAR-10 dataset handler"""

    def __init__(self, data, labels, transform, config):
        self.data = data  # shape (N, 3072)
        self.labels = labels  # shape (N,)
        self.transform=transform
        self.augmentations = False if config['data_augmentation']==1 else {"data_augmentation":config['data_augmentation']}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        # Apply augmentations like MixUp or CutMix
        if self.augmentations:
            for aug_name, aug in self.augmentations.items():
                if aug_name in ["MixUp", "CutMix"] and np.random.random() < 0.5:
                    idx2 = np.random.randint(len(self.data))
                    img2 = self.data[idx2].reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
                    label2 = self.labels[idx2]
                    if self.transform:
                        img2 = self.transform(img2)
                    img, label, label2, lam = aug(img, img2, label, label2)
                    return img, (label, label2, lam)
                
        return img, label


class CIFARTestDataset(Dataset):
    """Dataset handler for CIFAR test data"""

    def __init__(self, data, ids=None, transform=None):
        self.data = data
        self.ids = ids if ids is not None else np.arange(len(data))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
        img_id = self.ids[idx]
        if self.transform:
            img = self.transform(img)
        return img, img_id


def get_train_transforms(config):
    """Get transforms for training data"""
    train_transforms = [transforms.ToPILImage()]

    if config.get("add_augmentation", True):
        train_transforms.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    if config.get("add_jitter", False):
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )

    train_transforms.append(transforms.ToTensor())

    if config.get("add_normalization", True):
        train_transforms.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )

    return transforms.Compose(train_transforms)


def get_val_transforms(add_normalization=True):
    """Get transforms for test/validation data"""
    test_transforms = [transforms.ToPILImage(), transforms.ToTensor()]

    if add_normalization:
        test_transforms.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )

    return transforms.Compose(test_transforms)


def load_train_data(config, logger):
    """Load CIFAR-10 training data"""
    logger.log_message("Loading training data...")
    data_list, labels_list = [], []
    for i in range(1, 6):
        batch_file = os.path.join(dataset_dir, f"data_batch_{i}")
        batch = utils.unpickle(batch_file)
        data_list.append(batch[b"data"])
        labels_list.extend(batch[b"labels"])
    X = np.concatenate(data_list, axis=0)
    y = np.array(labels_list)
    logger.log_message("Training data loaded successfully.")
    return CIFARDataset(
        X, 
        y, 
        transform=get_train_transforms(config),
        config=config
    )


def load_val_data(config, logger):
    """Load CIFAR-10 validation data"""
    logger.log_message("Loading validation data...")
    val_batch = utils.unpickle(val_dataset_dir)
    logger.log_message("Validation data loaded successfully.")
    return CIFARDataset(
        val_batch[b"data"],
        val_batch[b"labels"],
        transform=get_val_transforms(True),
        config=config
    )


def load_test_data(config):
    """Load CIFAR-10 test data"""
    test_batch = utils.unpickle(test_dataset_dir)
    return CIFARTestDataset(
        test_batch[b"data"], transform=get_val_transforms(True)
    )


def get_data_loaders(config, logger):
    """Get train and validation data loaders"""
    logger.log_message("Initializing data loaders...")
    train_dataset = load_train_data(config, logger)
    val_dataset = load_val_data(config, logger)

    device = utils.get_device()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(4, os.cpu_count()) if device == "mps" else 16,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"] / 2),
        shuffle=False,
        num_workers=min(4, os.cpu_count()) if device == "mps" else 16,
        pin_memory=(device == "cuda"),
    )

    logger.log_message("Data loaders initialized successfully.")
    return train_loader, val_loader


def get_test_data_loader(config, logger):
    """Get test data loader"""
    logger.log_message("Loading test data...")
    test_batch = utils.unpickle(test_dataset_dir)
    test_dataset = CIFARTestDataset(
        test_batch[b"data"], transform=get_val_transforms(config.get("add_normalization", True))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"] / 2),
        shuffle=False,
        num_workers=16,
        pin_memory=torch.cuda.is_available(),
    )
    logger.log_message("Test data loader initialized successfully.")
    return test_loader


def save_sample_images(config, output_dir):
    """Save sample images from the dataset for visualization"""
    data_config = {
        "add_augmentation": False,
        "add_normalization": False
    }

    # Load train and test datasets without transforms
    train_dataset = load_train_data(data_config)
    test_dataset = load_test_data(data_config)

    # Class names for CIFAR-10
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Create directory for sample images
    os.makedirs(os.path.join(output_dir, "sample_images"), exist_ok=True)

    # Function to save sample images for each class
    def save_samples(dataset, prefix):
        # Group indices by class
        class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        # Save 10 images from each class
        for class_idx, indices in class_indices.items():
            for i, idx in enumerate(indices[:10]):
                img, _ = dataset[idx]
                # Convert tensor to PIL Image
                from PIL import Image

                img_pil = Image.fromarray(img)
                img_path = os.path.join(
                    output_dir,
                    "sample_images",
                    f"{prefix}_{class_names[class_idx]}_{i}.png",
                )
                img_pil.save(img_path)

    # Save sample train and test images
    save_samples(train_dataset, "train")
    save_samples(test_dataset, "test")
