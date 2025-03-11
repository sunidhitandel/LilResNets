from torchvision import transforms

def get_train_transforms(config):
    train_transforms = [transforms.ToPILImage()]

    if config.add_augmentation:
        train_transforms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])
    if config.add_jitter:
        train_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    train_transforms.append(transforms.ToTensor())

    if config.add_normalization:
        train_transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    return transforms.Compose(train_transforms)


def get_test_transforms(add_normalization: bool = True):
    test_transforms = [transforms.ToPILImage(), transforms.ToTensor()]

    if add_normalization:
        test_transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    return transforms.Compose(test_transforms)