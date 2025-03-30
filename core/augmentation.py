import torch
import numpy as np
from torchvision import transforms

class CutMix:
    """CutMix augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, img1, img2, label1, label2):
        lam = np.random.beta(self.alpha, self.alpha)
        h, w = img1.size(1), img1.size(2)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        img1[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))

        return img1, label1, label2, lam


class Mixup:
    """MixUp augmentation"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def __call__(self, img1, img2, label1, label2):
        lam = np.random.beta(self.alpha, self.alpha)
        img = lam * img1 + (1 - lam) * img2
        return img, label1, label2, lam


def get_augmentations(config):
    """Get augmentations based on configuration"""
    augmentations = {}

    if "CutMix" in config.get("data_augmentation", []):
        augmentations["CutMix"] = CutMix(alpha=1.0)
    if "MixUp" in config.get("data_augmentation", []):
        augmentations["MixUp"] = Mixup(alpha=0.5)
    if "AutoAugment" in config.get("data_augmentation", []):
        augmentations["AutoAugment"] = transforms.AutoAugment()
    if "RandAugment" in config.get("data_augmentation", []):
        augmentations["RandAugment"] = transforms.RandAugment()

    return augmentations
