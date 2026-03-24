# -*- coding: utf-8 -*-
"""Data augmentation avec torchvision.transforms pour le dataset d'embarcations."""

import torchvision.transforms as T
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_train_transforms(image_size: int = None) -> T.Compose:
    """Transformations d'entraînement avec augmentation agressive."""
    size = image_size or config.IMAGE_SIZE
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=config.ROTATION_RANGE),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        T.ColorJitter(
            brightness=(config.BRIGHTNESS_RANGE[0], config.BRIGHTNESS_RANGE[1]),
            contrast=(config.CONTRAST_RANGE[0], config.CONTRAST_RANGE[1]),
            saturation=(config.SATURATION_RANGE[0], config.SATURATION_RANGE[1]),
            hue=config.HUE_RANGE,
        ),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = None) -> T.Compose:
    """Transformations d'évaluation (sans augmentation)."""
    size = image_size or config.IMAGE_SIZE
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_autoencoder_transforms(image_size: int = None) -> T.Compose:
    """Transformations pour l'auto-encodeur (normalisation [0,1] sans ImageNet stats)."""
    size = image_size or config.IMAGE_SIZE
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])


def get_autoencoder_train_transforms(image_size: int = None) -> T.Compose:
    """Transformations d'entraînement pour l'auto-encodeur avec augmentation légère."""
    size = image_size or config.IMAGE_SIZE
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
    ])
