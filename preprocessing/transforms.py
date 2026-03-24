# -*- coding: utf-8 -*-
"""Dataset PyTorch et utilitaires de chargement pour les images d'embarcations."""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class BoatDataset(Dataset):
    """Dataset d'images d'embarcations avec labels optionnels."""

    def __init__(self, root_dir: str = None, transform=None, labels_path: str = None):
        self.root_dir = root_dir or config.IMAGES_DIR
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(self.root_dir)
            if f.lower().endswith(config.IMAGE_EXTENSIONS)
        ])
        self.labels = {}
        lp = labels_path or config.LABELS_PATH
        if os.path.isfile(lp):
            with open(lp, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.root_dir, filename)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels.get(filename, {}).get("label", "non_etiqueté")
        return image, label, filename


def get_dataloader(dataset: Dataset, batch_size: int = None,
                   shuffle: bool = True) -> DataLoader:
    bs = batch_size or config.AE_BATCH_SIZE
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=0)


def list_images(images_dir: str = None) -> list:
    """Liste les noms de fichiers images dans le dossier."""
    d = images_dir or config.IMAGES_DIR
    if not os.path.isdir(d):
        return []
    return sorted([
        f for f in os.listdir(d)
        if f.lower().endswith(config.IMAGE_EXTENSIONS)
    ])
