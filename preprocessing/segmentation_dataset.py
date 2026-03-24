# -*- coding: utf-8 -*-
"""Dataset PyTorch pour la segmentation sémantique (paires image + masque annoté)."""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class SegmentationDataset(Dataset):
    """Dataset de paires (image, masque) pour l'entraînement du U-Net.

    Les masques sont des images en niveaux de gris où chaque pixel = indice de classe (0-9).
    Les pixels non annotés ont la valeur 255 (ignorés dans la loss).
    """

    def __init__(self, images_dir: str = None, masks_dir: str = None,
                 image_size: int = None, augment: bool = True):
        self.images_dir = images_dir or config.IMAGES_DIR
        self.masks_dir = masks_dir or config.MASKS_DIR
        self.image_size = image_size or config.SEG_IMAGE_SIZE
        self.augment = augment

        mask_files = set()
        if os.path.isdir(self.masks_dir):
            for f in os.listdir(self.masks_dir):
                if f.lower().endswith(".png"):
                    mask_files.add(f)

        self.pairs = []
        if os.path.isdir(self.images_dir):
            for f in sorted(os.listdir(self.images_dir)):
                if f.lower().endswith(config.IMAGE_EXTENSIONS):
                    mask_name = os.path.splitext(f)[0] + ".png"
                    if mask_name in mask_files:
                        self.pairs.append((f, mask_name))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size],
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask >= config.NUM_SEG_CLASSES] = config.IGNORE_INDEX

        return image, mask

    def _augment(self, image, mask):
        """Augmentation spatiale identique sur image et masque."""
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=255)

        if random.random() > 0.5:
            i, j, h, w = self._random_crop_params(image, scale=(0.7, 1.0))
            image = TF.resized_crop(image, i, j, h, w,
                                    [self.image_size, self.image_size])
            mask = TF.resized_crop(mask, i, j, h, w,
                                   [self.image_size, self.image_size],
                                   interpolation=TF.InterpolationMode.NEAREST)

        # Augmentation couleur (image seulement)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        return image, mask

    def _random_crop_params(self, img, scale=(0.7, 1.0)):
        w, h = img.size
        area = h * w
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(0.8, 1.2)
        new_w = int(round((target_area * aspect) ** 0.5))
        new_h = int(round((target_area / aspect) ** 0.5))
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        i = random.randint(0, h - new_h)
        j = random.randint(0, w - new_w)
        return i, j, new_h, new_w


def get_seg_dataloader(images_dir=None, masks_dir=None,
                       batch_size=None, augment=True, shuffle=True):
    ds = SegmentationDataset(images_dir, masks_dir, augment=augment)
    bs = batch_size or config.UNET_BATCH_SIZE
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)


def count_annotated_images(masks_dir=None) -> int:
    d = masks_dir or config.MASKS_DIR
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith(".png"))
