# -*- coding: utf-8 -*-
"""Dataset pour la segmentation sémantique — pré-chargé en RAM, augmentation légère."""

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
    """Dataset pré-chargé en RAM pour éliminer les I/O disque pendant l'entraînement.

    Avec 3-13 images, tout tient facilement en mémoire (~50 MB).
    Les images et masques sont chargés UNE FOIS au __init__, puis
    les augmentations sont appliquées sur les tensors déjà en mémoire.
    """

    def __init__(self, images_dir: str = None, masks_dir: str = None,
                 image_size: int = None, augment: bool = True):
        self.image_size = image_size or config.SEG_IMAGE_SIZE
        self.augment = augment
        self.repeat = config.UNET_DATASET_REPEAT if augment else 1
        images_dir = images_dir or config.IMAGES_DIR
        masks_dir = masks_dir or config.MASKS_DIR

        mask_files = set()
        if os.path.isdir(masks_dir):
            mask_files = {f for f in os.listdir(masks_dir) if f.lower().endswith(".png")}

        # Pré-charger toutes les paires en RAM comme tensors
        self._images = []
        self._masks = []
        self._names = []

        if os.path.isdir(images_dir):
            for f in sorted(os.listdir(images_dir)):
                if not f.lower().endswith(config.IMAGE_EXTENSIONS):
                    continue
                mask_name = os.path.splitext(f)[0] + ".png"
                if mask_name not in mask_files:
                    continue

                img = Image.open(os.path.join(images_dir, f)).convert("RGB")
                img = TF.resize(img, [self.image_size, self.image_size])
                img_tensor = TF.to_tensor(img)

                mask = Image.open(os.path.join(masks_dir, mask_name)).convert("L")
                mask = TF.resize(mask, [self.image_size, self.image_size],
                                 interpolation=TF.InterpolationMode.NEAREST)
                mask_tensor = torch.from_numpy(np.array(mask)).long()
                mask_tensor[mask_tensor >= config.NUM_SEG_CLASSES] = config.IGNORE_INDEX

                self._images.append(img_tensor)
                self._masks.append(mask_tensor)
                self._names.append(f)

    def __len__(self):
        return len(self._images) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self._images)
        image = self._images[real_idx].clone()
        mask = self._masks[real_idx].clone()

        if self.augment:
            image, mask = self._augment(image, mask)

        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        return image, mask

    def _augment(self, image, mask):
        """Augmentation sur tensors — pas de PIL, pas d'I/O."""
        # image: (3, H, W) float, mask: (H, W) long
        mask_4d = mask.unsqueeze(0).unsqueeze(0).float()

        if random.random() > 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [1])

        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle, fill=0)
            mask_rot = TF.rotate(mask_4d, angle, fill=float(config.IGNORE_INDEX),
                                 interpolation=TF.InterpolationMode.NEAREST)
            mask = mask_rot.squeeze().long()

        if random.random() > 0.4:
            image = TF.adjust_brightness(image, random.uniform(0.6, 1.4))
        if random.random() > 0.4:
            image = TF.adjust_contrast(image, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        image = image.clamp(0, 1)
        return image, mask


def get_seg_dataloader(images_dir=None, masks_dir=None,
                       batch_size=None, augment=True, shuffle=True):
    ds = SegmentationDataset(images_dir, masks_dir, augment=augment)
    bs = batch_size or config.UNET_BATCH_SIZE
    use_cuda = torch.cuda.is_available()
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0,
                      pin_memory=use_cuda, drop_last=False,
                      persistent_workers=False)


def count_annotated_images(masks_dir=None) -> int:
    d = masks_dir or config.MASKS_DIR
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith(".png"))
