# -*- coding: utf-8 -*-
"""Extraction de features via CNN pré-entraîné (ResNet-18 par défaut)."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class FeatureExtractor(nn.Module):
    """Wrapper autour d'un CNN pré-entraîné pour l'extraction de features."""

    def __init__(self, backbone: str = config.BACKBONE):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512

        self.features = nn.Sequential(*list(base.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feat = self.features(x)
        return feat.squeeze(-1).squeeze(-1)


def extract_features(dataloader, backbone: str = config.BACKBONE,
                     device: str = None) -> tuple:
    """Extrait les features CNN pour toutes les images du dataloader.

    Returns:
        (features_array, filenames_list)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FeatureExtractor(backbone).to(device)
    model.eval()
    all_features = []
    all_filenames = []

    with torch.no_grad():
        for images, _, filenames in dataloader:
            images = images.to(device)
            feats = model(images)
            all_features.append(feats.cpu().numpy())
            all_filenames.extend(filenames)

    return np.vstack(all_features), all_filenames
