# -*- coding: utf-8 -*-
"""U-Net avec encodeur ResNet-18 pré-entraîné pour la segmentation sémantique.

L'utilisateur annote les zones (eau, terre, ciel, bateau moteur, voilier…)
puis le modèle apprend à segmenter automatiquement de nouvelles images.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DecoderBlock(nn.Module):
    """Bloc de décodeur : upsample → concat skip → 2×(Conv3×3 + BN + ReLU)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net avec encodeur ResNet-18 pré-entraîné.

    Entrée  : (B, 3, H, W)  — image RGB normalisée
    Sortie  : (B, num_classes, H, W) — logits par pixel
    """

    def __init__(self, num_classes: int = config.NUM_SEG_CLASSES, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Encodeur = couches ResNet-18
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # → 64, H/2
        self.pool0 = resnet.maxpool                                        # → 64, H/4
        self.enc1 = resnet.layer1   # → 64,  H/4
        self.enc2 = resnet.layer2   # → 128, H/8
        self.enc3 = resnet.layer3   # → 256, H/16
        self.enc4 = resnet.layer4   # → 512, H/32

        # Décodeur
        self.dec4 = DecoderBlock(512, 256, 256)   # H/32 → H/16
        self.dec3 = DecoderBlock(256, 128, 128)   # H/16 → H/8
        self.dec2 = DecoderBlock(128, 64, 64)     # H/8  → H/4
        self.dec1 = DecoderBlock(64, 64, 64)      # H/4  → H/2

        # Couche finale : H/2 → H
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)          # 64, H/2
        p0 = self.pool0(e0)        # 64, H/4
        e1 = self.enc1(p0)         # 64, H/4
        e2 = self.enc2(e1)         # 128, H/8
        e3 = self.enc3(e2)         # 256, H/16
        e4 = self.enc4(e3)         # 512, H/32

        # Decoder avec skip connections
        d4 = self.dec4(e4, e3)     # 256, H/16
        d3 = self.dec3(d4, e2)     # 128, H/8
        d2 = self.dec2(d3, e1)     # 64, H/4
        d1 = self.dec1(d2, e0)     # 64, H/2

        out = self.final_up(d1)    # 32, H
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(out)

    def predict(self, x):
        """Retourne les classes prédites par pixel (argmax)."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str = None, num_classes: int = config.NUM_SEG_CLASSES):
        path = path or os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
        model = cls(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model


def train_unet(train_loader, val_loader=None,
               epochs: int = config.UNET_EPOCHS,
               lr: float = config.UNET_LEARNING_RATE,
               device: str = None) -> UNet:
    """Entraîne le U-Net sur les paires (image, masque annoté)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(pretrained=True).to(device)

    # Geler l'encodeur au début pour ne pas détruire les features pré-entraînées
    for param in model.enc0.parameters():
        param.requires_grad = False
    for param in model.enc1.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config.UNET_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Dégeler l'encodeur après 20 epochs pour fine-tuning
        if epoch == 20:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1,
                                         weight_decay=config.UNET_WEIGHT_DECAY)
            print("  [epoch 20] Encodeur dégelé, fine-tuning complet")

        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            msg = f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}"
            if val_loader:
                val_loss = _evaluate(model, val_loader, criterion, device)
                msg += f" — val_loss: {val_loss:.4f}"
            print(msg)

        if patience_counter >= 25:
            print(f"  Early stopping à l'epoch {epoch+1}")
            break

    return model


def _evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total += criterion(logits, masks).item()
    return total / max(len(loader), 1)


def predict_image(model: UNet, image_tensor: torch.Tensor,
                  device: str = None) -> np.ndarray:
    """Prédit le masque de segmentation pour une image.

    Args:
        image_tensor: (3, H, W) normalisé
    Returns:
        mask: (H, W) avec les indices de classes
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def mask_to_colored(mask: np.ndarray) -> np.ndarray:
    """Convertit un masque d'indices de classes en image RGB colorée."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_info in config.SEGMENTATION_CLASSES:
        cls_id = cls_info["id"]
        hex_color = cls_info["color"]
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        colored[mask == cls_id] = [r, g, b]
    return colored
