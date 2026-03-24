# -*- coding: utf-8 -*-
"""U-Net avec encodeur ResNet-18 pré-entraîné pour la segmentation sémantique."""

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
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
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
    def __init__(self, num_classes: int = config.NUM_SEG_CLASSES, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0 = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        p0 = self.pool0(e0)
        e1 = self.enc1(p0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        out = self.final_up(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(out)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

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


def _compute_class_weights(train_loader, device) -> torch.Tensor:
    """Calcule les poids par classe sur GPU — les classes rares pèsent plus."""
    counts = torch.zeros(config.NUM_SEG_CLASSES, device=device)
    for _, masks in train_loader:
        masks_gpu = masks.to(device, non_blocking=True)
        for c in range(config.NUM_SEG_CLASSES):
            counts[c] += (masks_gpu == c).sum().float()

    total = counts.sum()
    if total == 0:
        return torch.ones(config.NUM_SEG_CLASSES, device=device)

    weights = total / (config.NUM_SEG_CLASSES * counts.clamp(min=1))
    return weights.clamp(max=10.0)


def train_unet(train_loader, val_loader=None,
               epochs: int = config.UNET_EPOCHS,
               lr: float = config.UNET_LEARNING_RATE,
               device: str = None) -> UNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = (device == "cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    model = UNet(pretrained=True).to(device)

    for name, param in model.named_parameters():
        if name.startswith("enc"):
            param.requires_grad = False

    import sys
    def _print(msg):
        print(msg)
        sys.stdout.flush()

    _print(f"  Device: {device}" + (" (mixed precision fp16)" if use_cuda else ""))
    _print(f"  Calcul des poids par classe...")
    class_weights = _compute_class_weights(train_loader, device)
    _print(f"  Poids: {', '.join(f'{w:.1f}' for w in class_weights.tolist())}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config.UNET_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=config.IGNORE_INDEX
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    scaler = torch.amp.GradScaler(device, enabled=use_cuda)

    best_loss = float("inf")
    patience_counter = 0
    unfreeze_epoch = config.UNET_UNFREEZE_EPOCH

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            for name, param in model.named_parameters():
                if name.startswith("enc3") or name.startswith("enc4"):
                    param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters()
                            if not (n.startswith("enc3") or n.startswith("enc4")) and p.requires_grad],
                 "lr": lr * 0.3},
                {"params": [p for n, p in model.named_parameters()
                            if (n.startswith("enc3") or n.startswith("enc4")) and p.requires_grad],
                 "lr": lr * 0.1},
            ], weight_decay=config.UNET_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2
            )
            _print(f"  [epoch {unfreeze_epoch}] enc3+enc4 dégelés (LR ×0.1)")

        if epoch == unfreeze_epoch + 20:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr * 0.05,
                weight_decay=config.UNET_WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2
            )
            _print(f"  [epoch {unfreeze_epoch + 20}] Tout dégelé, fine-tuning complet (LR ×0.05)")

        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=use_cuda):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            msg = f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} — lr: {cur_lr:.1e}"
            if val_loader:
                val_loss = _evaluate(model, val_loader, criterion, device)
                msg += f" — val: {val_loss:.4f}"
            _print(msg)

        if patience_counter >= 35:
            _print(f"  Early stopping à l'epoch {epoch+1}")
            break

    return model


def _evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            total += criterion(model(images), masks).item()
    return total / max(len(loader), 1)


def predict_image(model: UNet, image_tensor: torch.Tensor,
                  device: str = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).argmax(dim=1).squeeze(0).cpu().numpy()


def mask_to_colored(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    colored = np.full((h, w, 3), 51, dtype=np.uint8)
    for cls_info in config.SEGMENTATION_CLASSES:
        cls_id = cls_info["id"]
        if cls_id == 0:
            continue
        hex_color = cls_info["color"]
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        colored[mask == cls_id] = [r, g, b]
    return colored
