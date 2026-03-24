# -*- coding: utf-8 -*-
"""Auto-encodeur convolutionnel pour l'extraction de représentations latentes."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 224→112
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 112→56
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 56→28
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 28→14
            nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.flatten_size = 256 * 14 * 14
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, latent_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 14→28
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 28→56
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 56→112
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 112→224
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 14, 14)
        return self.deconv(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def save(self, path: str = None):
        path = path or os.path.join(config.MODELS_SAVE_DIR, "autoencoder.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str = None, latent_dim: int = config.LATENT_DIM):
        path = path or os.path.join(config.MODELS_SAVE_DIR, "autoencoder.pth")
        model = cls(latent_dim)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model


def train_autoencoder(dataloader, epochs: int = config.AE_EPOCHS,
                      lr: float = config.AE_LEARNING_RATE,
                      device: str = None) -> ConvAutoencoder:
    """Entraîne l'auto-encodeur et retourne le modèle entraîné."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=config.AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=False
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _, _ in dataloader:
            images = images.to(device)
            recon, _ = model(images)
            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.6f}")

        if patience_counter >= 20:
            print(f"  Early stopping à l'epoch {epoch+1}")
            break

    return model


def get_latent_vectors(model: ConvAutoencoder, dataloader,
                       device: str = None) -> tuple:
    """Extrait les vecteurs latents pour toutes les images du dataloader."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    all_vectors = []
    all_filenames = []

    with torch.no_grad():
        for images, _, filenames in dataloader:
            images = images.to(device)
            z = model.encode(images)
            all_vectors.append(z.cpu().numpy())
            all_filenames.extend(filenames)

    return np.vstack(all_vectors), all_filenames
