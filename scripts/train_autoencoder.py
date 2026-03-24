# -*- coding: utf-8 -*-
"""Script d'entraînement de l'auto-encodeur convolutionnel."""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from preprocessing.augmentation import get_autoencoder_train_transforms, get_autoencoder_transforms
from preprocessing.transforms import BoatDataset, get_dataloader
from models.autoencoder import train_autoencoder, ConvAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Entraînement de l'auto-encodeur convolutionnel")
    parser.add_argument("--epochs", type=int, default=config.AE_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.AE_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.AE_BATCH_SIZE)
    parser.add_argument("--latent-dim", type=int, default=config.LATENT_DIM)
    args = parser.parse_args()

    print(f"=== Entraînement auto-encodeur ===")
    print(f"  Images : {config.IMAGES_DIR}")
    print(f"  Epochs : {args.epochs}")
    print(f"  LR     : {args.lr}")
    print(f"  Latent : {args.latent_dim}D")
    print()

    train_dataset = BoatDataset(transform=get_autoencoder_train_transforms())
    print(f"  Dataset : {len(train_dataset)} images")

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = train_autoencoder(train_loader, epochs=args.epochs, lr=args.lr)

    print(f"\n  Modèle sauvegardé dans {config.MODELS_SAVE_DIR}/autoencoder.pth")

    # Vérification rapide de la reconstruction
    eval_dataset = BoatDataset(transform=get_autoencoder_transforms())
    eval_loader = get_dataloader(eval_dataset, batch_size=1, shuffle=False)

    import torch
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    total_mse = 0
    with torch.no_grad():
        for images, _, _ in eval_loader:
            images = images.to(device)
            recon, _ = model(images)
            mse = ((recon - images) ** 2).mean().item()
            total_mse += mse
    avg_mse = total_mse / len(eval_loader)
    print(f"  MSE moyenne de reconstruction : {avg_mse:.6f}")


if __name__ == "__main__":
    main()
