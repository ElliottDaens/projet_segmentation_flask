# -*- coding: utf-8 -*-
"""Script d'entraînement du U-Net pour la segmentation sémantique."""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from preprocessing.segmentation_dataset import get_seg_dataloader, count_annotated_images
from models.unet import train_unet


def main():
    parser = argparse.ArgumentParser(description="Entraînement U-Net segmentation sémantique")
    parser.add_argument("--epochs", type=int, default=config.UNET_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.UNET_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.UNET_BATCH_SIZE)
    args = parser.parse_args()

    n_masks = count_annotated_images()
    print(f"=== Entraînement U-Net (segmentation sémantique) ===")
    print(f"  Masques annotés : {n_masks}")
    print(f"  Classes : {config.NUM_SEG_CLASSES}")
    print(f"  Taille image : {config.SEG_IMAGE_SIZE}×{config.SEG_IMAGE_SIZE}")
    print(f"  Epochs : {args.epochs}")
    print(f"  LR : {args.lr}")
    print()

    if n_masks < 3:
        print(f"  ERREUR : seulement {n_masks} masques annotés.")
        print("  Annotez au moins 3 images via l'interface web (/annotation).")
        return

    loader = get_seg_dataloader(batch_size=args.batch_size, augment=True)
    print(f"  Dataset : {len(loader.dataset)} paires (image, masque)")
    print(f"  Batches : {len(loader)}")
    print()

    model = train_unet(loader, epochs=args.epochs, lr=args.lr)
    print(f"\n  Modèle sauvegardé : {config.MODELS_SAVE_DIR}/unet.pth")
    print("  Vous pouvez maintenant utiliser la prédiction dans l'interface web (/prediction).")


if __name__ == "__main__":
    main()
