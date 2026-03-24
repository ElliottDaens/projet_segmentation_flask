# -*- coding: utf-8 -*-
"""Calcul et sauvegarde des embeddings (CNN pré-entraîné ou auto-encodeur)."""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def main():
    parser = argparse.ArgumentParser(description="Calcul des embeddings d'images")
    parser.add_argument("--method", choices=["cnn", "autoencoder"], default="cnn")
    args = parser.parse_args()

    print(f"=== Calcul des embeddings ({args.method}) ===")

    if args.method == "cnn":
        from preprocessing.augmentation import get_eval_transforms
        from preprocessing.transforms import BoatDataset, get_dataloader
        from models.feature_extractor import extract_features

        dataset = BoatDataset(transform=get_eval_transforms())
        print(f"  Dataset : {len(dataset)} images")
        loader = get_dataloader(dataset, shuffle=False)
        embeddings, filenames = extract_features(loader)

    else:
        from preprocessing.augmentation import get_autoencoder_transforms
        from preprocessing.transforms import BoatDataset, get_dataloader
        from models.autoencoder import ConvAutoencoder, get_latent_vectors

        ae_path = os.path.join(config.MODELS_SAVE_DIR, "autoencoder.pth")
        if not os.path.isfile(ae_path):
            print("  ERREUR : auto-encodeur non entraîné.")
            print("  Lancez d'abord : python scripts/train_autoencoder.py")
            return

        model = ConvAutoencoder.load(ae_path)
        dataset = BoatDataset(transform=get_autoencoder_transforms())
        print(f"  Dataset : {len(dataset)} images")
        loader = get_dataloader(dataset, shuffle=False)
        embeddings, filenames = get_latent_vectors(model, loader)

    print(f"  Embeddings : {embeddings.shape}")

    np.save(os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(config.EMBEDDINGS_DIR, "filenames.json"), "w") as f:
        json.dump(filenames, f)

    # Réduction UMAP
    print("  Réduction UMAP (2D)…")
    from clustering.pipeline import reduce_dimensions
    embeddings_2d = reduce_dimensions(embeddings, method="umap")
    np.save(os.path.join(config.EMBEDDINGS_DIR, "embeddings_2d.npy"), embeddings_2d)

    print(f"  Sauvegardé dans {config.EMBEDDINGS_DIR}/")
    print("  Terminé.")


if __name__ == "__main__":
    main()
