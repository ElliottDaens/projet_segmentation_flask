# -*- coding: utf-8 -*-
"""Classifieur supervisé (MLP) sur les embeddings pour affiner les prédictions."""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class EmbeddingClassifier(nn.Module):
    """MLP simple pour classifier les embeddings en catégories d'embarcations."""

    def __init__(self, input_dim: int = config.FEATURE_DIM,
                 hidden_dims: list = None, n_classes: int = None):
        super().__init__()
        hidden_dims = hidden_dims or config.CLASSIFIER_HIDDEN
        n_classes = n_classes or len(config.DEFAULT_LABELS)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(True),
                nn.Dropout(0.3),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_classifier(embeddings: np.ndarray, labels: list,
                     label_names: list = None,
                     epochs: int = config.CLASSIFIER_EPOCHS,
                     lr: float = config.CLASSIFIER_LR) -> tuple:
    """Entraîne le classifieur MLP.

    Returns:
        (model, label_to_idx, idx_to_label)
    """
    label_names = label_names or sorted(set(labels))
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingClassifier(
        input_dim=embeddings.shape[1],
        n_classes=len(label_names)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += len(batch_y)

        if (epoch + 1) % 10 == 0:
            acc = correct / total * 100
            print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/len(loader):.4f} — acc: {acc:.1f}%")

    model.eval()
    save_path = os.path.join(config.MODELS_SAVE_DIR, "classifier.pth")
    torch.save({
        "model_state": model.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "input_dim": embeddings.shape[1],
        "n_classes": len(label_names),
    }, save_path)

    return model, label_to_idx, idx_to_label


def predict(model: EmbeddingClassifier, embeddings: np.ndarray,
            idx_to_label: dict) -> list:
    """Prédit les labels pour un ensemble d'embeddings."""
    device = next(model.parameters()).device
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1).cpu().numpy()
    return [idx_to_label[int(p)] for p in preds], probs.cpu().numpy()
