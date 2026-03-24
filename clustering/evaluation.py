# -*- coding: utf-8 -*-
"""Évaluation de la qualité du clustering et génération de rapports visuels."""

import os
import sys
import json
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_metrics(embeddings: np.ndarray, labels: list) -> dict:
    """Calcule les métriques de qualité du clustering."""
    labels_arr = np.array(labels)
    unique_labels = set(labels_arr)
    unique_labels.discard(-1)

    metrics = {
        "n_clusters": len(unique_labels),
        "n_samples": len(labels),
        "cluster_sizes": {},
    }

    for lbl in sorted(unique_labels):
        metrics["cluster_sizes"][int(lbl)] = int((labels_arr == lbl).sum())

    if len(unique_labels) >= 2 and len(unique_labels) < len(embeddings):
        mask = labels_arr != -1
        if mask.sum() >= 2:
            metrics["silhouette_avg"] = float(
                silhouette_score(embeddings[mask], labels_arr[mask])
            )
            metrics["silhouette_per_sample"] = silhouette_samples(
                embeddings[mask], labels_arr[mask]
            ).tolist()
            metrics["davies_bouldin"] = float(
                davies_bouldin_score(embeddings[mask], labels_arr[mask])
            )

    n_noise = int((labels_arr == -1).sum())
    if n_noise > 0:
        metrics["n_noise"] = n_noise

    return metrics


def generate_cluster_report(embeddings_2d: np.ndarray, labels: list,
                            filenames: list, manual_labels: dict = None) -> dict:
    """Génère un rapport structuré pour l'affichage dans l'interface.

    Returns:
        dict avec clusters, points pour scatter plot, et métriques.
    """
    labels_arr = np.array(labels)
    clusters = {}

    for i, (fname, lbl) in enumerate(zip(filenames, labels)):
        lbl_key = int(lbl)
        if lbl_key not in clusters:
            clusters[lbl_key] = []
        point = {
            "filename": fname,
            "x": float(embeddings_2d[i, 0]),
            "y": float(embeddings_2d[i, 1]),
            "cluster": lbl_key,
        }
        if manual_labels and fname in manual_labels:
            point["manual_label"] = manual_labels[fname].get("label", "")
        clusters[lbl_key].append(point)

    scatter_data = []
    for lbl_key in sorted(clusters.keys()):
        for pt in clusters[lbl_key]:
            scatter_data.append(pt)

    return {
        "clusters": {k: v for k, v in sorted(clusters.items())},
        "scatter_data": scatter_data,
        "n_clusters": len(clusters),
        "total_images": len(filenames),
    }
