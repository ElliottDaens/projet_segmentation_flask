# -*- coding: utf-8 -*-
"""Pipeline de clustering : K-Means, DBSCAN, réduction de dimension, semi-supervisé."""

import os
import sys
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def reduce_dimensions(embeddings: np.ndarray, method: str = "umap",
                      n_components: int = 2) -> np.ndarray:
    """Réduit la dimension des embeddings pour visualisation.

    Pipeline : StandardScaler → PCA (si dim > 50) → UMAP ou t-SNE.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    if scaled.shape[1] > config.PCA_N_COMPONENTS:
        pca = PCA(n_components=min(config.PCA_N_COMPONENTS, scaled.shape[0] - 1))
        scaled = pca.fit_transform(scaled)

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_neighbors=min(config.UMAP_N_NEIGHBORS, len(scaled) - 1),
            min_dist=config.UMAP_MIN_DIST,
            n_components=n_components,
            random_state=42,
        )
        return reducer.fit_transform(scaled)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, len(scaled) - 1)
        tsne = TSNE(n_components=n_components, perplexity=max(perplexity, 2),
                     random_state=42)
        return tsne.fit_transform(scaled)

    elif method == "pca":
        pca = PCA(n_components=n_components)
        return pca.fit_transform(scaled)

    return scaled[:, :n_components]


def run_kmeans(embeddings: np.ndarray, n_clusters: int = config.N_CLUSTERS) -> dict:
    """Exécute K-Means et retourne les résultats."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(scaled)
    return {
        "labels": labels.tolist(),
        "centers": kmeans.cluster_centers_.tolist(),
        "inertia": float(kmeans.inertia_),
        "n_clusters": n_clusters,
        "method": "kmeans",
    }


def run_dbscan(embeddings: np.ndarray, eps: float = config.DBSCAN_EPS,
               min_samples: int = config.DBSCAN_MIN_SAMPLES) -> dict:
    """Exécute DBSCAN (utile pour détecter des anomalies / classes rares)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "eps": eps,
        "min_samples": min_samples,
        "method": "dbscan",
    }


def find_optimal_k(embeddings: np.ndarray,
                   k_range: range = config.K_RANGE) -> dict:
    """Méthode du coude + silhouette pour trouver K optimal."""
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    inertias = []
    silhouettes = []
    k_values = list(k_range)

    for k in k_values:
        if k >= len(embeddings):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        inertias.append(float(kmeans.inertia_))
        if k > 1 and k < len(embeddings):
            sil = float(silhouette_score(scaled, labels))
        else:
            sil = 0.0
        silhouettes.append(sil)

    best_k_idx = int(np.argmax(silhouettes)) if silhouettes else 0
    return {
        "k_values": k_values[:len(inertias)],
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": k_values[best_k_idx],
    }


def semi_supervised_clustering(embeddings: np.ndarray, partial_labels: dict,
                               filenames: list,
                               n_clusters: int = config.N_CLUSTERS) -> dict:
    """Clustering guidé par les labels existants.

    Stratégie : initialiser les centres K-Means à partir des moyennes
    des embeddings des images déjà étiquetées.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    label_groups = {}
    for i, fname in enumerate(filenames):
        if fname in partial_labels:
            lbl = partial_labels[fname].get("label", "")
            if lbl and lbl != "non_etiqueté":
                label_groups.setdefault(lbl, []).append(i)

    if len(label_groups) >= 2:
        init_centers = []
        for lbl in sorted(label_groups.keys()):
            indices = label_groups[lbl]
            center = scaled[indices].mean(axis=0)
            init_centers.append(center)
        init_centers = np.array(init_centers)
        actual_k = len(init_centers)
        kmeans = KMeans(n_clusters=actual_k, init=init_centers, n_init=1,
                        max_iter=300, random_state=42)
    else:
        actual_k = n_clusters
        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(scaled)
    return {
        "labels": labels.tolist(),
        "n_clusters": actual_k,
        "guided": len(label_groups) >= 2,
        "label_groups_used": list(label_groups.keys()),
        "method": "semi_supervised_kmeans",
    }
