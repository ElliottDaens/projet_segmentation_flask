# -*- coding: utf-8 -*-
"""Analyse par régions : superpixels SLIC, détection de contours, anomalies locales."""

import os
import sys
import io
import base64
import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage import color

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_superpixels(image: np.ndarray,
                        n_segments: int = config.SLIC_N_SEGMENTS,
                        compactness: float = config.SLIC_COMPACTNESS) -> np.ndarray:
    """Calcule les superpixels SLIC sur une image RGB."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    segments = slic(image, n_segments=n_segments, compactness=compactness,
                    start_label=0, channel_axis=2)
    return segments


def extract_region_features(image: np.ndarray, segments: np.ndarray) -> list:
    """Extrait les features pour chaque région (superpixel).

    Features par région : couleur moyenne RGB, écart-type, position (centroïde),
    taille relative, ratio d'aspect de la bounding box.
    """
    regions = []
    unique_labels = np.unique(segments)
    h, w = image.shape[:2]
    total_pixels = h * w

    for label_id in unique_labels:
        mask = segments == label_id
        pixels = image[mask]
        ys, xs = np.where(mask)

        mean_color = pixels.mean(axis=0).tolist()
        std_color = pixels.std(axis=0).tolist()
        centroid_y = float(ys.mean()) / h
        centroid_x = float(xs.mean()) / w
        size_ratio = float(mask.sum()) / total_pixels

        min_y, max_y = int(ys.min()), int(ys.max())
        min_x, max_x = int(xs.min()), int(xs.max())
        bbox_h = max(max_y - min_y, 1)
        bbox_w = max(max_x - min_x, 1)
        aspect_ratio = bbox_w / bbox_h

        regions.append({
            "label_id": int(label_id),
            "mean_color": mean_color,
            "std_color": std_color,
            "centroid": [centroid_x, centroid_y],
            "size_ratio": size_ratio,
            "aspect_ratio": aspect_ratio,
            "bbox": [min_x, min_y, max_x, max_y],
            "n_pixels": int(mask.sum()),
        })

    return regions


def cluster_regions(region_features: list, n_clusters: int = 5) -> list:
    """Clustering des régions par leurs features."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feature_matrix = []
    for r in region_features:
        feat = r["mean_color"] + r["std_color"] + r["centroid"] + \
               [r["size_ratio"], r["aspect_ratio"]]
        feature_matrix.append(feat)

    feature_matrix = np.array(feature_matrix)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    n_clusters = min(n_clusters, len(region_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    for i, r in enumerate(region_features):
        r["region_cluster"] = int(labels[i])

    return region_features


def detect_anomaly_regions(region_features: list, threshold: float = 2.0) -> list:
    """Détecte les régions anomales (outliers) via z-score sur les features."""
    from sklearn.preprocessing import StandardScaler

    feature_matrix = []
    for r in region_features:
        feat = r["mean_color"] + [r["size_ratio"], r["aspect_ratio"]]
        feature_matrix.append(feat)

    feature_matrix = np.array(feature_matrix)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    distances = np.linalg.norm(scaled, axis=1)
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    mad = max(mad, 1e-6)
    z_scores = (distances - median_dist) / (1.4826 * mad)

    anomalies = []
    for i, r in enumerate(region_features):
        r["anomaly_score"] = float(z_scores[i])
        r["is_anomaly"] = bool(z_scores[i] > threshold)
        if r["is_anomaly"]:
            anomalies.append(r)

    return anomalies


def overlay_regions(image: np.ndarray, segments: np.ndarray,
                    region_features: list = None,
                    highlight_anomalies: bool = True) -> np.ndarray:
    """Crée un overlay visuel des régions sur l'image.

    Les anomalies sont surlignées en rouge, les régions normales ont un contour léger.
    """
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    overlay = mark_boundaries(image, segments, color=(0.3, 0.3, 0.8), mode="outer")
    overlay = (overlay * 255).astype(np.uint8)

    if highlight_anomalies and region_features:
        for r in region_features:
            if r.get("is_anomaly", False):
                mask = segments == r["label_id"]
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * 0.5 +
                    np.array([255, 60, 60], dtype=np.float32) * 0.5
                ).astype(np.uint8)

    return overlay


def detect_contours(image: np.ndarray,
                    low: int = config.CANNY_LOW,
                    high: int = config.CANNY_HIGH) -> np.ndarray:
    """Détection de contours Canny sur l'image."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


def image_to_base64(image: np.ndarray) -> str:
    """Convertit une image numpy en string base64 pour l'affichage HTML."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyze_image_regions(image_path: str,
                          n_segments: int = config.SLIC_N_SEGMENTS,
                          n_region_clusters: int = 5) -> dict:
    """Pipeline complet d'analyse par régions pour une image.

    Returns:
        dict avec l'image originale, overlay, contours, features et anomalies
        (tout en base64 pour l'affichage HTML).
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    segments = compute_superpixels(image, n_segments=n_segments)
    region_features = extract_region_features(image, segments)
    region_features = cluster_regions(region_features, n_clusters=n_region_clusters)
    anomalies = detect_anomaly_regions(region_features)
    overlay = overlay_regions(image, segments, region_features, highlight_anomalies=True)
    edges = detect_contours(image)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return {
        "original_b64": image_to_base64(image),
        "overlay_b64": image_to_base64(overlay),
        "contours_b64": image_to_base64(edges_rgb),
        "n_regions": len(region_features),
        "n_anomalies": len(anomalies),
        "n_region_clusters": n_region_clusters,
        "region_features": region_features,
        "anomalies": anomalies,
    }
