# -*- coding: utf-8 -*-
"""Normalisation d'images : CLAHE, histogram equalization, balance des blancs."""

import cv2
import numpy as np
from PIL import Image


def clahe_equalization(image: np.ndarray, clip_limit: float = 2.0,
                       tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Applique CLAHE (Contrast Limited Adaptive Histogram Equalization) sur chaque canal."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Histogram equalization sur le canal L (espace LAB)."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def white_balance(image: np.ndarray) -> np.ndarray:
    """Balance des blancs simple par gray-world assumption."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    result = image.astype(np.float32)
    avg_b, avg_g, avg_r = result[:, :, 0].mean(), result[:, :, 1].mean(), result[:, :, 2].mean()
    avg_gray = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] *= avg_gray / max(avg_b, 1e-6)
    result[:, :, 1] *= avg_gray / max(avg_g, 1e-6)
    result[:, :, 2] *= avg_gray / max(avg_r, 1e-6)
    return result.clip(0, 255).astype(np.uint8)


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Débruitage non-local means."""
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def normalize_image(image: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Pipeline de normalisation combiné."""
    methods = {
        "clahe": clahe_equalization,
        "histeq": histogram_equalization,
        "white_balance": white_balance,
        "denoise": denoise,
    }
    fn = methods.get(method, clahe_equalization)
    return fn(image)


def pil_normalize(pil_image: Image.Image, method: str = "clahe") -> Image.Image:
    """Wrapper PIL → normalisation → PIL."""
    arr = np.array(pil_image.convert("RGB"))
    normalized = normalize_image(arr, method)
    return Image.fromarray(normalized)
