# -*- coding: utf-8 -*-
"""Système de voisinage : cohérence spatiale, sémantique, hiérarchique et taille."""

import os
import sys
import numpy as np
import cv2
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ── Classe → ID de segmentation ─────────────────────────────────────

_NAME_TO_ID = {c["name"]: c["id"] for c in config.SEGMENTATION_CLASSES}
_ID_TO_NAME = {c["id"]: c["name"] for c in config.SEGMENTATION_CLASSES}

# IDs des 3 zones de scène
_ZONE_IDS = {name: info["seg_id"] for name, info in config.ZONE_CLASSES.items()}


def get_zone_for_class(class_name: str) -> str:
    """Retourne la zone parente d'une classe objet."""
    for zone_name, objects in config.OBJETS_PAR_ZONE.items():
        if class_name in objects:
            return zone_name
    return "mer"


# ── Score de cohérence zone (hiérarchique) ───────────────────────────

def score_zone(object_mask: np.ndarray, zone_mask: np.ndarray,
               class_name: str) -> float:
    """Quel pourcentage de l'objet détecté est dans la bonne zone ?

    Returns:
        float entre 0 et 1 (1 = 100% dans la bonne zone)
    """
    expected_zone = get_zone_for_class(class_name)
    zone_id = _ZONE_IDS.get(expected_zone)
    if zone_id is None:
        return 0.5

    obj_pixels = object_mask.sum()
    if obj_pixels == 0:
        return 0.0

    in_zone = ((object_mask > 0) & (zone_mask == zone_id)).sum()
    return float(in_zone) / float(obj_pixels)


# ── Score de cohérence spatiale (voisinage) ──────────────────────────

def score_voisinage_spatial(object_mask: np.ndarray, full_pred: np.ndarray,
                            class_name: str,
                            rayon: int = config.VOISINAGE_RAYON) -> float:
    """Analyse le voisinage spatial autour de l'objet.

    Vérifie que les classes voisines sont sémantiquement compatibles.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rayon * 2 + 1, rayon * 2 + 1))
    dilated = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=1)
    ring = (dilated > 0) & (object_mask == 0)

    if ring.sum() == 0:
        return 0.5

    neighbor_classes = full_pred[ring]
    neighbor_names = set()
    for cid in np.unique(neighbor_classes):
        name = _ID_TO_NAME.get(int(cid), "")
        if name and name != "fond":
            neighbor_names.add(name)

    if not neighbor_names:
        return 0.7

    compatibles = config.VOISINS_COMPATIBLES.get(class_name, set())
    zone_names = set(config.ZONE_CLASSES.keys())
    # Les zones (mer, terre, ciel) sont toujours compatibles en tant que voisins
    all_compatible = compatibles | {"eau", "terre", "ciel", "fond"}

    n_compat = sum(1 for n in neighbor_names if n in all_compatible)
    return float(n_compat) / float(len(neighbor_names)) if neighbor_names else 0.5


# ── Score de cohérence taille ────────────────────────────────────────

def score_taille(object_mask: np.ndarray, zone_mask: np.ndarray,
                 class_name: str) -> float:
    """Vérifie que la taille de l'objet est cohérente par rapport à la zone."""
    obj_pixels = float(object_mask.sum())
    expected_zone = get_zone_for_class(class_name)
    zone_id = _ZONE_IDS.get(expected_zone)

    if zone_id is None:
        zone_pixels = float(zone_mask.size)
    else:
        zone_pixels = float((zone_mask == zone_id).sum())

    if zone_pixels == 0:
        return 0.0

    ratio = obj_pixels / zone_pixels
    min_ratio, max_ratio = config.TAILLE_ATTENDUE.get(class_name, (0.001, 0.5))

    if min_ratio <= ratio <= max_ratio:
        return 1.0
    elif ratio < min_ratio:
        return max(0.0, ratio / min_ratio)
    else:
        return max(0.0, 1.0 - (ratio - max_ratio) / max_ratio)


# ── Score du modèle ──────────────────────────────────────────────────

def score_modele(proba_map: np.ndarray, object_mask: np.ndarray) -> float:
    """Probabilité moyenne du modèle pour les pixels de l'objet."""
    if object_mask.sum() == 0:
        return 0.0
    return float(proba_map[object_mask > 0].mean())


# ── Score de confiance combiné ───────────────────────────────────────

def compute_confidence(score_mod: float, score_z: float,
                       score_vois: float, score_t: float) -> float:
    """Combine les 4 scores en un score de confiance final pondéré."""
    w = config.POIDS_SCORE
    total = (
        w["modele"] * score_mod +
        w["zone"] * score_z +
        w["voisinage"] * score_vois +
        w["taille"] * score_t
    )
    return round(total, 3)


def explain_score(score_mod, score_z, score_vois, score_t, confidence) -> str:
    """Explication textuelle du score de confiance."""
    parts = []
    if score_z < 0.5:
        parts.append(f"objet hors de sa zone ({score_z:.0%} dans la bonne zone)")
    if score_vois < 0.5:
        parts.append(f"voisins incohérents ({score_vois:.0%} compatibles)")
    if score_t < 0.5:
        parts.append(f"taille inhabituelle (score {score_t:.0%})")
    if score_mod < 0.3:
        parts.append(f"modèle peu sûr ({score_mod:.0%})")

    if not parts:
        return f"Détection fiable (confiance {confidence:.0%})"
    return f"Confiance {confidence:.0%} — " + " ; ".join(parts)


# ── Analyse complète d'une détection ─────────────────────────────────

def analyze_detection(object_mask: np.ndarray, zone_mask: np.ndarray,
                      full_pred: np.ndarray, proba_map: np.ndarray,
                      class_name: str) -> dict:
    """Analyse complète d'un objet détecté avec tous les scores de voisinage."""
    s_mod = score_modele(proba_map, object_mask)
    s_zone = score_zone(object_mask, zone_mask, class_name)
    s_vois = score_voisinage_spatial(object_mask, full_pred, class_name)
    s_tail = score_taille(object_mask, zone_mask, class_name)
    conf = compute_confidence(s_mod, s_zone, s_vois, s_tail)

    return {
        "class_name": class_name,
        "score_modele": round(s_mod, 3),
        "score_zone": round(s_zone, 3),
        "score_voisinage": round(s_vois, 3),
        "score_taille": round(s_tail, 3),
        "confiance": conf,
        "explication": explain_score(s_mod, s_zone, s_vois, s_tail, conf),
        "accepte": conf >= config.SEUIL_CONFIANCE,
        "n_pixels": int(object_mask.sum()),
    }


# ── Extraction des objets détectés ───────────────────────────────────

def extract_objects(pred_mask: np.ndarray, proba_maps: np.ndarray = None,
                    min_pixels: int = 20) -> list:
    """Extrait les objets individuels (composantes connexes) du masque de prédiction.

    Ignore les zones de scène (fond, eau, terre, ciel = IDs 0-3).
    """
    zone_ids = {0, 1, 2, 3}
    object_ids = [c["id"] for c in config.SEGMENTATION_CLASSES if c["id"] not in zone_ids]

    objects = []
    for cls_id in object_ids:
        class_mask = (pred_mask == cls_id).astype(np.uint8)
        if class_mask.sum() < min_pixels:
            continue

        labeled, n_components = ndimage.label(class_mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            if comp_mask.sum() < min_pixels:
                continue

            ys, xs = np.where(comp_mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            centroid = [float(xs.mean()), float(ys.mean())]
            proba_map = proba_maps[cls_id] if proba_maps is not None else np.ones_like(pred_mask, dtype=np.float32) * 0.5

            objects.append({
                "class_id": cls_id,
                "class_name": _ID_TO_NAME.get(cls_id, f"class_{cls_id}"),
                "mask": comp_mask,
                "bbox": bbox,
                "centroid": centroid,
                "n_pixels": int(comp_mask.sum()),
                "proba_map": proba_map,
            })

    return objects
