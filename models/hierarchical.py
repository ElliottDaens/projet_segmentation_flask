# -*- coding: utf-8 -*-
"""Pipeline de segmentation hiérarchique : scène (mer/terre/ciel) → objets par zone."""

import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.unet import UNet, mask_to_colored
from clustering.neighborhood import (
    extract_objects, analyze_detection, get_zone_for_class
)

_ID_TO_NAME = {c["id"]: c["name"] for c in config.SEGMENTATION_CLASSES}
_NAME_TO_ID = {c["name"]: c["id"] for c in config.SEGMENTATION_CLASSES}
_ZONE_IDS = {name: info["seg_id"] for name, info in config.ZONE_CLASSES.items()}


def get_annotated_classes() -> set:
    """Retourne l'ensemble des classes qui ont été annotées au moins une fois."""
    annotated = set()
    ann_dir = config.ANNOTATIONS_DIR
    if not os.path.isdir(ann_dir):
        return annotated
    for f in os.listdir(ann_dir):
        if not f.endswith(".json"):
            continue
        path = os.path.join(ann_dir, f)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for poly in data.get("polygons", []):
            name = poly.get("class_name", "")
            if name:
                annotated.add(name)
    return annotated


def predict_hierarchical(image_path: str, model: UNet = None,
                         device: str = None) -> dict:
    """Pipeline complet de segmentation hiérarchique.

    Étape 1 : prédiction U-Net → masque complet
    Étape 2 : extraction des 3 zones (mer/terre/ciel)
    Étape 3 : détection d'objets par zone (seulement les classes annotées)
    Étape 4 : analyse de voisinage + scores de confiance
    Étape 5 : filtrage par seuil de confiance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model_path = os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError("Modèle U-Net non entraîné.")
        model = UNet.load(model_path)

    model = model.to(device).eval()

    # ── Charger et préparer l'image ──────────────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    resized = pil_img.resize((config.SEG_IMAGE_SIZE, config.SEG_IMAGE_SIZE), Image.LANCZOS)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    x = tensor.unsqueeze(0).to(device)

    # ── Étape 1 : prédiction brute ──────────────────────────────────
    with torch.no_grad():
        logits = model(x)
        probas = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # ── Étape 2 : extraire les zones de scène ───────────────────────
    zone_mask = np.zeros_like(pred_mask)
    for zone_name, zone_info in config.ZONE_CLASSES.items():
        zid = zone_info["seg_id"]
        zone_mask[pred_mask == zid] = zid

    # Les pixels d'objets héritent de la zone dans laquelle ils sont
    # (par propagation du voisin le plus proche dans la zone mask)
    object_pixel_mask = (zone_mask == 0) & (pred_mask > 0)
    if object_pixel_mask.any():
        zone_only = zone_mask.copy()
        zone_only[zone_only == 0] = 0
        # Propager les zones vers les pixels non-zone via distance transform
        from scipy.ndimage import distance_transform_edt
        for zid in _ZONE_IDS.values():
            z = (zone_mask == zid).astype(np.float32)
            if z.sum() > 0:
                dist = distance_transform_edt(1 - z)
                # On garde la zone la plus proche pour chaque pixel objet
                # (simplifié : on itère et prend le min)
                pass
        # Approche simplifiée : les objets dans la mer = mer, etc.
        # On regarde pour chaque composante d'objet dans quelle zone est son centroïde

    zone_stats = {}
    total_px = pred_mask.size
    for zone_name, zone_info in config.ZONE_CLASSES.items():
        zid = zone_info["seg_id"]
        count = int((pred_mask == zid).sum())
        zone_stats[zone_name] = {
            "pixels": count,
            "pourcentage": round(count / total_px * 100, 1),
            "color": zone_info["color"],
        }

    # ── Étape 3 : classes annotées uniquement ───────────────────────
    annotated = get_annotated_classes()
    zone_class_names = {"eau", "terre", "ciel", "fond"}
    annotated_objects = annotated - zone_class_names

    # Masquer les classes non annotées dans la prédiction
    filtered_pred = pred_mask.copy()
    for cls_info in config.SEGMENTATION_CLASSES:
        cid = cls_info["id"]
        cname = cls_info["name"]
        if cid > 3 and cname not in annotated_objects:
            filtered_pred[filtered_pred == cid] = 0

    # ── Étape 4 : extraire et analyser les objets ───────────────────
    raw_objects = extract_objects(filtered_pred, proba_maps=probas, min_pixels=15)

    detections = []
    for obj in raw_objects:
        # Vérifier que l'objet est dans une zone autorisée
        expected_zone = get_zone_for_class(obj["class_name"])
        allowed_objects = config.OBJETS_PAR_ZONE.get(expected_zone, [])

        if obj["class_name"] not in allowed_objects:
            continue

        analysis = analyze_detection(
            object_mask=obj["mask"],
            zone_mask=pred_mask,
            full_pred=pred_mask,
            proba_map=obj["proba_map"],
            class_name=obj["class_name"],
        )
        analysis["bbox"] = obj["bbox"]
        analysis["centroid"] = obj["centroid"]
        detections.append(analysis)

    # ── Étape 5 : filtrage par confiance ─────────────────────────────
    accepted = [d for d in detections if d["accepte"]]
    rejected = [d for d in detections if not d["accepte"]]

    # ── Construire les images de sortie ──────────────────────────────
    # Masque des zones uniquement (niveau 1)
    zone_colored = np.zeros((config.SEG_IMAGE_SIZE, config.SEG_IMAGE_SIZE, 3), dtype=np.uint8)
    for zone_name, zone_info in config.ZONE_CLASSES.items():
        zid = zone_info["seg_id"]
        hex_c = zone_info["color"]
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        zone_colored[pred_mask == zid] = [r, g, b]

    # Masque complet (zones + objets acceptés)
    full_colored = zone_colored.copy()
    for det in accepted:
        cls_id = _NAME_TO_ID.get(det["class_name"])
        if cls_id is None:
            continue
        obj_mask = (filtered_pred == cls_id)
        cls_info = next((c for c in config.SEGMENTATION_CLASSES if c["id"] == cls_id), None)
        if cls_info:
            hex_c = cls_info["color"]
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            full_colored[obj_mask] = [r, g, b]

    # Overlay avec bounding boxes et scores
    orig_small = np.array(resized)
    overlay = (orig_small.astype(np.float32) * 0.45 +
               full_colored.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)

    for det in accepted:
        x1, y1, x2, y2 = det["bbox"]
        cls_info = next((c for c in config.SEGMENTATION_CLASSES
                         if c["name"] == det["class_name"]), None)
        if cls_info:
            hex_c = cls_info["color"]
            color_bgr = (int(hex_c[5:7], 16), int(hex_c[3:5], 16), int(hex_c[1:3], 16))
        else:
            color_bgr = (255, 255, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{det['class_name']} {det['confiance']:.0%}"
        cv2.putText(overlay, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── Encoder en base64 ────────────────────────────────────────────
    import io, base64

    def to_b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Nettoyer les détections pour JSON (retirer les numpy arrays)
    clean_detections = []
    for d in accepted:
        clean_detections.append({k: v for k, v in d.items() if k != "mask"})
    clean_rejected = []
    for d in rejected:
        clean_rejected.append({k: v for k, v in d.items() if k != "mask"})

    return {
        "original_b64": to_b64(orig_small),
        "zones_b64": to_b64(zone_colored),
        "full_b64": to_b64(full_colored),
        "overlay_b64": to_b64(overlay),
        "zone_stats": zone_stats,
        "detections": clean_detections,
        "rejected": clean_rejected,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "classes_annotees": sorted(list(annotated_objects)),
        "classes_non_annotees": sorted([
            c["name"] for c in config.SEGMENTATION_CLASSES
            if c["id"] > 3 and c["name"] not in annotated_objects
        ]),
    }
