# -*- coding: utf-8 -*-
"""Pipeline de segmentation hiérarchique — optimisé GPU.

Le maximum de calculs est fait sur GPU (probas, lissage, argmax).
Seul le post-traitement morphologique reste sur CPU (OpenCV).
"""

import os
import sys
import io
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.unet import UNet
from clustering.neighborhood import (
    extract_objects, analyze_detection, get_zone_for_class
)

_ID_TO_NAME = {c["id"]: c["name"] for c in config.SEGMENTATION_CLASSES}
_NAME_TO_ID = {c["name"]: c["id"] for c in config.SEGMENTATION_CLASSES}
_ZONE_IDS = {name: info["seg_id"] for name, info in config.ZONE_CLASSES.items()}
_ALL_ZONE_IDS = set(_ZONE_IDS.values())

# Cache du modèle pour ne pas le recharger à chaque prédiction
_model_cache = {"model": None, "path": None}


def get_annotated_classes() -> set:
    annotated = set()
    ann_dir = config.ANNOTATIONS_DIR
    if not os.path.isdir(ann_dir):
        return annotated
    for f in os.listdir(ann_dir):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(ann_dir, f), "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for poly in data.get("polygons", []):
            name = poly.get("class_name", "")
            if name:
                annotated.add(name)
    return annotated


def _get_model(device):
    """Charge le modèle une seule fois et le garde en mémoire GPU."""
    model_path = os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Modèle U-Net non entraîné.")

    mtime = os.path.getmtime(model_path)
    if _model_cache["model"] is not None and _model_cache["path"] == mtime:
        return _model_cache["model"]

    model = UNet.load(model_path).to(device).eval()
    _model_cache["model"] = model
    _model_cache["path"] = mtime
    return model


# ── Lissage GPU ──────────────────────────────────────────────────────

def _gaussian_smooth_gpu(probas: torch.Tensor, sigma: float = 2.5) -> torch.Tensor:
    """Flou gaussien sur les probabilités — entièrement sur GPU.

    Utilise un kernel gaussien séparable en 2 passes (horizontal + vertical).
    """
    ksize = int(6 * sigma + 1) | 1
    x = torch.arange(ksize, dtype=torch.float32, device=probas.device) - ksize // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    C = probas.shape[0]
    # Passe horizontale
    k_h = kernel_1d.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
    padded = F.pad(probas.unsqueeze(0), (ksize // 2, ksize // 2, 0, 0), mode="reflect")
    smoothed = F.conv2d(padded, k_h, groups=C)
    # Passe verticale
    k_v = kernel_1d.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
    padded = F.pad(smoothed, (0, 0, ksize // 2, ksize // 2), mode="reflect")
    smoothed = F.conv2d(padded, k_v, groups=C)

    return smoothed.squeeze(0)


# ── Post-traitement TOUT SUR GPU ─────────────────────────────────────
# max_pool2d = dilatation, -max_pool2d(-x) = érosion
# close = dilate → erode, open = erode → dilate

def _gpu_dilate(mask_bool: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """Dilatation morphologique sur GPU via max_pool2d."""
    x = mask_bool.float().unsqueeze(0).unsqueeze(0)
    pad = ksize // 2
    return F.max_pool2d(F.pad(x, (pad, pad, pad, pad), mode="constant", value=0),
                        ksize, stride=1).squeeze() > 0.5


def _gpu_erode(mask_bool: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """Érosion morphologique sur GPU via -max_pool2d(-x)."""
    x = mask_bool.float().unsqueeze(0).unsqueeze(0)
    pad = ksize // 2
    neg = -F.pad(-x, (pad, pad, pad, pad), mode="constant", value=0)
    return F.max_pool2d(neg, ksize, stride=1).squeeze() > 0.5


def _gpu_close(mask_bool: torch.Tensor, ksize: int = 5) -> torch.Tensor:
    return _gpu_erode(_gpu_dilate(mask_bool, ksize), ksize)


def _gpu_fill_unknown(pred: torch.Tensor, max_iter: int = 200) -> torch.Tensor:
    """Remplace les pixels 0 par dilatation itérative de chaque classe — sur GPU."""
    filled = pred.clone()
    unknown = (filled == 0)
    if not unknown.any():
        return filled

    unique_classes = filled.unique()
    unique_classes = unique_classes[unique_classes != 0]

    for _ in range(max_iter):
        if not unknown.any():
            break
        for cid in unique_classes:
            class_mask = (filled == cid)
            expanded = _gpu_dilate(class_mask, 3)
            new_pixels = expanded & unknown
            filled[new_pixels] = cid
            unknown = unknown & ~new_pixels
    return filled


def _gpu_morpho_cleanup(pred: torch.Tensor) -> torch.Tensor:
    """Fermeture morphologique par classe sur GPU."""
    cleaned = pred.clone()
    for cid in pred.unique():
        if cid == 0:
            continue
        binary = (pred == cid)
        closed = _gpu_close(binary, ksize=5)
        cleaned[closed & (cleaned == 0)] = cid
    return cleaned


def _gpu_remove_tiny(pred: torch.Tensor, min_px: int = 50) -> torch.Tensor:
    """Supprime les petites régions sur CPU (connectedComponents) puis retourne sur GPU."""
    device = pred.device
    mask_np = pred.cpu().numpy()
    cleaned = mask_np.copy()
    for cid in np.unique(mask_np):
        if cid == 0:
            continue
        binary = (mask_np == cid).astype(np.uint8)
        n, labeled = cv2.connectedComponents(binary)
        for comp in range(1, n):
            if (labeled == comp).sum() < min_px:
                cleaned[labeled == comp] = 0
    return torch.from_numpy(cleaned).to(device)


def _gpu_build_zone_mask(pred: torch.Tensor) -> torch.Tensor:
    """Construit le masque de zones par dilatation itérative sur GPU."""
    zone_mask = torch.zeros_like(pred)
    for zid in _ALL_ZONE_IDS:
        zone_mask[pred == zid] = zid

    unknown = (zone_mask == 0)
    if not unknown.any():
        return zone_mask

    zone_ids = [zid for zid in _ALL_ZONE_IDS if (zone_mask == zid).any()]
    for _ in range(max(pred.shape)):
        if not unknown.any():
            break
        for zid in zone_ids:
            expanded = _gpu_dilate((zone_mask == zid), 3)
            new_px = expanded & unknown
            zone_mask[new_px] = zid
            unknown = unknown & ~new_px
    return zone_mask


def _hex_to_rgb(hex_c):
    return int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)


def _to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Pipeline principal ───────────────────────────────────────────────

def predict_hierarchical(image_path: str, device: str = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _get_model(device)

    # ── Image → GPU ──────────────────────────────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    seg_size = config.SEG_IMAGE_SIZE
    resized = pil_img.resize((seg_size, seg_size), Image.LANCZOS)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = tensor.unsqueeze(0).to(device)

    # ── Inférence + lissage + argmax — tout sur GPU ──────────────────
    with torch.no_grad(), torch.amp.autocast(device):
        logits = model(x)
        probas = F.softmax(logits, dim=1).squeeze(0)          # (C, H, W) GPU
        probas_smooth = _gaussian_smooth_gpu(probas, sigma=2.5) # GPU
        pred_gpu = probas_smooth.argmax(dim=0)                  # (H, W) GPU

    # ── Post-traitement entièrement sur GPU ────────────────────────────
    pred_gpu = _gpu_morpho_cleanup(pred_gpu)
    pred_gpu = _gpu_remove_tiny(pred_gpu, min_px=50)
    pred_gpu = _gpu_fill_unknown(pred_gpu, max_iter=200)
    zone_gpu = _gpu_build_zone_mask(pred_gpu)

    # Résultat final → CPU (une seule copie GPU→CPU, tout le reste était sur GPU)
    pred_mask = pred_gpu.byte().cpu().numpy()
    zone_mask_np = zone_gpu.byte().cpu().numpy()
    probas_np = probas_smooth.float().cpu().numpy().astype(np.float64)

    # ── Resize au ratio original ─────────────────────────────────────
    display_max = 700
    scale = min(display_max / orig_w, display_max / orig_h)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

    pred_disp = cv2.resize(pred_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    zone_disp = cv2.resize(zone_mask_np, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    probas_disp = np.stack([
        cv2.resize(probas_np[c], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR).clip(0, 1)
        for c in range(probas_np.shape[0])
    ])
    orig_display = np.array(pil_img.resize((disp_w, disp_h), Image.LANCZOS))

    # ── Stats ────────────────────────────────────────────────────────
    total_px = pred_disp.size
    zone_stats = {}
    accounted = 0
    for zone_name, zone_info in config.ZONE_CLASSES.items():
        zid = zone_info["seg_id"]
        count = int((pred_disp == zid).sum())
        accounted += count
        zone_stats[zone_name] = {
            "pixels": count, "pourcentage": round(count / total_px * 100, 1),
            "color": zone_info["color"],
        }
    obj_px = int((pred_disp > 3).sum())
    accounted += obj_px
    non_reconnu = total_px - accounted
    if non_reconnu > 0:
        zone_stats["non_reconnu"] = {
            "pixels": non_reconnu, "pourcentage": round(non_reconnu / total_px * 100, 1),
            "color": "#333333",
        }

    # ── Classes annotées ─────────────────────────────────────────────
    annotated = get_annotated_classes()
    zone_class_names = {"eau", "terre", "ciel", "non_reconnu"}
    annotated_objects = annotated - zone_class_names

    filtered_pred = pred_disp.copy()
    for cls_info in config.SEGMENTATION_CLASSES:
        cid, cname = cls_info["id"], cls_info["name"]
        if cid > 3 and cname not in annotated_objects:
            filtered_pred[filtered_pred == cid] = 0

    # ── Objets + voisinage ───────────────────────────────────────────
    raw_objects = extract_objects(filtered_pred, proba_maps=probas_disp, min_pixels=15)
    detections = []
    for obj in raw_objects:
        expected_zone = get_zone_for_class(obj["class_name"])
        if obj["class_name"] not in config.OBJETS_PAR_ZONE.get(expected_zone, []):
            continue
        analysis = analyze_detection(
            object_mask=obj["mask"], zone_mask=zone_disp,
            full_pred=pred_disp, proba_map=obj["proba_map"],
            class_name=obj["class_name"],
        )
        analysis["bbox"] = obj["bbox"]
        analysis["centroid"] = obj["centroid"]
        detections.append(analysis)

    accepted = [d for d in detections if d["accepte"]]
    rejected = [d for d in detections if not d["accepte"]]

    for det in detections:
        cname = det["class_name"]
        if not det["accepte"]:
            continue
        if cname not in zone_stats:
            zone_stats[cname] = {
                "pixels": det.get("n_pixels", 0),
                "pourcentage": round(det.get("n_pixels", 0) / total_px * 100, 1),
                "color": next((c["color"] for c in config.SEGMENTATION_CLASSES if c["name"] == cname), "#808080"),
            }
        elif cname not in config.ZONE_CLASSES and cname != "non_reconnu":
            zone_stats[cname]["pixels"] += det.get("n_pixels", 0)
            zone_stats[cname]["pourcentage"] = round(zone_stats[cname]["pixels"] / total_px * 100, 1)

    # ── Images de sortie ─────────────────────────────────────────────
    zone_colored = np.full((disp_h, disp_w, 3), 51, dtype=np.uint8)
    for zone_name, zone_info in config.ZONE_CLASSES.items():
        r, g, b = _hex_to_rgb(zone_info["color"])
        zone_colored[pred_disp == zone_info["seg_id"]] = [r, g, b]

    full_colored = zone_colored.copy()
    for det in accepted:
        cls_id = _NAME_TO_ID.get(det["class_name"])
        cls_info = next((c for c in config.SEGMENTATION_CLASSES if c["id"] == cls_id), None)
        if cls_info:
            r, g, b = _hex_to_rgb(cls_info["color"])
            full_colored[filtered_pred == cls_id] = [r, g, b]

    overlay = (orig_display.astype(np.float32) * 0.45 +
               full_colored.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)

    for det in accepted:
        x1, y1, x2, y2 = det["bbox"]
        cls_info = next((c for c in config.SEGMENTATION_CLASSES if c["name"] == det["class_name"]), None)
        color_bgr = (255, 255, 255)
        if cls_info:
            r, g, b = _hex_to_rgb(cls_info["color"])
            color_bgr = (b, g, r)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 2)
        lbl = f"{det['class_name'].replace('_', ' ')} {det['confiance']:.0%}"
        cv2.putText(overlay, lbl, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    clean_det = [{k: v for k, v in d.items() if k != "mask"} for d in accepted]
    clean_rej = [{k: v for k, v in d.items() if k != "mask"} for d in rejected]

    return {
        "original_b64": _to_b64(orig_display),
        "zones_b64": _to_b64(zone_colored),
        "full_b64": _to_b64(full_colored),
        "overlay_b64": _to_b64(overlay),
        "zone_stats": zone_stats,
        "detections": clean_det, "rejected": clean_rej,
        "n_accepted": len(accepted), "n_rejected": len(rejected),
        "classes_annotees": sorted(list(annotated_objects)),
        "classes_non_annotees": sorted([
            c["name"] for c in config.SEGMENTATION_CLASSES
            if c["id"] > 3 and c["name"] not in annotated_objects
        ]),
        "image_size": f"{disp_w}x{disp_h}",
    }
