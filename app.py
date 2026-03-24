# -*- coding: utf-8 -*-
"""Application Flask — segmentation, clustering et étiquetage d'images d'embarcations."""

import io
import os
import json
import datetime
import numpy as np
from PIL import Image
from flask import (Flask, render_template, request, jsonify,
                   send_file, abort, redirect, url_for)

import config
from preprocessing.transforms import list_images
from preprocessing.normalization import pil_normalize

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ── État en mémoire ──────────────────────────────────────────────────
_state = {
    "embeddings": None,       # np.ndarray (N, D)
    "embeddings_2d": None,    # np.ndarray (N, 2) — réduction UMAP/PCA
    "filenames": None,        # list[str]
    "cluster_labels": None,   # list[int]
    "cluster_report": None,   # dict
    "metrics": None,          # dict
    "method": None,           # "cnn" | "autoencoder"
    "optimal_k": None,        # dict
}


# ── Utilitaires labels ──────────────────────────────────────────────

def load_labels() -> dict:
    if os.path.isfile(config.LABELS_PATH):
        with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict):
    os.makedirs(os.path.dirname(config.LABELS_PATH), exist_ok=True)
    with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def load_labels_history() -> list:
    if os.path.isfile(config.LABELS_HISTORY_PATH):
        with open(config.LABELS_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_labels_history(history: list):
    os.makedirs(os.path.dirname(config.LABELS_HISTORY_PATH), exist_ok=True)
    with open(config.LABELS_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def add_history_entry(filename: str, old_label: str, new_label: str):
    history = load_labels_history()
    history.append({
        "filename": filename,
        "old_label": old_label,
        "new_label": new_label,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    save_labels_history(history)


# ── ROUTE : Page d'accueil ──────────────────────────────────────────

@app.route("/")
def index():
    images = list_images()
    from preprocessing.segmentation_dataset import count_annotated_images
    n_annotated = count_annotated_images()
    model_path = os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
    model_ready = os.path.isfile(model_path)
    return render_template("index.html",
                           n_images=len(images),
                           n_annotated=n_annotated,
                           model_ready=model_ready)


# ── ROUTE : Clustering ──────────────────────────────────────────────

@app.route("/clustering")
def clustering_page():
    images = list_images()
    return render_template("clustering.html",
                           images=images,
                           state=_state,
                           n_clusters_default=config.N_CLUSTERS)


@app.route("/api/compute-embeddings", methods=["POST"])
def api_compute_embeddings():
    """Calcule les embeddings (CNN ou auto-encodeur)."""
    method = request.json.get("method", "cnn")
    try:
        if method == "cnn":
            from preprocessing.augmentation import get_eval_transforms
            from preprocessing.transforms import BoatDataset, get_dataloader
            from models.feature_extractor import extract_features

            dataset = BoatDataset(transform=get_eval_transforms())
            loader = get_dataloader(dataset, shuffle=False)
            embeddings, filenames = extract_features(loader)
        else:
            from preprocessing.augmentation import get_autoencoder_transforms
            from preprocessing.transforms import BoatDataset, get_dataloader
            from models.autoencoder import ConvAutoencoder, get_latent_vectors

            ae_path = os.path.join(config.MODELS_SAVE_DIR, "autoencoder.pth")
            if not os.path.isfile(ae_path):
                return jsonify({"error": "Auto-encodeur non entraîné. Lancez d'abord scripts/train_autoencoder.py"}), 400

            model = ConvAutoencoder.load(ae_path)
            dataset = BoatDataset(transform=get_autoencoder_transforms())
            loader = get_dataloader(dataset, shuffle=False)
            embeddings, filenames = get_latent_vectors(model, loader)

        _state["embeddings"] = embeddings
        _state["filenames"] = filenames
        _state["method"] = method

        from clustering.pipeline import reduce_dimensions
        _state["embeddings_2d"] = reduce_dimensions(embeddings, method="umap")

        np.save(os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy"), embeddings)
        np.save(os.path.join(config.EMBEDDINGS_DIR, "embeddings_2d.npy"), _state["embeddings_2d"])
        with open(os.path.join(config.EMBEDDINGS_DIR, "filenames.json"), "w") as f:
            json.dump(filenames, f)

        return jsonify({
            "success": True,
            "n_images": len(filenames),
            "embedding_dim": embeddings.shape[1],
            "method": method,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run-clustering", methods=["POST"])
def api_run_clustering():
    """Exécute le clustering sur les embeddings calculés."""
    if _state["embeddings"] is None:
        return jsonify({"error": "Embeddings non calculés. Lancez d'abord le calcul."}), 400

    method = request.json.get("method", "kmeans")
    n_clusters = request.json.get("n_clusters", config.N_CLUSTERS)

    try:
        from clustering.pipeline import run_kmeans, run_dbscan, find_optimal_k
        from clustering.evaluation import compute_metrics, generate_cluster_report

        if method == "kmeans":
            result = run_kmeans(_state["embeddings"], n_clusters=n_clusters)
        elif method == "dbscan":
            eps = request.json.get("eps", config.DBSCAN_EPS)
            min_samples = request.json.get("min_samples", config.DBSCAN_MIN_SAMPLES)
            result = run_dbscan(_state["embeddings"], eps=eps, min_samples=min_samples)
        else:
            return jsonify({"error": f"Méthode inconnue : {method}"}), 400

        _state["cluster_labels"] = result["labels"]

        metrics = compute_metrics(_state["embeddings"], result["labels"])
        _state["metrics"] = metrics

        labels_dict = load_labels()
        report = generate_cluster_report(
            _state["embeddings_2d"], result["labels"],
            _state["filenames"], manual_labels=labels_dict
        )
        _state["cluster_report"] = report

        optimal = find_optimal_k(_state["embeddings"])
        _state["optimal_k"] = optimal

        return jsonify({
            "success": True,
            "report": report,
            "metrics": metrics,
            "optimal_k": optimal,
            "result": result,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/semi-supervised-clustering", methods=["POST"])
def api_semi_supervised():
    """Clustering semi-supervisé guidé par les labels manuels."""
    if _state["embeddings"] is None:
        return jsonify({"error": "Embeddings non calculés."}), 400

    try:
        from clustering.pipeline import semi_supervised_clustering
        from clustering.evaluation import compute_metrics, generate_cluster_report

        labels_dict = load_labels()
        n_clusters = request.json.get("n_clusters", config.N_CLUSTERS)

        result = semi_supervised_clustering(
            _state["embeddings"], labels_dict,
            _state["filenames"], n_clusters=n_clusters
        )
        _state["cluster_labels"] = result["labels"]

        metrics = compute_metrics(_state["embeddings"], result["labels"])
        _state["metrics"] = metrics

        report = generate_cluster_report(
            _state["embeddings_2d"], result["labels"],
            _state["filenames"], manual_labels=labels_dict
        )
        _state["cluster_report"] = report

        return jsonify({
            "success": True,
            "report": report,
            "metrics": metrics,
            "result": result,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ROUTE : Étiquetage ──────────────────────────────────────────────

@app.route("/labeling")
def labeling_page():
    images = list_images()
    labels = load_labels()
    cluster_filter = request.args.get("cluster", None)

    display_images = []
    for fname in images:
        info = {
            "filename": fname,
            "label": labels.get(fname, {}).get("label", "non_etiqueté"),
        }
        if _state["filenames"] and fname in _state["filenames"]:
            idx = _state["filenames"].index(fname)
            if _state["cluster_labels"]:
                info["cluster"] = _state["cluster_labels"][idx]
        display_images.append(info)

    if cluster_filter is not None:
        cluster_filter = int(cluster_filter)
        display_images = [im for im in display_images if im.get("cluster") == cluster_filter]

    return render_template("labeling.html",
                           images=display_images,
                           default_labels=config.DEFAULT_LABELS,
                           all_labels=labels)


@app.route("/api/label", methods=["POST"])
def api_save_label():
    """Sauvegarde un label pour une image."""
    data = request.json
    filename = data.get("filename")
    label = data.get("label")
    if not filename or not label:
        return jsonify({"error": "filename et label requis"}), 400

    labels = load_labels()
    old_label = labels.get(filename, {}).get("label", "non_etiqueté")
    labels[filename] = {
        "label": label,
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "manual",
    }
    save_labels(labels)

    if old_label != label:
        add_history_entry(filename, old_label, label)

    return jsonify({"success": True, "filename": filename, "label": label})


@app.route("/api/label-bulk", methods=["POST"])
def api_label_bulk():
    """Sauvegarde des labels en masse."""
    data = request.json
    entries = data.get("entries", [])
    if not entries:
        return jsonify({"error": "Aucune entrée fournie"}), 400

    labels = load_labels()
    count = 0
    for entry in entries:
        fname = entry.get("filename")
        lbl = entry.get("label")
        if fname and lbl:
            old = labels.get(fname, {}).get("label", "non_etiqueté")
            labels[fname] = {
                "label": lbl,
                "timestamp": datetime.datetime.now().isoformat(),
                "source": "bulk",
            }
            if old != lbl:
                add_history_entry(fname, old, lbl)
            count += 1

    save_labels(labels)
    return jsonify({"success": True, "count": count})


@app.route("/api/suggest-labels", methods=["POST"])
def api_suggest_labels():
    """Suggère un label pour les images proches dans l'espace latent."""
    if _state["embeddings"] is None or _state["filenames"] is None:
        return jsonify({"error": "Embeddings non calculés."}), 400

    filename = request.json.get("filename")
    n_suggestions = request.json.get("n", 5)

    if filename not in _state["filenames"]:
        return jsonify({"error": "Image non trouvée dans les embeddings."}), 404

    idx = _state["filenames"].index(filename)
    target = _state["embeddings"][idx]
    distances = np.linalg.norm(_state["embeddings"] - target, axis=1)
    nearest_indices = np.argsort(distances)[1:n_suggestions + 1]

    labels = load_labels()
    suggestions = []
    for ni in nearest_indices:
        fname = _state["filenames"][ni]
        suggestions.append({
            "filename": fname,
            "distance": float(distances[ni]),
            "label": labels.get(fname, {}).get("label", "non_etiqueté"),
        })

    return jsonify({"suggestions": suggestions, "source": filename})


# ── ROUTE : Corrections ─────────────────────────────────────────────

@app.route("/corrections")
def corrections_page():
    labels = load_labels()
    history = load_labels_history()
    images = list_images()

    by_label = {}
    for fname in images:
        lbl = labels.get(fname, {}).get("label", "non_etiqueté")
        by_label.setdefault(lbl, []).append(fname)

    return render_template("corrections.html",
                           labels=labels,
                           by_label=by_label,
                           history=history[-50:],
                           default_labels=config.DEFAULT_LABELS)


# ── ROUTE : Régions ─────────────────────────────────────────────────

@app.route("/regions")
def regions_page():
    images = list_images()
    return render_template("regions.html", images=images)


@app.route("/api/analyze-regions", methods=["POST"])
def api_analyze_regions():
    """Analyse par régions (superpixels) d'une image."""
    filename = request.json.get("filename")
    n_segments = request.json.get("n_segments", config.SLIC_N_SEGMENTS)
    n_clusters = request.json.get("n_clusters", 5)

    path = os.path.join(config.IMAGES_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "Image non trouvée"}), 404

    try:
        from clustering.region_analysis import analyze_image_regions
        result = analyze_image_regions(path, n_segments=n_segments,
                                       n_region_clusters=n_clusters)
        result.pop("region_features", None)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ROUTE : Annotation par zones ─────────────────────────────────────

@app.route("/annotation")
def annotation_page():
    images = list_images()
    selected = request.args.get("image", images[0] if images else "")
    from preprocessing.segmentation_dataset import count_annotated_images
    n_annotated = count_annotated_images()
    return render_template("annotation.html",
                           images=images,
                           selected_image=selected,
                           classes=config.SEGMENTATION_CLASSES,
                           n_annotated=n_annotated,
                           n_total=len(images))


@app.route("/api/save-annotation", methods=["POST"])
def api_save_annotation():
    """Sauvegarde les polygones d'annotation + génère le masque PNG pour l'entraînement."""
    data = request.json
    filename = data.get("filename")
    poly_data = data.get("polygons", [])
    img_w = data.get("image_width", 0)
    img_h = data.get("image_height", 0)

    if not filename or not poly_data:
        return jsonify({"error": "filename et polygons requis"}), 400

    try:
        # 1. Sauvegarder les polygones en JSON (éditable)
        ann_name = os.path.splitext(filename)[0] + ".json"
        ann_path = os.path.join(config.ANNOTATIONS_DIR, ann_name)
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": filename,
                "image_width": img_w,
                "image_height": img_h,
                "polygons": poly_data,
            }, f, indent=2, ensure_ascii=False)

        # 2. Rasteriser les polygones en masque PNG (pixel = indice de classe)
        import cv2
        mask = np.full((img_h, img_w), config.IGNORE_INDEX, dtype=np.uint8)
        for poly in poly_data:
            pts = np.array(poly["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], int(poly["class_id"]))

        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(config.MASKS_DIR, mask_name)
        from PIL import Image as PILImage
        PILImage.fromarray(mask).save(mask_path)

        return jsonify({
            "success": True,
            "n_polygons": len(poly_data),
            "annotation_path": ann_path,
            "mask_path": mask_path,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-annotation/<filename>")
def api_load_annotation(filename):
    """Charge les polygones existants pour une image."""
    ann_name = os.path.splitext(filename)[0] + ".json"
    ann_path = os.path.join(config.ANNOTATIONS_DIR, ann_name)

    if not os.path.isfile(ann_path):
        return jsonify({"error": "Pas d'annotation existante"}), 404

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/train-segmentation", methods=["POST"])
def api_train_segmentation():
    """Lance l'entraînement du U-Net sur les masques annotés."""
    from preprocessing.segmentation_dataset import count_annotated_images, get_seg_dataloader
    from models.unet import train_unet

    n = count_annotated_images()
    if n < 3:
        return jsonify({"error": f"Seulement {n} masques annotés. Il en faut au moins 3."}), 400

    try:
        epochs = request.json.get("epochs", config.UNET_EPOCHS)
        loader = get_seg_dataloader(augment=True, shuffle=True)
        model = train_unet(loader, epochs=epochs)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
        with torch.no_grad():
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                total_loss += criterion(logits, masks).item()
        final_loss = total_loss / max(len(loader), 1)

        return jsonify({"success": True, "n_images": n, "final_loss": final_loss})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ROUTE : Prédiction automatique ──────────────────────────────────

@app.route("/prediction")
def prediction_page():
    images = list_images()
    model_path = os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
    model_ready = os.path.isfile(model_path)
    return render_template("prediction.html",
                           images=images,
                           classes=config.SEGMENTATION_CLASSES,
                           model_ready=model_ready)


@app.route("/api/predict-segmentation", methods=["POST"])
def api_predict_segmentation():
    """Segmente automatiquement une image avec le U-Net entraîné."""
    import base64
    from models.unet import UNet, predict_image, mask_to_colored
    import torchvision.transforms.functional as TF
    import torch

    model_path = os.path.join(config.MODELS_SAVE_DIR, "unet.pth")
    if not os.path.isfile(model_path):
        return jsonify({"error": "Modèle non entraîné."}), 400

    filename = request.json.get("filename")
    path = os.path.join(config.IMAGES_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "Image non trouvée"}), 404

    try:
        model = UNet.load(model_path)

        pil_img = Image.open(path).convert("RGB")
        orig_w, orig_h = pil_img.size
        resized = pil_img.resize((config.SEG_IMAGE_SIZE, config.SEG_IMAGE_SIZE), Image.LANCZOS)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        mask = predict_image(model, tensor)
        colored_mask = mask_to_colored(mask)

        # Créer l'overlay (image + masque semi-transparent)
        mask_pil = Image.fromarray(colored_mask).resize((orig_w, orig_h), Image.NEAREST)
        orig_arr = np.array(pil_img).astype(np.float32)
        mask_arr = np.array(mask_pil).astype(np.float32)
        overlay_arr = (orig_arr * 0.5 + mask_arr * 0.5).clip(0, 255).astype(np.uint8)

        def to_b64(arr):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # Proportions de chaque classe
        total_pixels = mask.shape[0] * mask.shape[1]
        class_pcts = {}
        for cls_info in config.SEGMENTATION_CLASSES:
            count = int((mask == cls_info["id"]).sum())
            pct = count / total_pixels * 100
            class_pcts[cls_info["name"]] = round(pct, 1)

        # Redimensionner original pour affichage cohérent
        display_size = 400
        scale = min(display_size / orig_w, display_size / orig_h, 1)
        dw, dh = int(orig_w * scale), int(orig_h * scale)
        orig_display = np.array(pil_img.resize((dw, dh), Image.LANCZOS))
        mask_display = np.array(mask_pil.resize((dw, dh), Image.NEAREST))
        overlay_display = (orig_display.astype(np.float32) * 0.5 +
                          mask_display.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)

        return jsonify({
            "success": True,
            "original_b64": to_b64(orig_display),
            "mask_b64": to_b64(mask_display),
            "overlay_b64": to_b64(overlay_display),
            "class_percentages": class_pcts,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ROUTE : Documentation ───────────────────────────────────────────

@app.route("/documentation")
def documentation_page():
    import markdown
    docs = {}
    doc_files = {
        "utilisation": os.path.join(config.BASE_DIR, "reports", "doc_utilisation.md"),
        "verrous": os.path.join(config.BASE_DIR, "reports", "doc_verrous.md"),
    }
    for key, path in doc_files.items():
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                docs[key] = markdown.markdown(f.read(), extensions=["tables", "fenced_code"])
        else:
            docs[key] = f"<p><em>Documentation '{key}' non encore générée.</em></p>"

    return render_template("documentation.html", docs=docs)


# ── ROUTE : Service des images ──────────────────────────────────────

@app.route("/api/image/<filename>")
def serve_image(filename):
    """Sert une image du dataset."""
    path = os.path.join(config.IMAGES_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/api/image-normalized/<filename>")
def serve_image_normalized(filename):
    """Sert une image normalisée (CLAHE) du dataset."""
    path = os.path.join(config.IMAGES_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    method = request.args.get("method", "clahe")
    img = Image.open(path).convert("RGB")
    img_norm = pil_normalize(img, method=method)
    buf = io.BytesIO()
    img_norm.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


# ── ROUTE : API état ────────────────────────────────────────────────

@app.route("/api/state")
def api_state():
    """Retourne l'état courant du pipeline."""
    return jsonify({
        "has_embeddings": _state["embeddings"] is not None,
        "has_clusters": _state["cluster_labels"] is not None,
        "n_images": len(_state["filenames"]) if _state["filenames"] else 0,
        "method": _state["method"],
        "n_clusters": _state["cluster_report"]["n_clusters"] if _state["cluster_report"] else 0,
        "metrics": _state["metrics"],
    })


@app.route("/api/labels")
def api_get_labels():
    return jsonify(load_labels())


@app.route("/api/images")
def api_list_images():
    return jsonify(list_images())


# ── Chargement des embeddings sauvegardés au démarrage ──────────────

def _load_saved_state():
    emb_path = os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy")
    emb2d_path = os.path.join(config.EMBEDDINGS_DIR, "embeddings_2d.npy")
    fn_path = os.path.join(config.EMBEDDINGS_DIR, "filenames.json")

    if os.path.isfile(emb_path) and os.path.isfile(fn_path):
        _state["embeddings"] = np.load(emb_path)
        with open(fn_path, "r") as f:
            _state["filenames"] = json.load(f)
        _state["method"] = "cnn"
        if os.path.isfile(emb2d_path):
            _state["embeddings_2d"] = np.load(emb2d_path)


_load_saved_state()


# ── Lancement ────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        debug=config.FLASK_DEBUG,
        port=config.FLASK_PORT,
        exclude_patterns=["*.venv*"],
        reloader_type="stat",
    )
