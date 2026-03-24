# -*- coding: utf-8 -*-
"""Configuration centralisée du projet de segmentation d'embarcations."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Chemins ──────────────────────────────────────────────────────────
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
DATA_DIR = os.path.join(BASE_DIR, "data")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
LABELS_HISTORY_PATH = os.path.join(DATA_DIR, "labels_history.json")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODELS_SAVE_DIR = os.path.join(DATA_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# ── Images ───────────────────────────────────────────────────────────
IMAGE_SIZE = 224          # côté du carré de redimensionnement (px)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")

# ── Augmentation ─────────────────────────────────────────────────────
AUGMENTATION_FACTOR = 5   # nb d'images augmentées générées par original
ROTATION_RANGE = 30       # degrés
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.7, 1.3)
HUE_RANGE = 0.05
NOISE_STD = 0.02

# ── Auto-encodeur ───────────────────────────────────────────────────
LATENT_DIM = 128
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 80
AE_BATCH_SIZE = 8
AE_WEIGHT_DECAY = 1e-5

# ── Feature extractor (CNN pré-entraîné) ────────────────────────────
BACKBONE = "resnet18"     # resnet18 | resnet50 | efficientnet_b0
FEATURE_DIM = 512         # dimension du vecteur de features (ResNet-18)

# ── Clustering ───────────────────────────────────────────────────────
N_CLUSTERS = 4            # K par défaut pour K-Means
K_RANGE = range(2, 8)     # plage pour la méthode du coude
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 2
UMAP_N_NEIGHBORS = 5
UMAP_MIN_DIST = 0.3
UMAP_N_COMPONENTS = 2
PCA_N_COMPONENTS = 50

# ── Classifieur supervisé ───────────────────────────────────────────
CLASSIFIER_HIDDEN = [256, 128]
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 50

# ── Régions (superpixels) ───────────────────────────────────────────
SLIC_N_SEGMENTS = 100
SLIC_COMPACTNESS = 10.0
CANNY_LOW = 50
CANNY_HIGH = 150

# ── Labels prédéfinis ───────────────────────────────────────────────
DEFAULT_LABELS = [
    "bateau_moteur",
    "voilier",
    "paddle",
    "kayak",
    "gonflable",
    "jet_ski",
    "barque",
    "autre",
]

# ── Segmentation sémantique (annotation par zones) ─────────────────
MASKS_DIR = os.path.join(DATA_DIR, "masks")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
SEGMENTATION_CLASSES = [
    {"id": 0, "name": "non_reconnu",    "color": "#333333"},
    {"id": 1, "name": "eau",            "color": "#0077FF"},
    {"id": 2, "name": "terre",          "color": "#8B5A2B"},
    {"id": 3, "name": "ciel",           "color": "#87CEEB"},
    {"id": 4, "name": "bateau_moteur",  "color": "#FF0000"},
    {"id": 5, "name": "voilier",        "color": "#FFA500"},
    {"id": 6, "name": "paddle",         "color": "#00FF00"},
    {"id": 7, "name": "kayak",          "color": "#FFFF00"},
    {"id": 8, "name": "gonflable",      "color": "#FF00FF"},
    {"id": 9, "name": "autre_objet",    "color": "#808080"},
]
NUM_SEG_CLASSES = len(SEGMENTATION_CLASSES)
SEG_IMAGE_SIZE = 384
UNET_LEARNING_RATE = 3e-4
UNET_EPOCHS = 150
UNET_BATCH_SIZE = 4
UNET_WEIGHT_DECAY = 1e-4
UNET_UNFREEZE_EPOCH = 30
UNET_DATASET_REPEAT = 8
IGNORE_INDEX = 255

# ── Segmentation hiérarchique ────────────────────────────────────────
# Niveau 1 : 3 zones de scène (les IDs correspondent aux SEGMENTATION_CLASSES)
ZONE_CLASSES = {
    "mer":   {"seg_id": 1, "color": "#0077FF", "label": "Mer / Eau"},
    "terre": {"seg_id": 2, "color": "#8B5A2B", "label": "Terre / Sol"},
    "ciel":  {"seg_id": 3, "color": "#87CEEB", "label": "Ciel"},
}

# Niveau 2 : objets autorisés par zone (seg_id → liste de noms de classes objets)
OBJETS_PAR_ZONE = {
    "mer":   ["bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"],
    "terre": ["autre_objet"],
    "ciel":  ["autre_objet"],
}

# Compatibilité sémantique entre objets voisins (paires autorisées)
VOISINS_COMPATIBLES = {
    "bateau_moteur": {"bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"},
    "voilier":       {"bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"},
    "paddle":        {"bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"},
    "kayak":         {"bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"},
    "gonflable":     {"bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"},
    "autre_objet":   {"bateau_moteur", "voilier", "paddle", "kayak", "gonflable", "autre_objet"},
}

# Taille relative attendue des objets (fraction de la zone parente)
TAILLE_ATTENDUE = {
    "bateau_moteur": (0.005, 0.40),
    "voilier":       (0.005, 0.50),
    "paddle":        (0.002, 0.15),
    "kayak":         (0.002, 0.15),
    "gonflable":     (0.002, 0.20),
    "autre_objet":   (0.001, 0.60),
}

# Poids pour le score de confiance final
POIDS_SCORE = {
    "modele":       0.35,
    "zone":         0.30,
    "voisinage":    0.20,
    "taille":       0.15,
}

# Seuil minimum de confiance pour garder une détection
SEUIL_CONFIANCE = 0.25

# Rayon de voisinage spatial (en pixels sur le masque redimensionné)
VOISINAGE_RAYON = 15

# ── Flask ────────────────────────────────────────────────────────────
FLASK_PORT = 5000
FLASK_DEBUG = True
SECRET_KEY = "segmentation-embarcations-2025"
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# Créer les dossiers nécessaires au démarrage
for d in [DATA_DIR, EMBEDDINGS_DIR, MODELS_SAVE_DIR, FIGURES_DIR, UPLOAD_DIR, MASKS_DIR, ANNOTATIONS_DIR]:
    os.makedirs(d, exist_ok=True)
