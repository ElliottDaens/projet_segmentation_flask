# Segmentation hiérarchique de paysages maritimes

Application web de segmentation sémantique d'images de paysages maritimes.
Le système identifie automatiquement les zones (mer, terre, ciel) puis détecte
les embarcations dans chaque zone, avec un système de voisinage pour filtrer
les faux positifs.

## Principe

1. **Annotation** — L'utilisateur trace des polygones sur les images pour délimiter les zones
2. **Entraînement** — Un U-Net (ResNet-18) apprend à segmenter à partir des annotations
3. **Prédiction hiérarchique** — Découpe en zones → détection d'objets par zone → voisinage → filtrage

## Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| Annotation polygones | Dessin interactif, zoom molette, plein écran, raccourcis clavier |
| Segmentation hiérarchique | Niveau 1 (mer/terre/ciel) → Niveau 2 (objets par zone) |
| Système de voisinage | 4 scores de cohérence combinés en score de confiance |
| Classes conditionnelles | Seules les classes annotées sont détectées |
| Accélération GPU | Mixed precision fp16, post-traitement sur GPU, cache modèle VRAM |
| Post-traitement | Flou gaussien sur probabilités, morphologie, propagation voisin |

## Installation

```bash
git clone https://github.com/ElliottDaens/projet_segmentation_flask.git
cd projet_segmentation_flask
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Pour l'accélération GPU (NVIDIA) :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Utilisation

```bash
python app.py
```
Ouvrir http://localhost:5000

## Structure

```
├── app.py                  # Application Flask (25 routes)
├── config.py               # Configuration centralisée
├── models/                 # U-Net, pipeline hiérarchique, voisinage
├── preprocessing/          # Dataset pré-chargé RAM, augmentation tensors
├── clustering/             # Scores de voisinage, clustering, régions
├── templates/              # Interface web (annotation, prédiction, etc.)
├── static/                 # CSS, JS (outil polygones + zoom)
├── data/                   # Annotations JSON, masques PNG, modèles
├── Images/                 # Dataset (14 photos de paysages)
├── scripts/                # Scripts CLI
└── reports/                # Documentation Markdown
```

Chaque dossier contient un README détaillé.

## Technologies

| Composant | Bibliothèque | Rôle |
|---|---|---|
| Web | Flask | Application et API |
| Segmentation | PyTorch (U-Net + ResNet-18) | Prédiction pixel par pixel |
| GPU | CUDA + mixed precision fp16 | Accélération entraînement et inférence |
| Post-traitement | PyTorch (`F.max_pool2d`) | Morphologie et remplissage sur GPU |
| Voisinage | scipy + OpenCV | Cohérence spatiale, zone, taille |
| Visualisation | Plotly.js | Graphiques interactifs |
