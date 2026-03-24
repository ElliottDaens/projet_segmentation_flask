# Projet Segmentation d'Embarcations sous Flask

Application web de segmentation, clustering et étiquetage d'images d'embarcations (bateaux moteur, voiliers, paddles, kayaks, etc.).

## Fonctionnalités

- **Clustering d'images** : pipeline complet extraction de features (CNN pré-entraîné / auto-encodeur) → réduction de dimension (PCA + UMAP) → clustering (K-Means, DBSCAN)
- **Visualisation interactive** : scatter plot UMAP avec Plotly, navigation par cluster, grille de miniatures
- **Étiquetage manuel** : interface de labeling avec labels prédéfinis et personnalisés, persistance JSON
- **Correction de labels** : modification individuelle ou en masse, historique des corrections
- **Analyse par régions** : superpixels SLIC, détection de contours, identification d'anomalies locales
- **Documentation intégrée** : onglets Markdown (Utilisation, Verrous & Corrections)

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ElliottDaens/projet_segmentation_flask.git
cd projet_segmentation_flask

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Lancer l'application web

```bash
python app.py
```

Ouvrir http://localhost:5000 dans le navigateur.

### 2. Pipeline en ligne de commande

```bash
# Calculer les embeddings (CNN pré-entraîné)
python scripts/compute_embeddings.py --method cnn

# Entraîner l'auto-encodeur
python scripts/train_autoencoder.py --epochs 80

# Calculer les embeddings (auto-encodeur)
python scripts/compute_embeddings.py --method autoencoder

# Lancer le clustering
python scripts/run_clustering.py --n-clusters 4

# Générer le rapport / slides
python scripts/generate_report.py
```

## Structure du projet

```
├── config.py                  # Configuration centralisée
├── app.py                     # Application Flask
├── requirements.txt           # Dépendances
├── Images/                    # Dataset d'images d'embarcations
├── data/                      # Labels, embeddings, modèles sauvegardés
├── models/                    # Auto-encodeur, feature extractor, classifieur
├── preprocessing/             # Normalisation, augmentation, transforms
├── clustering/                # Pipeline clustering, évaluation, régions
├── templates/                 # Templates HTML Jinja2
├── static/                    # CSS, JS, uploads
├── scripts/                   # Scripts CLI utilitaires
└── reports/                   # Présentation, figures exportées
```

## Technologies

- **Python 3.10+**
- **Flask** — application web
- **PyTorch / torchvision** — modèles deep learning (auto-encodeur, CNN pré-entraîné)
- **scikit-learn** — clustering (K-Means, DBSCAN), métriques
- **scikit-image** — superpixels, traitement d'image
- **UMAP** — réduction de dimension
- **Plotly** — visualisation interactive
- **OpenCV** — prétraitement d'image
