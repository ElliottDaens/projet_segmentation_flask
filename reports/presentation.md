# Segmentation & Clustering d'Embarcations

Projet de segmentation et visualisation de données d'images — Application Flask

---

## Objectif du projet

- Construire un **pipeline complet** de segmentation et clustering d'images d'embarcations
- Développer une **interface web** pour la visualisation, l'étiquetage et la correction de labels
- Identifier et corriger les **verrous techniques** liés aux données
- Intégrer une **analyse par régions** pour la détection d'anomalies locales

---

## Dataset

- **13 images** de différentes embarcations (bateaux, voiliers, kayaks, etc.)
- Prises en conditions réelles : variations de luminosité, angle, taille
- Volume limité → nécessité de techniques adaptées (transfer learning, augmentation)

---

## Architecture du pipeline

1. **Prétraitement** : CLAHE, balance des blancs, data augmentation
2. **Extraction de features** : CNN pré-entraîné (ResNet-18) ou auto-encodeur convolutionnel
3. **Réduction de dimension** : PCA → UMAP (2D)
4. **Clustering** : K-Means (méthode du coude), DBSCAN (anomalies)
5. **Visualisation** : scatter plot interactif, grille par cluster
6. **Étiquetage** : interface manuelle avec suggestions par similarité
7. **Analyse régionale** : superpixels SLIC, détection de contours, anomalies

---

## Choix techniques

| Composant | Choix | Justification |
|-----------|-------|---------------|
| Framework web | Flask | Contrôle total sur l'UI, templates Jinja2 |
| Deep learning | PyTorch | Pythonique, bon écosystème, torchvision |
| CNN | ResNet-18 | Léger, performant, pré-entraîné ImageNet |
| Clustering | scikit-learn | KMeans, DBSCAN, métriques |
| Réduction dim. | UMAP | Meilleur que t-SNE pour la structure globale |
| Régions | SLIC (skimage) | Superpixels robustes et rapides |

---

## Verrous identifiés

- **Volume limité** (13 images) → transfer learning + augmentation agressive
- **Variations d'éclairage** → CLAHE + jitter de couleur
- **Bruit / artefacts** → débruitage non-local + flou gaussien en augmentation
- **Taille variable** → resize uniforme + random crop + analyse par régions
- **Déséquilibre de classes** → DBSCAN + clustering semi-supervisé

---

## Interface web

- **Page d'accueil** : tableau de bord du pipeline
- **Clustering** : calcul embeddings, choix algorithme, scatter UMAP, métriques
- **Étiquetage** : sélection image, labels prédéfinis/personnalisés, suggestions
- **Corrections** : filtrage par label, modification en masse, historique
- **Régions** : superpixels, contours, anomalies
- **Documentation** : guide d'utilisation et verrous techniques en Markdown

---

## Résultats

- Clustering efficace même avec 13 images grâce au transfer learning
- Interface utilisable pour l'étiquetage et la navigation dans les clusters
- Analyse par régions permet d'identifier les zones d'intérêt dans chaque image
- Pipeline reproductible : scripts CLI pour chaque étape

---

## Limites et pistes d'amélioration

- **Plus de données** : scraping web ou datasets publics (boats, maritime)
- **Fine-tuning** du CNN sur le dataset étiqueté
- **Détection d'objets** (YOLO, Faster R-CNN) pour localiser les embarcations
- **Segmentation sémantique** (U-Net) pour séparer embarcation / eau / ciel
- **Active learning** : proposer les images les plus incertaines à labelliser en priorité
- **Déploiement** : conteneurisation Docker, déploiement cloud
