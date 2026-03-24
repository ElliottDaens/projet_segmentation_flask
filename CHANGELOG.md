# Changelog

Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
versioning [Semantic Versioning](https://semver.org/lang/fr/).

---

## [3.1.0] — 2026-03-24

### Performances
- Accélération GPU complète (RTX 4060) : inférence, lissage gaussien, morphologie, remplissage de zones
- Mixed precision (fp16) pour l'entraînement et l'inférence via `torch.amp`
- Dataset pré-chargé en RAM (zéro I/O disque pendant l'entraînement)
- Augmentation sur tensors PyTorch au lieu de PIL
- Cache du modèle en VRAM entre les prédictions
- `cudnn.benchmark` + `pin_memory` + `non_blocking` transfers
- Flask reloader désactivé (supprime la charge CPU constante)

### Corrections
- Fix bug scheduler à l'epoch de dégel (recréation optimizer + scheduler)
- Fix resize probas (type float64 pour OpenCV)
- Post-traitement morphologique remplacé par opérations GPU (`F.max_pool2d`)

---

## [3.0.0] — 2026-03-24

### Ajouté
- Segmentation hiérarchique en 2 niveaux : zones (mer/terre/ciel) → objets par zone
- Système de voisinage : 4 scores (modèle, zone, voisinage spatial, taille)
- Score de confiance combiné pondéré + filtrage automatique
- Classes conditionnelles : seules les classes annotées sont détectées
- Flou gaussien sur les probabilités avant argmax (lisse les bordures)
- Propagation par voisin le plus proche pour éliminer les pixels non reconnus
- Nettoyage morphologique (fermeture + ouverture + suppression micro-régions)
- Conservation du ratio original de l'image (pas de déformation en carré)
- Pixels non reconnus affichés en gris avec label "Non reconnu" et inclus dans les %
- Documentation Markdown du système hiérarchique

### Modifié
- Page Prédiction réécrite : affichage zones + objets + barres de score
- Résolution augmentée à 384×384
- Entraînement : class weights, freeze progressif, gradient clipping, AdamW + CosineAnnealing
- Dataset ×8 répétitions avec augmentation renforcée

---

## [2.0.0] — 2026-03-24

### Ajouté
- Annotation par polygones (remplace le pinceau)
- Mode plein écran (F) + zoom molette + pan (Espace+clic)
- Raccourcis clavier (Ctrl+S, Ctrl+Z, Entrée, Échap, Suppr)
- Sauvegarde des annotations en JSON + export masque PNG
- Fichier VERSION et CHANGELOG.md

---

## [1.0.0] — 2026-03-24

### Ajouté
- Application Flask : segmentation sémantique de paysages maritimes
- U-Net avec encodeur ResNet-18 pré-entraîné (transfer learning)
- Interface d'annotation par polygones pour délimiter les zones
- Page de prédiction automatique avec visualisation des résultats
- Analyse par régions (superpixels SLIC, contours Canny)
- Clustering d'images (K-Means, DBSCAN, UMAP) comme outil d'exploration
- Étiquetage d'images et corrections de labels
- Prétraitement : CLAHE, histogram equalization, data augmentation
- Documentation intégrée avec tutoriels sur chaque page
- 14 images de paysages maritimes
