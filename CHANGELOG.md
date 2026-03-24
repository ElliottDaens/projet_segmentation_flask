# Changelog

Toutes les modifications notables du projet sont documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

---

## [3.0.0] — 2026-03-24

### Ajouté
- **Segmentation hiérarchique en 2 niveaux** : découpe mer/terre/ciel (niveau 1) puis détection d'objets par zone (niveau 2)
- **Système de voisinage complet** avec 4 scores : modèle, zone, voisinage spatial, taille
- **Score de confiance combiné** pondéré + filtrage automatique des détections peu fiables
- **Classes conditionnelles** : seules les classes annotées par l'utilisateur sont détectées
- **Logique zone → objets** : un bateau n'est cherché que dans la mer, jamais dans le ciel
- **Barres de score visuelles** par détection dans l'interface de prédiction
- **Explications textuelles** pour chaque détection (pourquoi acceptée/rejetée)
- `models/hierarchical.py` — pipeline complet de prédiction hiérarchique
- `clustering/neighborhood.py` — analyse de voisinage spatial, sémantique, taille
- Documentation Markdown intégrée (onglet Hiérarchique & Voisinage)

### Modifié
- Page Prédiction entièrement réécrite pour afficher zones + objets + scores
- Mode plein écran (F) + zoom molette + pan (Espace+clic) sur l'annotation
- Images d'annotation agrandies, occupent tout l'espace disponible

---

## [2.0.0] — 2026-03-24

### Ajouté
- **Annotation par polygones** : outil de dessin interactif pour délimiter les zones (eau, terre, ciel, bateaux…) avec des polygones précis
- **Modèle U-Net** : architecture de segmentation sémantique avec encodeur ResNet-18 pré-entraîné
- **Page Prédiction** : segmentation automatique de nouvelles images après entraînement
- **10 classes de segmentation** : fond, eau, terre, ciel, bateau moteur, voilier, paddle, kayak, gonflable, autre
- **Raccourcis clavier** : Ctrl+S (sauvegarder), Ctrl+Z (annuler), Entrée (fermer polygone), Échap (annuler polygone), Suppr (supprimer sélection)
- **Sauvegarde des annotations en JSON** : polygones éditables + export automatique du masque PNG
- **Pipeline d'entraînement** : data augmentation spatialement cohérente (image + masque), gel/dégel progressif de l'encodeur
- Fichier `VERSION` et `CHANGELOG.md` pour le suivi de versions

### Modifié
- Canvas d'annotation agrandi (utilise tout l'espace disponible)
- Navigation : section « Segmentation » ajoutée dans la sidebar (Annotation + Prédiction)
- Images affichées en plus grand dans l'interface d'annotation

---

## [1.0.0] — 2026-03-24

### Ajouté
- **Application Flask complète** avec interface web moderne (thème sombre/clair)
- **Pipeline de clustering** : extraction d'embeddings (CNN ResNet-18 / auto-encodeur convolutionnel), réduction de dimension (PCA + UMAP), clustering (K-Means, DBSCAN)
- **Visualisation interactive** : scatter plot UMAP avec Plotly, métriques (silhouette, Davies-Bouldin), méthode du coude
- **Interface d'étiquetage** : labels manuels, suggestions par similarité dans l'espace latent
- **Corrections de labels** : modification individuelle et en masse, historique des corrections
- **Analyse par régions** : superpixels SLIC, détection de contours Canny, détection d'anomalies
- **Clustering semi-supervisé** : guidé par les labels manuels
- **Auto-encodeur convolutionnel** : 4 blocs encodeur/décodeur, bottleneck 128D
- **Classifieur MLP** : classification supervisée sur embeddings
- **Documentation intégrée** : onglets Utilisation et Verrous & Corrections en Markdown
- **Présentation reveal.js** : génération automatique de slides depuis Markdown
- **Prétraitement d'images** : CLAHE, histogram equalization, balance des blancs, débruitage, data augmentation
- **Scripts CLI** : entraînement, calcul d'embeddings, clustering, génération de rapport
- **13 images d'embarcations** dans le dataset
