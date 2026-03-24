# Verrous techniques & Corrections

## Problèmes identifiés

### 1. Volume de données limité

**Problème** : avec seulement 13 images, le risque de sur-apprentissage est très élevé pour les modèles deep learning. Les méthodes de clustering ont aussi peu de points pour estimer des distributions.

**Corrections mises en place** :
- **Data augmentation agressive** : rotation (±30°), flip horizontal/vertical, jitter de luminosité/contraste/saturation, crop aléatoire, flou gaussien
- **Transfer learning** : utilisation de ResNet-18 pré-entraîné sur ImageNet (1M+ images) avec gel des poids — pas d'entraînement nécessaire
- **Régularisation de l'auto-encodeur** : weight decay, early stopping, batch normalization
- **UMAP avec paramètres adaptés** : n_neighbors réduit (5) pour s'adapter au petit nombre d'échantillons

### 2. Variations de luminosité et de contraste

**Problème** : les photos d'embarcations sont prises en extérieur avec des conditions d'éclairage variables (soleil, nuages, reflets sur l'eau).

**Corrections mises en place** :
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) : normalise le contraste localement, sans saturer les zones déjà bien exposées
- **Balance des blancs** (gray-world) : corrige les dominantes de couleur
- **Jitter de couleur en augmentation** : rend le modèle robuste aux variations de luminosité
- L'image normalisée est accessible via `/api/image-normalized/<filename>`

### 3. Bruit et artefacts

**Problème** : bruit capteur, compression JPEG, reflets sur l'eau, embruns.

**Corrections mises en place** :
- **Débruitage non-local means** (`fastNlMeansDenoisingColored`) disponible en prétraitement
- **Flou gaussien en augmentation** : le modèle apprend à ignorer le bruit
- **Détection d'anomalies par régions** : les régions « anormales » (z-score élevé) peuvent correspondre à des artefacts

### 4. Taille et forme variables des objets

**Problème** : une barque occupe une petite partie de l'image tandis qu'un yacht peut la remplir entièrement. Les embarcations ont des formes très différentes.

**Corrections mises en place** :
- **Redimensionnement uniforme** (224×224) avec conservation du ratio via torchvision
- **Random resize crop** en augmentation : simule différentes échelles
- **Approche par régions** (superpixels SLIC) : analyse locale indépendante de la taille globale de l'objet
- **Détection de contours** (Canny) : capture la forme indépendamment de la taille

### 5. Déséquilibre potentiel entre catégories

**Problème** : certaines catégories (ex: bateau moteur) peuvent être surreprésentées tandis que d'autres (ex: kayak) sont rares.

**Corrections mises en place** :
- **DBSCAN** : détecte naturellement les catégories rares comme des petits clusters ou du « bruit »
- **Clustering semi-supervisé** : les labels manuels guident le clustering pour mieux représenter les catégories minoritaires
- **Suggestions par similarité** : aide à identifier et étiqueter les images rares en trouvant leurs voisins

## Impact observé

| Technique | Sans correction | Avec correction |
|-----------|----------------|-----------------|
| CLAHE | Embeddings sensibles à l'éclairage | Embeddings plus stables, clusters plus cohérents |
| Data augmentation | Sur-apprentissage de l'auto-encodeur | Meilleure généralisation, reconstruction plus robuste |
| Transfer learning (ResNet-18) | Pas de features pertinentes | Features riches en 512D, bon clustering dès le départ |
| UMAP (n_neighbors=5) | Trop de voisins pour 13 images | Projection cohérente même avec peu de données |
| Labels semi-supervisés | Clusters purement géométriques | Clusters alignés avec les catégories sémantiques |

## Méthodes non retenues

- **Morphologie mathématique** (érosion, dilatation) : plus adaptée à la segmentation binaire d'objets qu'au clustering d'images entières. Utile si on segmente l'embarcation du fond, mais hors scope pour le clustering global.
- **GAN pour la génération de données** : intéressant mais trop complexe et instable pour 13 images. Le transfer learning est plus fiable.
- **t-SNE** : remplacé par UMAP qui préserve mieux la structure globale et est plus rapide.
