# Segmentation hiérarchique & Système de voisinage

## Architecture en 2 niveaux

### Niveau 1 — Segmentation de la scène

Le premier niveau découpe l'image en **3 zones exclusives** :

| Zone | Couleur | Description |
|------|---------|-------------|
| **Mer / Eau** | 🟦 `#0077FF` | Surface d'eau, océan, rivière |
| **Terre / Sol** | 🟫 `#8B5A2B` | Rochers, plage, végétation, bâtiments |
| **Ciel** | ☁️ `#87CEEB` | Ciel, nuages |

Ce découpage est réalisé par le U-Net entraîné sur les annotations utilisateur. Les pixels prédits avec les IDs 1 (eau), 2 (terre), 3 (ciel) forment la carte de zones.

### Niveau 2 — Détection d'objets par zone

Dans chaque zone, le système recherche uniquement les **objets pertinents** :

| Zone | Objets recherchés |
|------|-------------------|
| **Mer** | bateau moteur, voilier, paddle, kayak, gonflable |
| **Terre** | autre objet |
| **Ciel** | autre objet |

Un bateau ne sera **jamais** cherché dans le ciel. Un oiseau ne sera pas cherché dans la mer.

## Classes conditionnelles (annotations utilisateur)

Le système **ne détecte que les classes que l'utilisateur a annotées** :

- Si tu n'as jamais dessiné de polygone « paddle », le système ne cherchera pas de paddle
- Dès qu'une annotation est créée pour une classe, elle est activée pour les prédictions futures
- Les 3 zones (mer, terre, ciel) sont toujours actives

### Comment activer une nouvelle classe ?

1. Aller dans l'onglet **Annotation**
2. Dessiner un polygone avec la classe souhaitée sur au moins une image
3. Sauvegarder
4. Relancer l'entraînement si nécessaire
5. La classe est maintenant active en prédiction

## Système de voisinage

Chaque objet détecté est évalué avec **4 critères de cohérence** :

### 1. Score du modèle (poids : 35%)

Probabilité moyenne que le réseau de neurones assigne à cette classe pour les pixels de l'objet. Plus le modèle est sûr, plus le score est élevé.

### 2. Score de cohérence zone (poids : 30%)

Pourcentage des pixels de l'objet qui sont dans la **bonne zone parent** :
- Un bateau avec 95% de ses pixels dans la zone mer → score zone = 0.95
- Un bateau avec 30% dans la mer et 70% dans le ciel → score zone = 0.30 → rejeté

### 3. Score de voisinage spatial (poids : 20%)

Analyse des **pixels voisins** dans un rayon de 15px autour de l'objet :
- Quelles classes sont présentes dans le voisinage ?
- Sont-elles sémantiquement compatibles ?
- Bateau entouré d'eau → compatible ✅
- Bateau entouré de ciel → suspect ⚠️

### 4. Score de cohérence taille (poids : 15%)

Vérification que la **taille relative** de l'objet par rapport à sa zone est cohérente :
- Un bateau occupe entre 0.5% et 40% de la zone mer → OK
- Un bateau de 2 pixels → trop petit, probablement du bruit
- Un bateau de 80% de la zone → trop grand, probablement une erreur de segmentation

### Score de confiance final

```
confiance = 0.35 × score_modèle + 0.30 × score_zone + 0.20 × score_voisinage + 0.15 × score_taille
```

Les détections avec une confiance **inférieure à 25%** sont rejetées et affichées séparément.

## Stratégie technique

| Composant | Choix technique | Justification |
|-----------|----------------|---------------|
| Modèle zones + objets | U-Net (ResNet-18) unique | Un seul modèle prédit toutes les classes ; on filtre ensuite par zone |
| Extraction d'objets | Composantes connexes (scipy.ndimage.label) | Sépare les instances individuelles d'une même classe |
| Voisinage spatial | Dilatation morphologique (OpenCV kernel elliptique) | Rapide, rayon configurable |
| Voisinage sémantique | Dictionnaire de compatibilité (config.py) | Règles métier explicites, facile à maintenir |
| Voisinage taille | Ratio surface objet / surface zone | Bornes min/max par classe dans config.py |
| Persistance | JSON (annotations) + PNG (masques) | Éditable, versionnable, compatible avec le pipeline d'entraînement |
