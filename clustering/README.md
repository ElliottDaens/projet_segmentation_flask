# clustering/

Analyse : voisinage, clustering, évaluation et régions.

## Fichiers

### `neighborhood.py` — Système de voisinage

Analyse chaque objet détecté avec 4 critères de cohérence :

| Score | Poids | Méthode |
|---|---|---|
| Modèle | 35% | Probabilité moyenne du U-Net pour l'objet |
| Zone | 30% | % de l'objet dans la bonne zone parente |
| Voisinage spatial | 20% | Compatibilité sémantique des pixels voisins (dilatation kernel 15px) |
| Taille | 15% | Ratio surface objet / surface zone vs bornes attendues |

Confiance = somme pondérée. Seuil de filtrage : 25%.

Bibliothèques : `cv2` (dilatation morphologique), `scipy.ndimage` (composantes connexes)

### `pipeline.py` — Clustering (K-Means, DBSCAN, semi-supervisé)

Outil d'exploration du dataset. Réduction PCA → UMAP pour la visualisation 2D.

Bibliothèques : `scikit-learn`, `umap-learn`

### `evaluation.py` — Métriques (silhouette, Davies-Bouldin)

### `region_analysis.py` — Superpixels SLIC + contours Canny

Bibliothèques : `scikit-image`, `cv2`
