# Guide d'utilisation

## Lancement de l'application

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur Flask
python app.py
```

L'application est accessible sur **http://localhost:5000**.

## Chargement des données

Les images doivent être placées dans le dossier `Images/` à la racine du projet.
Formats supportés : JPG, JPEG, PNG, GIF, WEBP.

Le chemin est configurable dans `config.py` (variable `IMAGES_DIR`).

## Workflow recommandé

### Étape 1 : Calcul des embeddings

1. Aller dans l'onglet **Clustering**
2. Choisir la méthode d'extraction :
   - **CNN pré-entraîné (ResNet-18)** : rapide, bonnes features génériques — recommandé pour un premier essai
   - **Auto-encodeur** : nécessite un entraînement préalable (`python scripts/train_autoencoder.py`), features adaptées au dataset
3. Cliquer sur « Calculer les embeddings »

### Étape 2 : Clustering

1. Choisir l'algorithme :
   - **K-Means** : nombre de clusters fixe, bon pour des groupes bien séparés
   - **DBSCAN** : détecte automatiquement le nombre de clusters et les anomalies
2. Ajuster les paramètres (K, epsilon…)
3. Cliquer sur « Lancer le clustering »
4. Observer le scatter plot UMAP et les métriques

### Étape 3 : Étiquetage

1. Aller dans l'onglet **Étiquetage**
2. Cliquer sur une image dans la liste de gauche
3. Choisir un label parmi les catégories prédéfinies ou en saisir un personnalisé
4. Utiliser « Suggestions par similarité » pour voir les images proches dans l'espace latent

### Étape 4 : Corrections

1. Aller dans l'onglet **Corrections**
2. Filtrer par label pour voir les images d'une catégorie
3. Corriger individuellement (select en bas de chaque image) ou en masse (cases à cocher + bouton appliquer)
4. L'historique des modifications est visible en bas de page

### Étape 5 : Analyse par régions

1. Aller dans l'onglet **Régions**
2. Sélectionner une image et ajuster les paramètres (nombre de superpixels, clusters de régions)
3. Cliquer sur « Analyser » pour voir les superpixels, contours et anomalies détectées

## Interprétation des clusters

- **Scatter plot UMAP** : chaque point = une image. Les couleurs indiquent les clusters. Les points proches sont des images similaires.
- **Silhouette score** : entre -1 et 1. Plus c'est proche de 1, meilleure est la séparation des clusters.
- **Davies-Bouldin** : plus c'est bas, mieux c'est.
- **Méthode du coude** : le graphique montre l'inertie et la silhouette pour chaque K. Le K optimal est celui avec le meilleur compromis.
