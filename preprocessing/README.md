# preprocessing/

Prétraitement des images et chargement des données.

## Fichiers

### `segmentation_dataset.py` — Dataset pré-chargé en RAM

Charge toutes les paires (image, masque) en mémoire au démarrage.
Élimine les I/O disque pendant l'entraînement.

- **Pré-cache RAM** : images converties en tensors PyTorch au `__init__`
- **Répétition ×8** : chaque paire vue 8 fois par epoch avec augmentation différente
- **Augmentation sur tensors** : flip, rotation, jitter couleur — sans PIL
- **pin_memory** : transfert CPU→GPU asynchrone via DMA

Bibliothèques : `torch`, `torchvision.transforms.functional`, `PIL` (uniquement au chargement initial)

### `normalization.py` — Correction luminosité/contraste

CLAHE, histogram equalization, balance des blancs, débruitage.

Bibliothèques : `cv2` (OpenCV)

### `augmentation.py` — Augmentation pour CNN/auto-encodeur

Transforms composables pour le clustering.

Bibliothèques : `torchvision.transforms`

### `transforms.py` — Dataset générique

Chargement d'images pour le clustering et l'extraction d'embeddings.
