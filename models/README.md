# models/

Réseaux de neurones et pipelines d'inférence.

## Fichiers

### `unet.py` — U-Net (ResNet-18)

Architecture de segmentation sémantique. L'encodeur ResNet-18 pré-entraîné sur ImageNet
extrait les features visuelles, le décodeur reconstruit le masque pixel par pixel.

- **Transfer learning** : encodeur gelé au début, dégel progressif (epoch 30 → enc3+enc4, epoch 50 → tout)
- **Class weights** : les classes rares (bateaux) pèsent plus dans la loss
- **Mixed precision fp16** : entraînement et inférence accélérés via `torch.amp`
- **Gradient clipping** et Dropout2d pour la stabilité et la régularisation
- Résolution 384×384, 10 classes de segmentation

Bibliothèques : `torch`, `torchvision.models` (ResNet-18)

### `hierarchical.py` — Pipeline hiérarchique + GPU

Pipeline complet en 2 niveaux :
1. Prédiction U-Net → probabilités par classe (GPU)
2. Lissage gaussien des probabilités (GPU, kernel séparable `F.conv2d`)
3. Argmax → masque de classes (GPU)
4. Post-traitement morphologique (GPU, `F.max_pool2d` = dilatation)
5. Propagation des pixels inconnus par dilatation itérative (GPU)
6. Construction du masque de zones pour le voisinage (GPU)
7. Extraction d'objets + analyse de voisinage → scores de confiance
8. Resize au ratio original de l'image

Le modèle est caché en VRAM entre les prédictions.

Bibliothèques : `torch`, `cv2`, `PIL`

### `autoencoder.py` — Auto-encodeur convolutionnel

Compression d'images en vecteur 128D pour le clustering (outil d'exploration).

### `feature_extractor.py` — Features CNN

ResNet-18 sans la dernière couche → vecteur 512D par image.

### `classifier.py` — MLP supervisé

Classification d'embeddings en catégories.
