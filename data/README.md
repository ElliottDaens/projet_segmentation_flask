# data/

Données générées par l'application.

## Structure

| Dossier | Contenu | Format |
|---|---|---|
| `annotations/` | Polygones dessinés par l'utilisateur | JSON (éditables) |
| `masks/` | Masques de segmentation (pixel = indice de classe) | PNG 8-bit |
| `embeddings/` | Vecteurs de features pré-calculés | NumPy `.npy` |
| `saved_models/` | Poids du U-Net entraîné (384×384, 10 classes) | PyTorch `.pth` |

### Format des annotations

```json
{
  "filename": "MFDC2700.JPG",
  "image_width": 1065,
  "image_height": 1998,
  "polygons": [
    {"class_id": 1, "class_name": "eau", "color": "#0077FF", "points": [[x1,y1], ...]}
  ]
}
```

### Labels et historique

- `labels.json` — Labels globaux par image (outil d'étiquetage)
- `labels_history.json` — Journal des corrections
