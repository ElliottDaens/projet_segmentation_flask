# scripts/

Scripts CLI pour exécuter le pipeline sans l'interface web.

| Script | Commande | Description |
|---|---|---|
| `train_segmentation.py` | `python scripts/train_segmentation.py --epochs 80` | Entraîne le U-Net |
| `train_autoencoder.py` | `python scripts/train_autoencoder.py` | Entraîne l'auto-encodeur |
| `compute_embeddings.py` | `python scripts/compute_embeddings.py --method cnn` | Calcule les embeddings |
| `run_clustering.py` | `python scripts/run_clustering.py --n-clusters 4` | Lance le clustering |
| `generate_report.py` | `python scripts/generate_report.py` | Génère les slides reveal.js |
