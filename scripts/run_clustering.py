# -*- coding: utf-8 -*-
"""Exécution du clustering et export des résultats."""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def main():
    parser = argparse.ArgumentParser(description="Clustering des images")
    parser.add_argument("--method", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("--n-clusters", type=int, default=config.N_CLUSTERS)
    parser.add_argument("--eps", type=float, default=config.DBSCAN_EPS)
    parser.add_argument("--min-samples", type=int, default=config.DBSCAN_MIN_SAMPLES)
    args = parser.parse_args()

    emb_path = os.path.join(config.EMBEDDINGS_DIR, "embeddings.npy")
    fn_path = os.path.join(config.EMBEDDINGS_DIR, "filenames.json")

    if not os.path.isfile(emb_path):
        print("ERREUR : embeddings non trouvés. Lancez d'abord scripts/compute_embeddings.py")
        return

    embeddings = np.load(emb_path)
    with open(fn_path) as f:
        filenames = json.load(f)

    print(f"=== Clustering ({args.method}) ===")
    print(f"  Images : {len(filenames)}")
    print(f"  Embeddings : {embeddings.shape}")

    from clustering.pipeline import run_kmeans, run_dbscan, find_optimal_k
    from clustering.evaluation import compute_metrics

    if args.method == "kmeans":
        # Trouver K optimal
        print("\n  Recherche de K optimal…")
        optimal = find_optimal_k(embeddings)
        print(f"  K optimal (silhouette max) : {optimal['best_k']}")
        for k, sil in zip(optimal["k_values"], optimal["silhouettes"]):
            print(f"    K={k} → silhouette={sil:.3f}")

        n_clusters = args.n_clusters
        print(f"\n  K-Means avec K={n_clusters}…")
        result = run_kmeans(embeddings, n_clusters=n_clusters)
    else:
        print(f"\n  DBSCAN (eps={args.eps}, min_samples={args.min_samples})…")
        result = run_dbscan(embeddings, eps=args.eps, min_samples=args.min_samples)

    metrics = compute_metrics(embeddings, result["labels"])

    print(f"\n  Résultats :")
    print(f"    Clusters : {metrics['n_clusters']}")
    if "silhouette_avg" in metrics:
        print(f"    Silhouette : {metrics['silhouette_avg']:.3f}")
    if "davies_bouldin" in metrics:
        print(f"    Davies-Bouldin : {metrics['davies_bouldin']:.3f}")
    if "n_noise" in metrics:
        print(f"    Bruit : {metrics['n_noise']} points")

    print(f"\n  Assignation :")
    for fname, lbl in zip(filenames, result["labels"]):
        print(f"    {fname} → Cluster {lbl}")

    output_path = os.path.join(config.DATA_DIR, "clustering_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "result": result,
            "metrics": metrics,
            "filenames": filenames,
        }, f, indent=2)
    print(f"\n  Résultats sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()
