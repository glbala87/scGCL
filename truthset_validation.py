#!/usr/bin/env python3
"""
Truthset Validation for scGCL
=============================
Validates scGCL clustering against ground truth labels using:
1. PBMC3k dataset with known cell type annotations
2. Simulated data with known cluster structure
3. Comprehensive metrics: ARI, NMI, ACC, Purity, Silhouette, etc.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from scgcl import ScGCL
from scgcl.utils.data import load_data, simulate_scrna_data
from scgcl.utils.evaluation import evaluate_clustering, clustering_metrics, ClusterStabilityAnalyzer
from scgcl.analysis.qc import compute_cluster_purity


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title):
    print(f"\n--- {title} ---")


# ============================================================
# VALIDATION 1: Simulated Data (known ground truth)
# ============================================================
print_header("VALIDATION 1: Simulated Data (Known Ground Truth)")

print_section("Generating simulated scRNA-seq data")
X_sim, y_sim = simulate_scrna_data(n_cells=500, n_genes=1000, n_clusters=5, dropout_rate=0.5, seed=42)
print(f"Shape: {X_sim.shape}, True clusters: {len(np.unique(y_sim))}")
print(f"Cells per cluster: {dict(zip(*np.unique(y_sim, return_counts=True)))}")

print_section("Running scGCL on simulated data")
t0 = time.time()
model_sim = ScGCL(n_clusters=5, pretrain_epochs=50, ssc_epochs=200, seed=42)
labels_sim = model_sim.fit_predict(X_sim, y=y_sim, verbose=True)
sim_time = time.time() - t0
print(f"\nTime: {sim_time:.1f}s")

print_section("Simulated Data — Full Metrics")
embeddings_sim = model_sim.get_embeddings()
metrics_sim = evaluate_clustering(y_sim, labels_sim, embeddings_sim, verbose=True)

print_section("Simulated Data — Cluster Purity")
purity_sim = compute_cluster_purity(labels_sim, y_sim)
print(purity_sim.to_string(index=False))
avg_purity = purity_sim['purity'].mean()
print(f"\nAverage Purity: {avg_purity:.4f}")

# ============================================================
# VALIDATION 2: PBMC3k Real Dataset
# ============================================================
print_header("VALIDATION 2: PBMC3k Dataset (Real Ground Truth)")

data_path = os.path.join(os.path.dirname(__file__), "data", "pbmc3k_processed.h5ad")
if os.path.exists(data_path):
    print_section("Loading PBMC3k processed data")
    import anndata
    adata = anndata.read_h5ad(data_path)
    from scipy import sparse
    X_pbmc = adata.X.toarray() if sparse.issparse(adata.X) else adata.X

    # Ground truth is in 'louvain' column (Seurat/scanpy cell type annotations)
    y_pbmc_raw = None
    label_col = None
    for col in ['cell_type', 'louvain', 'leiden', 'celltype', 'cluster']:
        if col in adata.obs.columns:
            y_pbmc_raw = adata.obs[col].values
            label_col = col
            break

    gene_names = adata.var_names.tolist()
    print(f"Shape: {X_pbmc.shape}")
    print(f"Label column: {label_col}")

    if y_pbmc_raw is not None:
        # Encode string labels to integers
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_pbmc = le.fit_transform(y_pbmc_raw)
        n_true_clusters = len(le.classes_)
        print(f"True cell types ({n_true_clusters}): {list(le.classes_)}")
        print(f"Cells per type: {dict(zip(le.classes_, np.bincount(y_pbmc)))}")

        # Handle NaN/Inf from already-processed data and reduce dims
        X_pbmc = np.nan_to_num(X_pbmc, nan=0.0, posinf=0.0, neginf=0.0)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X_pbmc)
        X_scaled = np.clip(X_scaled, -10, 10)
        X_pca = PCA(n_components=50).fit_transform(X_scaled)
        print(f"PCA reduced: {X_pca.shape}")

        print_section("Running scGCL on PBMC3k")
        t0 = time.time()
        model_pbmc = ScGCL(n_clusters=n_true_clusters, pretrain_epochs=100, ssc_epochs=300, seed=42)
        labels_pbmc = model_pbmc.fit_predict(X_pca, y=y_pbmc, preprocess=False, verbose=True)
        pbmc_time = time.time() - t0
        print(f"\nTime: {pbmc_time:.1f}s")

        print_section("PBMC3k — Full Metrics")
        embeddings_pbmc = model_pbmc.get_embeddings()
        metrics_pbmc = evaluate_clustering(y_pbmc, labels_pbmc, embeddings_pbmc, verbose=True)

        print_section("PBMC3k — Cluster Purity")
        purity_pbmc = compute_cluster_purity(labels_pbmc, y_pbmc_raw)
        print(purity_pbmc.to_string(index=False))
        avg_purity_pbmc = purity_pbmc['purity'].mean()
        print(f"\nAverage Purity: {avg_purity_pbmc:.4f}")

        print_section("PBMC3k — Confidence Scores")
        confidence = model_pbmc.get_confidence_scores()
        print(f"Mean confidence:   {confidence.mean():.4f}")
        print(f"Median confidence: {np.median(confidence):.4f}")
        print(f"Min confidence:    {confidence.min():.4f}")
        print(f"Max confidence:    {confidence.max():.4f}")
        low_conf = (confidence < 0.5).sum()
        print(f"Low confidence cells (<0.5): {low_conf} ({low_conf/len(confidence)*100:.1f}%)")
    else:
        print("WARNING: No ground truth labels found in PBMC3k dataset.")
else:
    print(f"PBMC3k processed data not found at {data_path}, skipping.")

# ============================================================
# VALIDATION 3: Stability Analysis (Simulated Data)
# ============================================================
print_header("VALIDATION 3: Stability Analysis (3 runs, Simulated Data)")

analyzer = ClusterStabilityAnalyzer(n_runs=3)
for i in range(3):
    print(f"\nRun {i+1}/3...")
    model_i = ScGCL(n_clusters=5, pretrain_epochs=50, ssc_epochs=200, seed=42 + i)
    labels_i = model_i.fit_predict(X_sim, verbose=False)
    emb_i = model_i.get_embeddings()
    analyzer.add_result(y_sim, labels_i, emb_i)

analyzer.print_summary()

# ============================================================
# SUMMARY
# ============================================================
print_header("TRUTHSET VALIDATION SUMMARY")

print("\nSimulated Data Results:")
for k, v in metrics_sim.items():
    status = "PASS" if (k in ['ARI', 'NMI', 'ACC'] and v > 0.3) or k not in ['ARI', 'NMI', 'ACC'] else "WARN"
    print(f"  [{status}] {k:<20} {v:.4f}")

if 'metrics_pbmc' in dir():
    print("\nPBMC3k Results:")
    for k, v in metrics_pbmc.items():
        status = "PASS" if (k in ['ARI', 'NMI', 'ACC'] and v > 0.3) or k not in ['ARI', 'NMI', 'ACC'] else "WARN"
        print(f"  [{status}] {k:<20} {v:.4f}")

print("\nValidation complete.")
