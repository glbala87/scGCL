# scGCL

**Single-Cell Graph Contrastive Learning for Cell Type Clustering**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/yourusername/scGCL)

A Python framework for clustering single-cell RNA sequencing (scRNA-seq) data using graph neural networks, debiased contrastive learning, and self-supervised clustering refinement.

## Highlights

- **Adaptive Graph Construction**: Dynamically adjusts k-nearest neighbors based on local cell density
- **Debiased Contrastive Learning**: Robust representation learning with alignment and uniformity objectives
- **Self-Supervised Refinement**: Iterative cluster assignment optimization using Student's t-distribution
- **GPU Acceleration**: Mixed precision training and tiled operations for large datasets (>10,000 cells)
- **Multiple Encoders**: Support for both GCN and GAT architectures
- **Confidence Scores**: Get prediction confidence for each cell
- **Marker Gene Detection**: Identify discriminative genes per cluster
- **Cluster Stability Analysis**: Bootstrap-based stability assessment
- **Subclustering & Merging**: Refine cluster resolution dynamically
- **Gene Set Enrichment**: GO, KEGG, Reactome pathway analysis
- **Export Functions**: Seurat, cellxgene, and Loom format export
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **CLI Interface**: Run clustering from command line
- **Scanpy Integration**: Seamless workflow with AnnData objects

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scGCL.git
cd scGCL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Optional: Install tuning dependencies
pip install -e ".[tuning]"
```

## Quick Start

```python
from scgcl import ScGCL

# Load your data (cells x genes matrix)
# X: expression matrix, y: ground truth labels (optional)

# Create and fit model
model = ScGCL(
    n_clusters=10,           # Number of clusters (None for auto-detection)
    hidden_dim=64,           # Hidden layer dimension
    pretrain_epochs=100,     # Contrastive pretraining epochs
    ssc_epochs=500,          # Self-supervised clustering epochs
)

# Fit and get cluster labels
labels = model.fit_predict(X, y=y)

# Get learned cell embeddings
embeddings = model.get_embeddings()

# Get confidence scores for predictions
confidence = model.get_confidence_scores()
```

### GPU Version (for large datasets)

```python
from scgcl import ScGCLGPU

model = ScGCLGPU(
    n_clusters=10,
    use_amp=True,            # Mixed precision training
    batch_size=256,
)
labels = model.fit_predict(X)
```

## Features

### Confidence Scores

Get prediction confidence and soft cluster assignments:

```python
# Confidence score (0-1) for each cell
confidence = model.get_confidence_scores()

# Full probability distribution over clusters
soft_assignments = model.get_soft_assignments()

# Predict with confidence for new data
labels, confidence, probs = model.predict_with_confidence(X_new)
```

### Marker Gene Detection

Identify discriminative genes for each cluster:

```python
from scgcl import find_marker_genes, rank_genes_groups

# Quick marker detection
markers = find_marker_genes(
    X, labels, gene_names,
    n_markers=10,
    method='wilcoxon'  # or 't-test'
)
print(markers)  # DataFrame with gene, cluster, score, pval_adj, logfoldchange

# Full differential expression analysis
result = rank_genes_groups(X, labels, gene_names)
top_markers = result.top_markers(n=5)
```

### Training Callbacks

Monitor and control training with callbacks:

```python
from scgcl import ScGCL, EarlyStopping, ProgressLogger, Timer

callbacks = [
    EarlyStopping(monitor='loss', patience=10),
    ProgressLogger(log_interval=10),
    Timer(verbose=True)
]

model = ScGCL(n_clusters=10)
model.fit(X, callbacks=callbacks, memory_profiling=True)
```

### Hyperparameter Tuning

Automated hyperparameter optimization with Optuna:

```python
from scgcl import HyperparameterTuner, quick_tune

# Quick tuning (20 trials)
best_params = quick_tune(X, n_clusters=5, n_trials=20)

# Full tuning with custom settings
tuner = HyperparameterTuner(
    n_trials=100,
    metric='silhouette',  # or 'ari', 'nmi', 'calinski', 'davies'
    timeout=3600  # 1 hour
)
result = tuner.tune(X, y_true=y)
print(result.summary())

# Get model with best parameters
best_model = tuner.get_best_model(X)
```

### Command-Line Interface

Run scGCL directly from the terminal:

```bash
# Basic clustering
scgcl cluster data.h5ad -o results/ -n 10

# With marker gene detection
scgcl cluster data.h5ad -o results/ --markers --n-markers 10

# Hyperparameter tuning
scgcl tune data.csv --n-trials 50 --metric silhouette

# Show help
scgcl cluster --help
scgcl info
```

### Scanpy Integration

Seamless integration with Scanpy workflows:

```python
import scanpy as sc
from scgcl.integration import scgcl, scgcl_markers

# Load and preprocess with Scanpy
adata = sc.read_h5ad("data.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)

# Run scGCL clustering
scgcl(adata, n_clusters=10)

# Results stored in AnnData:
# - adata.obs['scgcl_clusters']
# - adata.obs['scgcl_confidence']
# - adata.obsm['X_scgcl']

# Find marker genes
markers = scgcl_markers(adata, n_genes=10)

# Visualize
sc.pl.umap(adata, color='scgcl_clusters')
```

### Cluster Stability Analysis

Assess cluster robustness with bootstrap resampling:

```python
from scgcl import cluster_stability, consensus_clustering

# Assess stability with bootstrap
stability = cluster_stability(
    embeddings, labels,
    n_bootstrap=100,
    sample_fraction=0.8
)
print(stability.summary())
# Shows: Overall ARI, per-cluster stability, cell stability scores

# Get per-cell stability scores
cell_scores = stability.cell_stability  # How consistently each cell is assigned

# Consensus clustering for robust results
labels, consensus_matrix = consensus_clustering(
    embeddings,
    n_clusters=5,
    n_iterations=100
)
```

### Subclustering

Split clusters into finer subgroups for detailed analysis:

```python
from scgcl import subcluster, subcluster_recursive

# Subcluster a specific cluster into 3 parts
result = subcluster(
    X, labels,
    cluster_id=0,           # Which cluster to split
    n_subclusters=3,        # Number of subclusters
    method='kmeans',        # 'kmeans', 'hierarchical', or 'scgcl'
    embeddings=embeddings   # Use learned embeddings
)
print(result.summary())
new_labels = result.full_labels  # Updated labels for all cells

# Recursive subclustering until minimum size
final_labels = subcluster_recursive(
    X, labels,
    min_cluster_size=50,    # Stop when clusters are this small
    max_depth=3,            # Maximum recursion depth
    n_subclusters=2,        # Binary splits
    embeddings=embeddings
)
```

### Cluster Merging

Merge similar clusters to reduce over-clustering:

```python
from scgcl import merge_clusters, auto_merge, merge_by_markers, plot_merge_dendrogram

# Merge to target number of clusters
result = merge_clusters(
    labels, embeddings,
    n_clusters=5,           # Target cluster count
    method='centroid'       # 'centroid', 'nearest', 'farthest'
)
merged_labels = result.merged_labels

# Merge by distance threshold
result = merge_clusters(
    labels, embeddings,
    threshold=2.0           # Merge clusters closer than this
)

# Automatically merge small clusters
result = auto_merge(
    labels, embeddings,
    min_cluster_size=10     # Merge clusters smaller than this
)

# Merge based on marker gene similarity
result = merge_by_markers(
    X, labels,
    correlation_threshold=0.8,  # Merge if correlation > threshold
    n_top_markers=20
)

# Visualize merge hierarchy
plot_merge_dendrogram(labels, embeddings, threshold=2.0, save_path="merge.png")
```

### Cluster Visualization

Visualize clustering quality and structure:

```python
from scgcl import (
    silhouette_plot, cluster_dendrogram, cluster_heatmap,
    plot_confidence_distribution, plot_cluster_composition
)

# Silhouette plot showing per-cell scores
silhouette_plot(embeddings, labels, save_path="silhouette.png")

# Hierarchical cluster dendrogram
cluster_dendrogram(embeddings, labels, method='ward')

# Cluster expression heatmap
cluster_heatmap(X, labels, n_genes=50, gene_names=gene_names)

# Confidence distribution per cluster
confidence = model.get_confidence_scores()
plot_confidence_distribution(confidence, labels)

# Cluster composition (with optional batch info)
plot_cluster_composition(labels, batch=batch_labels)
```

### Gene Set Enrichment Analysis

Perform pathway analysis on cluster marker genes:

```python
from scgcl import (
    cluster_enrichment, quick_enrich, enrich,
    load_gene_sets, plot_enrichment
)

# Quick enrichment with printed results
markers = find_marker_genes(X, labels, gene_names)
results = quick_enrich(markers, source='go_bp', top_n=5)

# Full enrichment analysis
enrichment_df = cluster_enrichment(
    markers,
    source='kegg',        # 'go_bp', 'go_mf', 'go_cc', 'kegg', 'reactome'
    organism='human',
    pval_cutoff=0.05,
    top_n=10
)

# Plot results
plot_enrichment(enrichment_df, top_n=5, save_path="enrichment.png")

# Custom gene sets from GMT file
gene_sets = load_gene_sets(custom_gmt='my_pathways.gmt')
results = enrich(my_genes, gene_sets)
```

### Export to Other Tools

Export results for use with R/Seurat, cellxgene, and other tools:

```python
from scgcl import to_seurat, to_cellxgene, to_loom, export_markers_to_gmt

# Export for Seurat (creates counts.csv, metadata.csv, embeddings.csv, load_seurat.R)
to_seurat(
    X, labels, embeddings,
    confidence=confidence,
    gene_names=gene_names,
    output_dir='seurat_export'
)
# In R: source('seurat_export/load_seurat.R')

# Export for cellxgene browser
to_cellxgene(adata, output_path='cellxgene.h5ad', title='My Analysis')
# Then: cellxgene launch cellxgene.h5ad

# Export for SCENIC (Loom format)
to_loom(X, labels, embeddings, output_path='scgcl.loom')

# Export markers to GMT for GSEA
export_markers_to_gmt(markers, output_path='markers.gmt')
```

### Reproducibility

Ensure consistent results across runs:

```python
from scgcl import set_seed, ReproducibilityContext

# Global seed setting
set_seed(42, deterministic=True)

# Or use context manager
with ReproducibilityContext(seed=42):
    model = ScGCL(n_clusters=5)
    labels = model.fit_predict(X)
```

### Memory Profiling

Track memory usage during training:

```python
from scgcl import MemoryProfiler, profile_memory

# Enable during training
model.fit(X, memory_profiling=True)

# Or use profiler directly
profiler = MemoryProfiler(enabled=True)
profiler.start()
profiler.snapshot("before_training")
model.fit(X)
profiler.snapshot("after_training")
profiler.report()

# Context manager for code blocks
with profile_memory("training_block"):
    model.fit(X)
```

## Method Overview

```
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Raw Count   │────▶│ Preprocessing  │────▶│ Graph Building  │────▶│ Contrastive  │
│   Matrix     │     │ • Normalize    │     │ • Adaptive kNN  │     │ Pretraining  │
│              │     │ • Log1p        │     │ • SNN weights   │     │              │
│              │     │ • HVG + PCA    │     │                 │     │              │
└──────────────┘     └────────────────┘     └─────────────────┘     └──────┬───────┘
                                                                          │
                                                                          ▼
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Final      │◀────│     SSC        │◀────│  K-means Init   │◀────│  Embeddings  │
│  Clusters    │     │  Refinement    │     │                 │     │              │
└──────────────┘     └────────────────┘     └─────────────────┘     └──────────────┘
```

### Key Components

1. **Preprocessing**: Library normalization, log transformation, HVG selection, PCA
2. **Graph Construction**: Adaptive kNN with SNN (Shared Nearest Neighbor) edge weighting
3. **Contrastive Learning**: Debiased InfoNCE + alignment + uniformity losses
4. **Self-Supervised Clustering**: KL divergence minimization with iterative refinement

## API Reference

### ScGCL Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | int | None | Number of clusters (auto-estimated if None) |
| `hidden_dim` | int | 64 | Hidden layer dimension |
| `proj_dim` | int | 32 | Projection head dimension |
| `num_layers` | int | 2 | Number of GNN layers |
| `encoder_type` | str | 'gcn' | Encoder type ('gcn' or 'gat') |
| `k_neighbors` | int | 15 | Number of neighbors for kNN |
| `temperature` | float | 0.5 | Contrastive loss temperature |
| `pretrain_epochs` | int | 100 | Pretraining epochs |
| `ssc_epochs` | int | 500 | SSC refinement epochs |
| `lr` | float | 0.001 | Learning rate |
| `device` | str | 'cpu' | Device ('cpu' or 'cuda') |

### Methods

```python
# Core methods
model.fit(X, y=None, callbacks=None, memory_profiling=False)
model.fit_predict(X, y=None)
model.predict(X)

# Embeddings and confidence
model.get_embeddings()
model.get_confidence_scores()
model.get_soft_assignments()
model.predict_with_confidence(X)

# Persistence
model.save(path)
model.load(path, input_dim=None)
```

## Project Structure

```
scGCL/
├── scgcl/
│   ├── __init__.py
│   ├── model.py              # Main ScGCL class
│   ├── model_gpu.py          # GPU-optimized version
│   ├── cli.py                # Command-line interface
│   ├── tuning.py             # Hyperparameter tuning
│   ├── models/
│   │   ├── encoder.py        # Contrastive encoder
│   │   ├── graph_conv.py     # GCN layers
│   │   └── attention.py      # GAT layers
│   ├── losses/
│   │   ├── contrastive.py    # Contrastive losses
│   │   └── clustering.py     # Clustering losses
│   ├── clustering/
│   │   └── ssc.py            # Self-supervised clustering
│   ├── analysis/
│   │   ├── markers.py        # Marker gene detection
│   │   ├── stability.py      # Cluster stability analysis
│   │   ├── visualization.py  # Silhouette, dendrogram, heatmap
│   │   ├── enrichment.py     # Gene set enrichment
│   │   ├── export.py         # Seurat, cellxgene, Loom export
│   │   └── refinement.py     # Subclustering and merging
│   ├── integration/
│   │   └── scanpy_integration.py  # Scanpy workflow
│   └── utils/
│       ├── data.py           # Data loading/preprocessing
│       ├── graph.py          # Graph construction
│       ├── augmentation.py   # Data augmentation
│       ├── evaluation.py     # Metrics
│       ├── visualization.py  # Plotting
│       ├── callbacks.py      # Training callbacks
│       ├── memory.py         # Memory profiling
│       └── reproducibility.py # Seed management
├── examples/
│   └── tutorial.py
├── tests/
│   ├── test_basic.py
│   ├── test_quick_wins.py
│   ├── test_enhancements.py
│   └── test_analysis.py
├── requirements.txt
├── setup.py
└── README.md
```

## Examples

### With Simulated Data

```python
from scgcl.utils import simulate_scrna_data
from scgcl import ScGCL, find_marker_genes

# Generate test data
X, y = simulate_scrna_data(
    n_cells=1000,
    n_genes=2000,
    n_clusters=5,
    dropout_rate=0.5
)

# Cluster
model = ScGCL(n_clusters=5, pretrain_epochs=50)
labels = model.fit_predict(X, y=y)

# Find markers
gene_names = [f"Gene_{i}" for i in range(2000)]
markers = find_marker_genes(X, labels, gene_names)
print(markers.head(10))
```

### Visualization

```python
from scgcl.utils import plot_umap, plot_clusters

embeddings = model.get_embeddings()

# UMAP visualization
plot_umap(embeddings, labels, save_path="umap.png")

# Compare true vs predicted
plot_clusters(embeddings, y_true, y_pred, save_path="comparison.png")
```

### Multiple Runs with Statistics

```python
from scgcl import run_experiment

results = run_experiment(X, y, n_runs=10, n_clusters=10)
print(f"ARI: {results['ARI']['mean']:.4f} ± {results['ARI']['std']:.4f}")
print(f"NMI: {results['NMI']['mean']:.4f} ± {results['NMI']['std']:.4f}")
```

## Evaluation Metrics

The framework computes:
- **ARI** (Adjusted Rand Index)
- **NMI** (Normalized Mutual Information)
- **Clustering Accuracy** (Hungarian algorithm)
- **Silhouette Score**
- **Homogeneity / Completeness / V-measure**

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torch-geometric 2.1+
- numpy, pandas, scipy, scikit-learn
- anndata, scanpy
- matplotlib, seaborn (for visualization)
- optuna (optional, for tuning)
- gseapy (optional, for full gene set enrichment)
- loompy (optional, for Loom export)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [scAURA](https://github.com/bozdaglab/scAURA) and recent advances in graph contrastive learning for single-cell analysis.

## Citation

If you use scGCL in your research, please cite:

```bibtex
@software{scgcl2024,
  title={scGCL: Single-Cell Graph Contrastive Learning},
  author={scGCL Team},
  year={2024},
  url={https://github.com/yourusername/scGCL}
}
```
