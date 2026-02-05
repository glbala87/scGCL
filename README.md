# scGCL

**Single-Cell Graph Contrastive Learning for Cell Type Clustering**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python framework for clustering single-cell RNA sequencing (scRNA-seq) data using graph neural networks, debiased contrastive learning, and self-supervised clustering refinement.

## Highlights

- **Adaptive Graph Construction**: Dynamically adjusts k-nearest neighbors based on local cell density
- **Debiased Contrastive Learning**: Robust representation learning with alignment and uniformity objectives
- **Self-Supervised Refinement**: Iterative cluster assignment optimization using Student's t-distribution
- **GPU Acceleration**: Mixed precision training and tiled operations for large datasets (>10,000 cells)
- **Multiple Encoders**: Support for both GCN and GAT architectures

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scGCL.git
cd scGCL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
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
model.fit(X, y=None)           # Fit the model
model.fit_predict(X, y=None)   # Fit and return labels
model.predict(X)               # Predict labels for new data
model.get_embeddings()         # Get learned embeddings
model.save(path)               # Save model
model.load(path)               # Load model
```

## Examples

### With Simulated Data

```python
from scgcl.utils import simulate_scrna_data
from scgcl import ScGCL

# Generate test data
X, y = simulate_scrna_data(
    n_cells=1000,
    n_genes=2000,
    n_clusters=5,
    dropout_rate=0.5
)

model = ScGCL(n_clusters=5, pretrain_epochs=50)
labels = model.fit_predict(X, y=y)
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

## Project Structure

```
scGCL/
├── scgcl/
│   ├── __init__.py
│   ├── model.py              # Main ScGCL class
│   ├── model_gpu.py          # GPU-optimized version
│   ├── models/
│   │   ├── encoder.py        # Contrastive encoder
│   │   ├── graph_conv.py     # GCN layers
│   │   └── attention.py      # GAT layers
│   ├── losses/
│   │   ├── contrastive.py    # Contrastive losses
│   │   └── clustering.py     # Clustering losses
│   ├── clustering/
│   │   └── ssc.py            # Self-supervised clustering
│   └── utils/
│       ├── data.py           # Data loading/preprocessing
│       ├── graph.py          # Graph construction
│       ├── augmentation.py   # Data augmentation
│       ├── evaluation.py     # Metrics
│       └── visualization.py  # Plotting
├── examples/
│   └── tutorial.py
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Evaluation Metrics

The framework computes:
- **ARI** (Adjusted Rand Index)
- **NMI** (Normalized Mutual Information)
- **Clustering Accuracy** (Hungarian algorithm)
- **Silhouette Score**
- **Homogeneity / Completeness / V-measure**

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [scAURA](https://github.com/bozdaglab/scAURA) and recent advances in graph contrastive learning for single-cell analysis.
