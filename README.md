# scGCL

**Single-Cell Graph Contrastive Learning for Cell Type Clustering**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/glbala87/scGCL)
[![Tests](https://img.shields.io/badge/tests-176%20passed-brightgreen.svg)](#testing)
[![CI](https://github.com/glbala87/scGCL/actions/workflows/python-app.yml/badge.svg)](https://github.com/glbala87/scGCL/actions)

A production-ready Python framework for clustering single-cell RNA sequencing (scRNA-seq) data using graph neural networks, debiased contrastive learning, and self-supervised clustering refinement.

---

## Benchmark Results

Validated against ground truth labels on standard datasets:

| Dataset | Cells | Types | ARI | NMI | ACC | Purity | Silhouette |
|---------|-------|-------|-----|-----|-----|--------|------------|
| **Simulated** | 500 | 5 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.8538 |
| **PBMC3k** | 2,638 | 8 | **0.9156** | **0.8809** | **0.9530** | **0.9047** | 0.7446 |

Stability across 3 runs (simulated data): ARI std = **0.0000** (perfectly reproducible).

Per-cluster purity on PBMC3k:

| Cell Type | Purity | Cells |
|-----------|--------|-------|
| Dendritic cells | 1.000 | 27 |
| CD14+ Monocytes | 0.993 | 454 |
| B cells | 0.988 | 343 |
| CD4 T cells | 0.959 | 1,180 |
| NK cells | 0.929 | 155 |
| CD8 T cells | 0.926 | 284 |
| FCGR3A+ Monocytes | 0.841 | 170 |
| Megakaryocytes | 0.600 | 25 |

---

## Highlights

### Core Algorithm
- **Adaptive Graph Construction** — dynamically adjusts k-nearest neighbors based on local cell density
- **Debiased Contrastive Learning** — representation learning with alignment and uniformity objectives
- **Self-Supervised Refinement** — iterative cluster optimization using Student's t-distribution
- **GPU Acceleration** — mixed precision training and tiled operations for large datasets (>10k cells)
- **Multiple Encoders** — GCN and GAT architectures
- **Confidence Scores** — prediction confidence for each cell

### Analysis Suite
- **Marker Gene Detection** — Wilcoxon, t-test, logfoldchange ranking
- **Cluster Stability** — bootstrap-based assessment with consensus clustering
- **Subclustering & Merging** — refine cluster resolution dynamically
- **Cell Type Annotation** — auto-label using built-in marker databases (PBMC, Brain, Immune, Tumor)
- **Trajectory Analysis** — pseudotime, diffusion maps, Slingshot, PAGA
- **Batch Integration** — Harmony, MNN, ComBat correction
- **Differential Expression** — Wilcoxon, t-test, negative binomial, logistic regression
- **Gene Set Enrichment** — GO, KEGG, Reactome pathway analysis
- **Cell Cycle Scoring** — G1, S, G2M phase assignment
- **Doublet Detection** — simulation-based doublet identification
- **Differential Abundance** — cluster proportion comparison between conditions

### Visualization & Export
- **Interactive Visualization** — Plotly-based UMAP, 3D plots, dashboards
- **HTML Reports** — comprehensive analysis reports
- **Export** — Seurat, cellxgene, Loom format
- **Cluster QC** — quality control metrics per cluster

### Engineering
- **Input Validation** — NaN/Inf handling, sparse matrix support, shape/dtype checks
- **Structured Logging** — `logging` module throughout (no print statements)
- **176 Tests** — unit, integration, and edge case coverage
- **CI/CD** — GitHub Actions testing Python 3.9–3.12
- **CLI Interface** — run clustering from the command line
- **Scanpy Integration** — seamless AnnData workflows
- **Hyperparameter Tuning** — automated optimization with Optuna
- **Reproducibility** — seed management, deterministic mode
- **Memory Profiling** — track CPU/GPU usage during training

---

## Installation

```bash
# Clone the repository
git clone https://github.com/glbala87/scGCL.git
cd scGCL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Optional extras
pip install -e ".[tuning]"      # Optuna for hyperparameter tuning
pip install -e ".[profiling]"   # psutil for memory profiling
pip install -e ".[dev]"         # pytest, black, flake8, isort
```

---

## Quick Start

```python
import logging
logging.basicConfig(level=logging.INFO)  # Enable scGCL log output

from scgcl import ScGCL

# Create and fit model
model = ScGCL(
    n_clusters=10,           # None for auto-detection
    hidden_dim=64,
    pretrain_epochs=100,
    ssc_epochs=500,
)

# Fit and get cluster labels
labels = model.fit_predict(X, y=y)  # y is optional ground truth

# Get learned cell embeddings
embeddings = model.get_embeddings()

# Get confidence scores for predictions
confidence = model.get_confidence_scores()

# Full probability distribution over clusters
soft_assignments = model.get_soft_assignments()
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

---

## Features

### Marker Gene Detection

```python
from scgcl import find_marker_genes, rank_genes_groups

# Quick marker detection
markers = find_marker_genes(X, labels, gene_names, n_markers=10, method='wilcoxon')

# Full differential expression per cluster
result = rank_genes_groups(X, labels, gene_names)
top_markers = result.top_markers(n=5)
```

### Training Callbacks

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

```python
from scgcl import HyperparameterTuner, quick_tune

# Quick tuning
best_params = quick_tune(X, n_clusters=5, n_trials=20)

# Full tuning
tuner = HyperparameterTuner(n_trials=100, metric='silhouette', timeout=3600)
result = tuner.tune(X, y_true=y)
best_model = tuner.get_best_model(X)
```

### Command-Line Interface

```bash
scgcl cluster data.h5ad -o results/ -n 10
scgcl cluster data.h5ad -o results/ --markers --n-markers 10
scgcl tune data.csv --n-trials 50 --metric silhouette
scgcl info
```

### Scanpy Integration

```python
import scanpy as sc
from scgcl.integration import scgcl, scgcl_markers

adata = sc.read_h5ad("data.h5ad")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)

scgcl(adata, n_clusters=10)
# Results: adata.obs['scgcl_clusters'], adata.obs['scgcl_confidence'], adata.obsm['X_scgcl']

markers = scgcl_markers(adata, n_genes=10)
sc.pl.umap(adata, color='scgcl_clusters')
```

### Cell Type Annotation

```python
from scgcl import annotate_clusters, quick_annotate, PBMC_MARKERS

# Quick annotation
annotations = quick_annotate(X, labels, gene_names, tissue='pbmc')
# {0: 'B cells', 1: 'CD4+ T cells', 2: 'NK cells', ...}

# Full annotation with confidence
result = annotate_clusters(X, labels, gene_names, tissue='pbmc')
cell_types = result.cell_types
confidence = result.cell_confidence

# Custom markers
custom_markers = {
    'Tumor cells': ['EPCAM', 'KRT19', 'MUC1'],
    'Fibroblasts': ['COL1A1', 'DCN', 'FAP'],
}
result = annotate_clusters(X, labels, gene_names, custom_markers=custom_markers)
```

**Built-in marker databases:** PBMC, Brain, Immune, Tumor.

### Cluster Stability & Refinement

```python
from scgcl import cluster_stability, subcluster, merge_clusters, auto_merge

# Stability analysis
stability = cluster_stability(embeddings, labels, n_bootstrap=100)
print(stability.summary())

# Subclustering
result = subcluster(X, labels, cluster_id=0, n_subclusters=3, embeddings=embeddings)

# Merge similar clusters
result = merge_clusters(labels, embeddings, n_clusters=5)
result = auto_merge(labels, embeddings, min_cluster_size=10)
```

### Trajectory Analysis

```python
from scgcl import diffusion_pseudotime, slingshot, paga, infer_trajectory

# Diffusion pseudotime
result = diffusion_pseudotime(embeddings, root=0)

# Slingshot trajectory inference
result = slingshot(embeddings, labels, start_cluster=0, end_clusters=[2, 3])

# PAGA graph abstraction
connectivity, edges = paga(embeddings, labels, n_neighbors=15)

# Unified interface
result = infer_trajectory(embeddings, labels, method='slingshot')
```

### Batch Integration

```python
from scgcl import harmony, mnn_correct, combat, integrate, compute_lisi

# Harmony
result = harmony(pca_embeddings, batch, n_clusters=100, theta=2.0)

# Unified interface
result = integrate(expression, batch, method='harmony')

# Evaluate integration
lisi = compute_lisi(embeddings, batch, n_neighbors=30)
```

### Differential Expression

```python
from scgcl import differential_expression, plot_volcano

result = differential_expression(
    expression, groups, gene_names,
    group1='control', group2='treatment',
    method='wilcoxon'
)

sig_genes = result.significant()    # padj < 0.05, |log2FC| > 0.5
up_genes = result.upregulated()
down_genes = result.downregulated()

plot_volcano(result, top_n=10, save_path="volcano.png")
```

### Interactive Visualization

```python
from scgcl import interactive_umap, interactive_3d, create_dashboard

fig = interactive_umap(umap_coords, labels, save_path="clusters.html")
fig = interactive_3d(embeddings_3d, labels, save_path="3d_plot.html")

create_dashboard(
    umap_coords, labels,
    expression=X, gene_names=gene_names,
    confidence=confidence, cell_types=cell_types,
    save_path="dashboard.html"
)
```

### Additional Features

```python
# Cell cycle scoring
from scgcl import score_cell_cycle, plot_cell_cycle
result = score_cell_cycle(X, gene_names)

# Doublet detection
from scgcl import detect_doublets, filter_doublets
result = detect_doublets(pca_embeddings, expected_doublet_rate=0.05)
X_filtered, mask = filter_doublets(X, result, return_mask=True)

# Cluster QC
from scgcl import compute_cluster_qc, compute_cluster_purity
qc_result = compute_cluster_qc(X, labels, gene_names=gene_names)
purity_df = compute_cluster_purity(labels, true_labels)

# Gene set enrichment
from scgcl import cluster_enrichment, quick_enrich
results = quick_enrich(markers, source='go_bp', top_n=5)

# Export
from scgcl import to_seurat, to_cellxgene, to_loom
to_seurat(X, labels, embeddings, gene_names=gene_names, output_dir='seurat_export')

# HTML reports
from scgcl import generate_clustering_report
generate_clustering_report(X, labels, embedding=umap_coords, output_path="report.html")

# Reproducibility
from scgcl import set_seed, ReproducibilityContext
with ReproducibilityContext(seed=42):
    labels = model.fit_predict(X)

# Memory profiling
from scgcl import profile_memory
with profile_memory("training"):
    model.fit(X)
```

---

## Method Overview

```
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Raw Count   │────>│ Preprocessing  │────>│ Graph Building  │────>│ Contrastive  │
│   Matrix     │     │ • Normalize    │     │ • Adaptive kNN  │     │ Pretraining  │
│              │     │ • Log1p        │     │ • SNN weights   │     │              │
│              │     │ • HVG + PCA    │     │                 │     │              │
└──────────────┘     └────────────────┘     └─────────────────┘     └──────┬───────┘
                                                                          │
                                                                          v
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Final      │<────│     SSC        │<────│  K-means Init   │<────│  Embeddings  │
│  Clusters    │     │  Refinement    │     │                 │     │              │
└──────────────┘     └────────────────┘     └─────────────────┘     └──────────────┘
```

**Key Components:**

1. **Preprocessing** — Library normalization, log transformation, HVG selection, PCA
2. **Graph Construction** — Adaptive kNN with SNN (Shared Nearest Neighbor) edge weighting
3. **Contrastive Learning** — Debiased InfoNCE + alignment + uniformity losses
4. **Self-Supervised Clustering** — KL divergence minimization with iterative refinement

---

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
# Core
model.fit(X, y=None, preprocess=True, callbacks=None, memory_profiling=False)
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

### Evaluation Metrics

The framework computes:
- **ARI** (Adjusted Rand Index)
- **NMI** (Normalized Mutual Information)
- **Clustering Accuracy** (Hungarian algorithm)
- **Silhouette Score**
- **Calinski-Harabasz Score**
- **Davies-Bouldin Score**
- **Homogeneity / Completeness / V-measure**

---

## Project Structure

```
scGCL/
├── scgcl/
│   ├── __init__.py              # Package init with logging setup
│   ├── model.py                 # Main ScGCL class (CPU)
│   ├── model_gpu.py             # GPU-optimized version
│   ├── cli.py                   # Command-line interface
│   ├── tuning.py                # Hyperparameter tuning
│   ├── models/
│   │   ├── encoder.py           # Contrastive encoder
│   │   ├── graph_conv.py        # GCN layers
│   │   └── attention.py         # GAT layers
│   ├── losses/
│   │   ├── contrastive.py       # Contrastive losses
│   │   └── clustering.py        # Clustering losses (KL, Student-t)
│   ├── clustering/
│   │   └── ssc.py               # Self-supervised clustering
│   ├── analysis/
│   │   ├── markers.py           # Marker gene detection
│   │   ├── stability.py         # Cluster stability analysis
│   │   ├── visualization.py     # Static plots
│   │   ├── interactive.py       # Plotly interactive visualization
│   │   ├── enrichment.py        # Gene set enrichment
│   │   ├── export.py            # Seurat, cellxgene, Loom export
│   │   ├── refinement.py        # Subclustering and merging
│   │   ├── annotation.py        # Cell type annotation
│   │   ├── trajectory.py        # Trajectory and pseudotime
│   │   ├── batch_integration.py # Batch correction methods
│   │   ├── differential_expression.py  # DE analysis
│   │   ├── differential.py      # Differential abundance
│   │   ├── cell_cycle.py        # Cell cycle scoring
│   │   ├── doublet.py           # Doublet detection
│   │   ├── qc.py                # QC metrics and batch visualization
│   │   └── report.py            # HTML report generation
│   ├── integration/
│   │   └── scanpy_integration.py
│   └── utils/
│       ├── data.py              # Data loading/preprocessing
│       ├── graph.py             # Graph construction
│       ├── augmentation.py      # Data augmentation
│       ├── evaluation.py        # Clustering metrics
│       ├── callbacks.py         # Training callbacks
│       ├── memory.py            # Memory profiling
│       └── reproducibility.py   # Seed management
├── tests/
│   ├── test_basic.py            # Core functionality
│   ├── test_edge_cases.py       # Edge cases and robustness
│   ├── test_analysis.py         # Analysis modules
│   ├── test_enhancements.py     # Advanced features
│   └── test_quick_wins.py       # Callbacks, profiling, persistence
├── examples/
│   └── tutorial.py
├── truthset_validation.py       # Benchmark validation script
├── pyproject.toml
├── setup.py
├── requirements.txt
└── LICENSE
```

---

## Testing

```bash
# Run all tests
pytest tests/ --tb=short

# Run with coverage
pytest tests/ --cov=scgcl --cov-report=term-missing

# Run truthset validation
python truthset_validation.py
```

**176 tests** covering core model, analysis modules, edge cases, and integration.

---

## Requirements

- Python 3.9+
- PyTorch 1.12+
- torch-geometric 2.1+
- numpy, pandas, scipy, scikit-learn
- anndata, scanpy
- matplotlib, seaborn, plotly
- tqdm, joblib

**Optional:**
- `optuna` — hyperparameter tuning
- `psutil` — memory profiling
- `gseapy` — full gene set enrichment
- `loompy` — Loom export

---

## Logging

scGCL uses Python's `logging` module. Enable output with:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

For production pipelines, configure handlers and formatters as needed. scGCL attaches a `NullHandler` by default, so no output is produced unless you configure logging.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [scAURA](https://github.com/bozdaglab/scAURA) and recent advances in graph contrastive learning for single-cell analysis.

## Citation

If you use scGCL in your research, please cite:

```bibtex
@software{scgcl2024,
  title={scGCL: Single-Cell Graph Contrastive Learning},
  author={BalaSubramani Gattu Linga},
  year={2024},
  url={https://github.com/glbala87/scGCL}
}
```
