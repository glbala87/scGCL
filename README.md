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
- **Cell Type Annotation**: Auto-label clusters using marker databases
- **Interactive Visualization**: Plotly-based interactive UMAP, dashboards
- **Gene Set Enrichment**: GO, KEGG, Reactome pathway analysis
- **Export Functions**: Seurat, cellxgene, and Loom format export
- **Differential Abundance**: Compare cluster proportions between conditions
- **Cell Cycle Scoring**: Assign cell cycle phases (G1, S, G2M)
- **Doublet Detection**: Identify likely doublet cells
- **Cluster QC Metrics**: Quality control statistics per cluster
- **Batch Visualization**: Visualize batch effects and mixing
- **HTML Reports**: Generate comprehensive analysis reports
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

### Cell Type Annotation

Automatically annotate clusters with cell type labels using marker gene databases:

```python
from scgcl import annotate_clusters, annotate_adata, quick_annotate, PBMC_MARKERS

# Quick annotation (returns dict of cluster -> cell type)
annotations = quick_annotate(X, labels, gene_names, tissue='pbmc')
# {0: 'B cells', 1: 'CD4+ T cells', 2: 'NK cells', ...}

# Full annotation with confidence scores
result = annotate_clusters(
    X, labels, gene_names,
    tissue='pbmc',          # 'pbmc', 'brain', 'immune', 'tumor', or 'all'
    method='mean',          # 'mean', 'zscore', or 'percent'
    min_confidence=0.1
)
print(result.summary())

# Access annotations
cell_types = result.cell_types              # Per-cell labels
confidence = result.cell_confidence         # Per-cell confidence
cluster_map = result.cluster_annotations    # {cluster: cell_type}

# Annotate AnnData directly
result = annotate_adata(adata, key='scgcl_clusters', tissue='pbmc')
# Adds: adata.obs['cell_type'], adata.obs['cell_type_confidence']

# Use custom markers
custom_markers = {
    'Tumor cells': ['EPCAM', 'KRT19', 'MUC1'],
    'Fibroblasts': ['COL1A1', 'DCN', 'FAP'],
    'Immune': ['PTPRC', 'CD3D', 'CD68']
}
result = annotate_clusters(X, labels, gene_names, custom_markers=custom_markers)

# Visualize results
from scgcl import plot_annotation, plot_marker_heatmap
plot_annotation(result, save_path="annotation.png")
plot_marker_heatmap(result, top_n=5, save_path="markers.png")
```

**Built-in marker databases:**
- `PBMC_MARKERS`: CD4+/CD8+ T cells, B cells, NK cells, Monocytes, DCs, etc.
- `BRAIN_MARKERS`: Neurons, Astrocytes, Oligodendrocytes, Microglia, etc.
- `IMMUNE_MARKERS`: T/B/NK cells, Monocytes, Macrophages, Neutrophils, etc.
- `TUMOR_MARKERS`: Epithelial, Fibroblasts, Endothelial, Immune cells, etc.

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

### Interactive Visualization

Create interactive Plotly-based visualizations:

```python
from scgcl import (
    interactive_umap, interactive_3d, interactive_embedding,
    interactive_gene_expression, interactive_comparison,
    interactive_violin, create_dashboard
)

# Interactive UMAP with hover info
fig = interactive_umap(
    umap_coords, labels,
    hover_data={'Cell Type': cell_types, 'Confidence': confidence},
    title="My Clusters",
    save_path="clusters.html"  # Opens in browser
)

# Color by continuous value (e.g., gene expression)
fig = interactive_umap(
    umap_coords,
    color_by=gene_expression,
    color_label="CD3D Expression",
    colorscale='Viridis'
)

# 3D visualization
fig = interactive_3d(embeddings_3d, labels, save_path="3d_plot.html")

# Gene expression multi-panel
fig = interactive_gene_expression(
    umap_coords, X, gene_names,
    genes=['CD3D', 'CD14', 'MS4A1', 'NKG7'],
    ncols=2,
    save_path="markers.html"
)

# Compare two clusterings side by side
fig = interactive_comparison(
    umap_coords, labels1, labels2,
    title1="Method A", title2="Method B"
)

# Interactive violin plot
fig = interactive_violin(confidence, labels, title="Confidence by Cluster")

# Full analysis dashboard
create_dashboard(
    umap_coords, labels,
    expression=X,
    gene_names=gene_names,
    confidence=confidence,
    cell_types=cell_types,
    marker_genes=['CD3D'],
    save_path="dashboard.html"
)
```

### Differential Abundance Analysis

Compare cluster proportions between experimental conditions:

```python
from scgcl import differential_abundance, plot_differential_abundance, abundance_barplot

# Test for differential abundance between conditions
da_results = differential_abundance(
    labels, condition,         # Cluster labels and condition per cell
    condition1='Control',      # Reference condition
    condition2='Treatment',    # Comparison condition
    method='fisher',           # 'fisher', 'chi2', or 'permutation'
    min_cells=10               # Minimum cells to test
)

# Results include log2 fold change and adjusted p-values
print(da_results[['cluster', 'log2FC', 'pvalue', 'padj']])

# Volcano plot of results
plot_differential_abundance(
    da_results,
    pval_threshold=0.05,
    fc_threshold=0.5,
    save_path="da_volcano.png"
)

# Grouped bar chart of proportions
abundance_barplot(labels, condition, save_path="abundance.png")
```

### Cell Cycle Scoring

Assign cell cycle phases based on expression of known markers:

```python
from scgcl import score_cell_cycle, plot_cell_cycle, regress_cell_cycle

# Score cell cycle phases
result = score_cell_cycle(
    X, gene_names,
    s_genes=None,              # Use default S phase genes (Tirosh et al.)
    g2m_genes=None             # Use default G2M phase genes
)

# Access results
print(result.summary())
# {'n_cells': 1000, 'phase_counts': {'G1': 500, 'S': 300, 'G2M': 200}, ...}

phases = result.phase          # Per-cell phase assignments
s_scores = result.s_scores     # S phase scores
g2m_scores = result.g2m_scores # G2M phase scores

# Visualize
plot_cell_cycle(result, embedding=umap_coords, save_path="cell_cycle.png")

# Regress out cell cycle effects (optional)
X_corrected = regress_cell_cycle(X, s_scores, g2m_scores)

# For AnnData objects
from scgcl import score_cell_cycle_adata
score_cell_cycle_adata(adata)
# Adds: adata.obs['S_score'], adata.obs['G2M_score'], adata.obs['phase']
```

### Doublet Detection

Identify likely doublet cells using simulation-based approach:

```python
from scgcl import detect_doublets, detect_doublets_scrublet, plot_doublet_scores, filter_doublets

# Detect doublets on PCA embeddings
result = detect_doublets(
    pca_embeddings,
    expected_doublet_rate=0.05,    # Expected doublet rate
    n_neighbors=30
)

print(result.summary())
# {'n_cells': 1000, 'n_doublets': 52, 'doublet_rate': 0.052, ...}

# Access results
scores = result.doublet_scores     # Doublet score per cell
is_doublet = result.predicted_doublets  # Boolean predictions

# Scrublet-like method on raw counts
result = detect_doublets_scrublet(
    raw_counts,
    expected_doublet_rate=0.05,
    n_prin_comps=30
)

# Visualize
plot_doublet_scores(result, embedding=umap_coords, save_path="doublets.png")

# Filter out doublets
X_filtered, singlet_mask = filter_doublets(X, result, return_mask=True)

# For AnnData objects
from scgcl import detect_doublets_adata
detect_doublets_adata(adata, use_rep='X_pca')
# Adds: adata.obs['doublet_score'], adata.obs['predicted_doublet']
```

### Cluster QC Metrics

Compute quality control statistics per cluster:

```python
from scgcl import (
    compute_cluster_qc, compute_cluster_purity,
    compute_batch_mixing, batch_effect_test,
    plot_cluster_qc
)

# Compute QC metrics (total counts, genes detected, MT%, etc.)
qc_result = compute_cluster_qc(
    X, labels,
    gene_names=gene_names,
    mt_prefix='MT-',          # Mitochondrial gene prefix
    rb_prefix='RP'            # Ribosomal gene prefix
)

# Per-cluster statistics
print(qc_result.cluster_stats)
# cluster  n_cells  mean_counts  mean_n_genes  mean_pct_mt  ...

# Overall summary
print(qc_result.overall_stats)

# Visualize QC
plot_cluster_qc(qc_result, save_path="qc_metrics.png")

# Compute cluster purity (if ground truth available)
purity_df = compute_cluster_purity(labels, true_labels)
print(purity_df)

# Compute batch mixing
mixing = compute_batch_mixing(embeddings, batch, n_neighbors=50)
print(f"Overall batch mixing: {mixing['overall_mixing']:.3f}")

# Statistical test for batch effects
result = batch_effect_test(embeddings, batch, labels, n_permutations=1000)
print(f"Batch effect p-value: {result['pvalue']:.4f}")
print(result['interpretation'])
```

### Batch Visualization

Visualize batch effects and distribution:

```python
from scgcl import plot_batch_distribution, plot_batch_umap

# Batch composition per cluster
plot_batch_distribution(
    labels, batch,
    normalize=True,            # Show proportions
    save_path="batch_dist.png"
)

# UMAP colored by batch and cluster
plot_batch_umap(
    umap_coords, batch,
    labels=labels,             # Optional cluster labels
    save_path="batch_umap.png"
)
```

### HTML Report Generation

Generate comprehensive analysis reports:

```python
from scgcl import (
    HTMLReportGenerator,
    generate_clustering_report,
    generate_comparison_report
)

# Quick report generation
report_path = generate_clustering_report(
    X, labels,
    embedding=umap_coords,
    gene_names=gene_names,
    metrics={'ARI': 0.85, 'NMI': 0.78, 'Silhouette': 0.42},
    markers=markers_df,
    title="My scGCL Analysis",
    output_path="report.html"
)

# Custom report with HTMLReportGenerator
report = HTMLReportGenerator(title="Custom Analysis Report")

# Add summary statistics
report.add_summary({
    'Total Cells': 5000,
    'Clusters': 10,
    'Mean Silhouette': 0.45
})

# Add tables
report.add_table("Top Markers", markers_df)

# Add figures
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=labels, s=5)
report.add_figure("UMAP", fig)

# Add Plotly figures (interactive)
import plotly.express as px
fig = px.scatter(x=umap_coords[:, 0], y=umap_coords[:, 1], color=labels)
report.add_plotly_figure("Interactive UMAP", fig)

# Add text/markdown
report.add_markdown("Methods", """
## Analysis Pipeline
- **Preprocessing:** Log normalization, HVG selection
- **Clustering:** scGCL with 10 clusters
- **Annotation:** PBMC marker database
""")

# Save report
report.save("custom_report.html")

# Compare multiple clustering methods
comparison_path = generate_comparison_report(
    results=[{'ARI': 0.85}, {'ARI': 0.72}],
    labels_list=[labels1, labels2],
    method_names=['scGCL', 'Leiden'],
    embedding=umap_coords,
    output_path="comparison.html"
)
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
│   │   ├── refinement.py     # Subclustering and merging
│   │   ├── annotation.py     # Cell type annotation
│   │   ├── interactive.py    # Plotly interactive visualization
│   │   ├── differential.py   # Differential abundance analysis
│   │   ├── cell_cycle.py     # Cell cycle scoring
│   │   ├── doublet.py        # Doublet detection
│   │   ├── qc.py             # QC metrics and batch visualization
│   │   └── report.py         # HTML report generation
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
- matplotlib, seaborn (for static visualization)
- plotly (for interactive visualization)
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
