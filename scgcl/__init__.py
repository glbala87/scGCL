"""
scGCL: Single-Cell Graph Contrastive Learning

A framework for clustering single-cell RNA sequencing data using
graph neural networks and contrastive learning.
"""

__version__ = "0.2.0"
__author__ = "scGCL Team"

from .model import ScGCL, run_experiment
from .model_gpu import ScGCLGPU

# Utilities
from .utils.reproducibility import set_seed, get_rng_state, set_rng_state, ReproducibilityContext
from .utils.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ProgressLogger,
    Timer,
    LambdaCallback,
    CheckpointCallback
)
from .utils.memory import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryStats,
    profile_memory,
    get_gpu_memory_info,
    clear_gpu_memory
)

# Analysis - Markers
from .analysis.markers import (
    find_marker_genes,
    rank_genes_groups,
    filter_markers,
    MarkerGeneResult
)

# Analysis - Stability
from .analysis.stability import (
    cluster_stability,
    consensus_clustering,
    plot_stability,
    StabilityResult
)

# Analysis - Visualization
from .analysis.visualization import (
    silhouette_plot,
    cluster_dendrogram,
    cluster_heatmap,
    plot_confidence_distribution,
    plot_cluster_composition
)

# Analysis - Export
from .analysis.export import (
    to_seurat,
    to_cellxgene,
    to_loom,
    export_markers_to_gmt
)

# Analysis - Enrichment
from .analysis.enrichment import (
    cluster_enrichment,
    enrich,
    load_gene_sets,
    load_gmt,
    plot_enrichment,
    quick_enrich,
    EnrichmentResult
)

# Analysis - Refinement
from .analysis.refinement import (
    subcluster,
    subcluster_recursive,
    merge_clusters,
    merge_by_markers,
    auto_merge,
    plot_merge_dendrogram,
    SubclusterResult,
    MergeResult
)

# Analysis - Annotation
from .analysis.annotation import (
    annotate_clusters,
    annotate_adata,
    get_marker_database,
    score_cell_types,
    plot_annotation,
    plot_marker_heatmap,
    quick_annotate,
    AnnotationResult,
    PBMC_MARKERS,
    BRAIN_MARKERS,
    IMMUNE_MARKERS,
    TUMOR_MARKERS,
)

# Analysis - Interactive
from .analysis.interactive import (
    interactive_umap,
    interactive_embedding,
    interactive_3d,
    interactive_gene_expression,
    interactive_comparison,
    interactive_cluster_composition,
    interactive_violin,
    create_dashboard,
)

# Analysis - Differential Abundance
from .analysis.differential import (
    differential_abundance,
    plot_differential_abundance,
    abundance_barplot,
    DifferentialAbundanceResult
)

# Analysis - Cell Cycle
from .analysis.cell_cycle import (
    score_cell_cycle,
    score_cell_cycle_adata,
    score_genes,
    plot_cell_cycle,
    regress_cell_cycle,
    CellCycleResult,
    S_PHASE_GENES,
    G2M_PHASE_GENES
)

# Analysis - Doublet Detection
from .analysis.doublet import (
    detect_doublets,
    detect_doublets_scrublet,
    detect_doublets_adata,
    plot_doublet_scores,
    filter_doublets,
    DoubletResult
)

# Analysis - QC and Batch
from .analysis.qc import (
    compute_cluster_qc,
    compute_cluster_purity,
    compute_batch_mixing,
    plot_cluster_qc,
    plot_batch_distribution,
    plot_batch_umap,
    batch_effect_test,
    ClusterQCResult
)

# Analysis - Report
from .analysis.report import (
    HTMLReportGenerator,
    generate_clustering_report,
    generate_comparison_report,
    ReportSection
)

# Tuning
from .tuning import (
    HyperparameterTuner,
    TuningResult,
    auto_tune,
    quick_tune
)

# Integration (lazy import to avoid dependency issues)
def _import_integration():
    from .integration import scgcl as scanpy_scgcl, scgcl_markers, to_anndata, from_anndata
    return scanpy_scgcl, scgcl_markers, to_anndata, from_anndata

__all__ = [
    # Main classes
    'ScGCL',
    'ScGCLGPU',
    'run_experiment',
    # Reproducibility
    'set_seed',
    'get_rng_state',
    'set_rng_state',
    'ReproducibilityContext',
    # Callbacks
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'ProgressLogger',
    'Timer',
    'LambdaCallback',
    'CheckpointCallback',
    # Memory
    'MemoryProfiler',
    'MemorySnapshot',
    'MemoryStats',
    'profile_memory',
    'get_gpu_memory_info',
    'clear_gpu_memory',
    # Analysis - Markers
    'find_marker_genes',
    'rank_genes_groups',
    'filter_markers',
    'MarkerGeneResult',
    # Analysis - Stability
    'cluster_stability',
    'consensus_clustering',
    'plot_stability',
    'StabilityResult',
    # Analysis - Visualization
    'silhouette_plot',
    'cluster_dendrogram',
    'cluster_heatmap',
    'plot_confidence_distribution',
    'plot_cluster_composition',
    # Analysis - Export
    'to_seurat',
    'to_cellxgene',
    'to_loom',
    'export_markers_to_gmt',
    # Analysis - Enrichment
    'cluster_enrichment',
    'enrich',
    'load_gene_sets',
    'load_gmt',
    'plot_enrichment',
    'quick_enrich',
    'EnrichmentResult',
    # Analysis - Refinement
    'subcluster',
    'subcluster_recursive',
    'merge_clusters',
    'merge_by_markers',
    'auto_merge',
    'plot_merge_dendrogram',
    'SubclusterResult',
    'MergeResult',
    # Analysis - Annotation
    'annotate_clusters',
    'annotate_adata',
    'get_marker_database',
    'score_cell_types',
    'plot_annotation',
    'plot_marker_heatmap',
    'quick_annotate',
    'AnnotationResult',
    'PBMC_MARKERS',
    'BRAIN_MARKERS',
    'IMMUNE_MARKERS',
    'TUMOR_MARKERS',
    # Analysis - Interactive
    'interactive_umap',
    'interactive_embedding',
    'interactive_3d',
    'interactive_gene_expression',
    'interactive_comparison',
    'interactive_cluster_composition',
    'interactive_violin',
    'create_dashboard',
    # Analysis - Differential Abundance
    'differential_abundance',
    'plot_differential_abundance',
    'abundance_barplot',
    'DifferentialAbundanceResult',
    # Analysis - Cell Cycle
    'score_cell_cycle',
    'score_cell_cycle_adata',
    'score_genes',
    'plot_cell_cycle',
    'regress_cell_cycle',
    'CellCycleResult',
    'S_PHASE_GENES',
    'G2M_PHASE_GENES',
    # Analysis - Doublet Detection
    'detect_doublets',
    'detect_doublets_scrublet',
    'detect_doublets_adata',
    'plot_doublet_scores',
    'filter_doublets',
    'DoubletResult',
    # Analysis - QC and Batch
    'compute_cluster_qc',
    'compute_cluster_purity',
    'compute_batch_mixing',
    'plot_cluster_qc',
    'plot_batch_distribution',
    'plot_batch_umap',
    'batch_effect_test',
    'ClusterQCResult',
    # Analysis - Report
    'HTMLReportGenerator',
    'generate_clustering_report',
    'generate_comparison_report',
    'ReportSection',
    # Tuning
    'HyperparameterTuner',
    'TuningResult',
    'auto_tune',
    'quick_tune',
]
