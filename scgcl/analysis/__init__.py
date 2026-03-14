"""Analysis module for downstream analysis of clustering results."""

from .markers import (
    find_marker_genes,
    rank_genes_groups,
    filter_markers,
    MarkerGeneResult
)

from .stability import (
    cluster_stability,
    consensus_clustering,
    plot_stability,
    StabilityResult
)

from .visualization import (
    silhouette_plot,
    cluster_dendrogram,
    cluster_heatmap,
    plot_confidence_distribution,
    plot_cluster_composition
)

from .export import (
    to_seurat,
    to_cellxgene,
    to_loom,
    export_markers_to_gmt
)

from .enrichment import (
    cluster_enrichment,
    enrich,
    load_gene_sets,
    load_gmt,
    plot_enrichment,
    quick_enrich,
    EnrichmentResult
)

from .refinement import (
    subcluster,
    subcluster_recursive,
    merge_clusters,
    merge_by_markers,
    auto_merge,
    plot_merge_dendrogram,
    SubclusterResult,
    MergeResult
)

from .annotation import (
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

from .interactive import (
    interactive_umap,
    interactive_embedding,
    interactive_3d,
    interactive_gene_expression,
    interactive_comparison,
    interactive_cluster_composition,
    interactive_violin,
    create_dashboard,
)

__all__ = [
    # Markers
    'find_marker_genes',
    'rank_genes_groups',
    'filter_markers',
    'MarkerGeneResult',
    # Stability
    'cluster_stability',
    'consensus_clustering',
    'plot_stability',
    'StabilityResult',
    # Visualization
    'silhouette_plot',
    'cluster_dendrogram',
    'cluster_heatmap',
    'plot_confidence_distribution',
    'plot_cluster_composition',
    # Export
    'to_seurat',
    'to_cellxgene',
    'to_loom',
    'export_markers_to_gmt',
    # Enrichment
    'cluster_enrichment',
    'enrich',
    'load_gene_sets',
    'load_gmt',
    'plot_enrichment',
    'quick_enrich',
    'EnrichmentResult',
    # Refinement
    'subcluster',
    'subcluster_recursive',
    'merge_clusters',
    'merge_by_markers',
    'auto_merge',
    'plot_merge_dendrogram',
    'SubclusterResult',
    'MergeResult',
    # Annotation
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
    # Interactive
    'interactive_umap',
    'interactive_embedding',
    'interactive_3d',
    'interactive_gene_expression',
    'interactive_comparison',
    'interactive_cluster_composition',
    'interactive_violin',
    'create_dashboard',
]
