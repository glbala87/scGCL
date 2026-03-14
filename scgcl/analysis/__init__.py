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
]
