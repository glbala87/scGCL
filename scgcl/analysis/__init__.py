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

from .differential import (
    differential_abundance,
    plot_differential_abundance,
    abundance_barplot,
    DifferentialAbundanceResult
)

from .cell_cycle import (
    score_cell_cycle,
    score_cell_cycle_adata,
    score_genes,
    plot_cell_cycle,
    regress_cell_cycle,
    CellCycleResult,
    S_PHASE_GENES,
    G2M_PHASE_GENES
)

from .doublet import (
    detect_doublets,
    detect_doublets_scrublet,
    detect_doublets_adata,
    plot_doublet_scores,
    filter_doublets,
    DoubletResult
)

from .qc import (
    compute_cluster_qc,
    compute_cluster_purity,
    compute_batch_mixing,
    plot_cluster_qc,
    plot_batch_distribution,
    plot_batch_umap,
    batch_effect_test,
    ClusterQCResult
)

from .report import (
    HTMLReportGenerator,
    generate_clustering_report,
    generate_comparison_report,
    ReportSection
)

from .trajectory import (
    diffusion_pseudotime,
    principal_curve,
    slingshot,
    paga,
    infer_trajectory,
    plot_trajectory,
    plot_pseudotime_heatmap,
    find_trajectory_genes,
    TrajectoryResult
)

from .batch_integration import (
    harmony,
    mnn_correct,
    combat,
    regress_batch,
    integrate,
    compute_lisi,
    plot_integration,
    IntegrationResult
)

from .differential_expression import (
    differential_expression,
    pairwise_de,
    one_vs_rest_de,
    de_between_conditions,
    plot_volcano,
    plot_ma,
    plot_de_heatmap,
    compare_de_methods,
    DEResult
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
    # Differential abundance
    'differential_abundance',
    'plot_differential_abundance',
    'abundance_barplot',
    'DifferentialAbundanceResult',
    # Cell cycle
    'score_cell_cycle',
    'score_cell_cycle_adata',
    'score_genes',
    'plot_cell_cycle',
    'regress_cell_cycle',
    'CellCycleResult',
    'S_PHASE_GENES',
    'G2M_PHASE_GENES',
    # Doublet detection
    'detect_doublets',
    'detect_doublets_scrublet',
    'detect_doublets_adata',
    'plot_doublet_scores',
    'filter_doublets',
    'DoubletResult',
    # QC and batch
    'compute_cluster_qc',
    'compute_cluster_purity',
    'compute_batch_mixing',
    'plot_cluster_qc',
    'plot_batch_distribution',
    'plot_batch_umap',
    'batch_effect_test',
    'ClusterQCResult',
    # Report
    'HTMLReportGenerator',
    'generate_clustering_report',
    'generate_comparison_report',
    'ReportSection',
    # Trajectory
    'diffusion_pseudotime',
    'principal_curve',
    'slingshot',
    'paga',
    'infer_trajectory',
    'plot_trajectory',
    'plot_pseudotime_heatmap',
    'find_trajectory_genes',
    'TrajectoryResult',
    # Batch integration
    'harmony',
    'mnn_correct',
    'combat',
    'regress_batch',
    'integrate',
    'compute_lisi',
    'plot_integration',
    'IntegrationResult',
    # Differential expression
    'differential_expression',
    'pairwise_de',
    'one_vs_rest_de',
    'de_between_conditions',
    'plot_volcano',
    'plot_ma',
    'plot_de_heatmap',
    'compare_de_methods',
    'DEResult',
]
