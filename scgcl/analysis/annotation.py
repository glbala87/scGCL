"""Cell type annotation using marker gene databases."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field
import warnings


# Built-in marker databases for common cell types
PBMC_MARKERS = {
    'CD4+ T cells': ['CD4', 'IL7R', 'CCR7', 'TCF7', 'LEF1', 'CD3D', 'CD3E'],
    'CD8+ T cells': ['CD8A', 'CD8B', 'GZMK', 'GZMA', 'CD3D', 'CD3E', 'CCL5'],
    'Naive T cells': ['CCR7', 'SELL', 'LEF1', 'TCF7', 'CD27', 'CD28'],
    'Memory T cells': ['S100A4', 'IL7R', 'CD44', 'GZMK'],
    'Regulatory T cells': ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'IKZF2'],
    'NK cells': ['NKG7', 'GNLY', 'KLRD1', 'KLRF1', 'NCAM1', 'FCGR3A', 'PRF1'],
    'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'PAX5', 'BANK1'],
    'Plasma cells': ['MZB1', 'SDC1', 'IGHG1', 'IGHG2', 'JCHAIN', 'XBP1'],
    'CD14+ Monocytes': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'FCN1'],
    'CD16+ Monocytes': ['FCGR3A', 'MS4A7', 'LST1', 'CDKN1C'],
    'Dendritic cells': ['FCER1A', 'CST3', 'CLEC10A', 'CD1C', 'ITGAX'],
    'Plasmacytoid DC': ['IL3RA', 'CLEC4C', 'NRP1', 'TCF4', 'IRF7'],
    'Megakaryocytes': ['PPBP', 'PF4', 'GP9', 'ITGA2B'],
    'Platelets': ['PPBP', 'PF4', 'TUBB1', 'GP1BA'],
    'Erythrocytes': ['HBA1', 'HBA2', 'HBB', 'GYPA'],
}

BRAIN_MARKERS = {
    'Excitatory neurons': ['SLC17A7', 'CAMK2A', 'SATB2', 'CUX2', 'RORB'],
    'Inhibitory neurons': ['GAD1', 'GAD2', 'SLC32A1', 'PVALB', 'SST', 'VIP'],
    'Astrocytes': ['GFAP', 'AQP4', 'SLC1A2', 'SLC1A3', 'ALDH1L1'],
    'Oligodendrocytes': ['MBP', 'MOG', 'MOBP', 'PLP1', 'MAG'],
    'OPCs': ['PDGFRA', 'CSPG4', 'OLIG1', 'OLIG2'],
    'Microglia': ['CX3CR1', 'P2RY12', 'TMEM119', 'AIF1', 'CSF1R'],
    'Endothelial': ['CLDN5', 'FLT1', 'PECAM1', 'VWF'],
    'Pericytes': ['PDGFRB', 'RGS5', 'KCNJ8', 'ABCC9'],
}

IMMUNE_MARKERS = {
    'T cells': ['CD3D', 'CD3E', 'CD3G', 'TRAC', 'TRBC1', 'TRBC2'],
    'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'BANK1'],
    'NK cells': ['NKG7', 'GNLY', 'KLRD1', 'NCAM1', 'FCGR3A'],
    'Monocytes': ['CD14', 'FCGR3A', 'LYZ', 'S100A8', 'S100A9'],
    'Macrophages': ['CD68', 'CD163', 'MRC1', 'MSR1', 'MARCO'],
    'Dendritic cells': ['ITGAX', 'CD1C', 'CLEC9A', 'FCER1A'],
    'Neutrophils': ['FCGR3B', 'CSF3R', 'CXCR2', 'S100A8', 'S100A9'],
    'Mast cells': ['KIT', 'TPSAB1', 'CPA3', 'MS4A2'],
    'Eosinophils': ['CCR3', 'SIGLEC8', 'IL5RA', 'PRG2'],
    'Basophils': ['GATA2', 'CLC', 'HDC', 'IL4'],
}

TUMOR_MARKERS = {
    'Epithelial': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1'],
    'Fibroblasts': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP'],
    'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'KDR', 'FLT1'],
    'Smooth muscle': ['ACTA2', 'MYH11', 'TAGLN', 'CNN1'],
    'Pericytes': ['RGS5', 'PDGFRB', 'KCNJ8', 'ABCC9'],
    'T cells': ['CD3D', 'CD3E', 'CD4', 'CD8A', 'CD8B'],
    'B cells': ['CD19', 'MS4A1', 'CD79A'],
    'Myeloid': ['CD68', 'CD14', 'ITGAM', 'LYZ'],
    'NK cells': ['NKG7', 'GNLY', 'KLRD1'],
}

# Combine all markers
ALL_MARKERS = {
    'pbmc': PBMC_MARKERS,
    'brain': BRAIN_MARKERS,
    'immune': IMMUNE_MARKERS,
    'tumor': TUMOR_MARKERS,
}


@dataclass
class AnnotationResult:
    """Container for cell type annotation results."""

    cluster_annotations: Dict[int, str]  # cluster -> cell type
    cluster_scores: Dict[int, Dict[str, float]]  # cluster -> {cell_type: score}
    cluster_confidence: Dict[int, float]  # cluster -> confidence (0-1)
    cell_types: np.ndarray  # Per-cell annotations
    cell_confidence: np.ndarray  # Per-cell confidence
    marker_expression: pd.DataFrame  # Marker expression per cluster

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "Cell Type Annotation Results",
            "=" * 60,
            "",
            "Cluster Annotations:",
        ]
        for cluster in sorted(self.cluster_annotations.keys()):
            cell_type = self.cluster_annotations[cluster]
            confidence = self.cluster_confidence[cluster]
            lines.append(f"  Cluster {cluster}: {cell_type} (confidence: {confidence:.2f})")

        lines.extend([
            "",
            "Cell Type Distribution:",
        ])
        unique, counts = np.unique(self.cell_types, return_counts=True)
        for ct, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            pct = 100 * count / len(self.cell_types)
            lines.append(f"  {ct}: {count} cells ({pct:.1f}%)")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        rows = []
        for cluster in sorted(self.cluster_annotations.keys()):
            rows.append({
                'cluster': cluster,
                'cell_type': self.cluster_annotations[cluster],
                'confidence': self.cluster_confidence[cluster],
                **{f'score_{ct}': self.cluster_scores[cluster].get(ct, 0)
                   for ct in set().union(*[set(s.keys()) for s in self.cluster_scores.values()])}
            })
        return pd.DataFrame(rows)


def get_marker_database(
    tissue: str = 'pbmc',
    custom_markers: Optional[Dict[str, List[str]]] = None
) -> Dict[str, List[str]]:
    """
    Get marker gene database for annotation.

    Parameters
    ----------
    tissue : str
        Tissue type: 'pbmc', 'brain', 'immune', 'tumor', or 'all'
    custom_markers : dict, optional
        Custom marker dictionary to use instead of built-in

    Returns
    -------
    dict
        Marker dictionary {cell_type: [genes]}
    """
    if custom_markers is not None:
        return custom_markers

    if tissue == 'all':
        # Combine all databases
        combined = {}
        for db in ALL_MARKERS.values():
            combined.update(db)
        return combined

    if tissue not in ALL_MARKERS:
        warnings.warn(f"Unknown tissue '{tissue}', using 'pbmc'")
        tissue = 'pbmc'

    return ALL_MARKERS[tissue]


def score_cell_types(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: List[str],
    markers: Dict[str, List[str]],
    method: str = 'mean'
) -> Tuple[Dict[int, Dict[str, float]], pd.DataFrame]:
    """
    Score each cluster for each cell type based on marker expression.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (cells x genes)
    labels : np.ndarray
        Cluster labels
    gene_names : list
        Gene names
    markers : dict
        Marker dictionary {cell_type: [genes]}
    method : str
        Scoring method: 'mean', 'percent', 'zscore'

    Returns
    -------
    scores : dict
        {cluster: {cell_type: score}}
    marker_expr : pd.DataFrame
        Marker expression per cluster
    """
    gene_names_upper = [g.upper() for g in gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_names_upper)}

    unique_clusters = np.unique(labels)

    # Compute cluster mean expression
    cluster_means = {}
    for cluster in unique_clusters:
        mask = labels == cluster
        cluster_means[cluster] = X[mask].mean(axis=0)

    # Overall mean for z-score
    overall_mean = X.mean(axis=0)
    overall_std = X.std(axis=0) + 1e-10

    # Score each cluster for each cell type
    scores = {c: {} for c in unique_clusters}
    marker_data = []

    for cell_type, marker_genes in markers.items():
        # Find markers present in data
        present_markers = []
        marker_indices = []
        for gene in marker_genes:
            gene_upper = gene.upper()
            if gene_upper in gene_to_idx:
                present_markers.append(gene)
                marker_indices.append(gene_to_idx[gene_upper])

        if not present_markers:
            continue

        for cluster in unique_clusters:
            cluster_expr = cluster_means[cluster]
            marker_expr = cluster_expr[marker_indices]

            if method == 'mean':
                # Mean expression of markers
                score = marker_expr.mean()
            elif method == 'percent':
                # Percent of markers expressed above threshold
                threshold = overall_mean[marker_indices] + 0.5 * overall_std[marker_indices]
                score = (marker_expr > threshold).mean()
            elif method == 'zscore':
                # Z-score of marker expression
                z_scores = (marker_expr - overall_mean[marker_indices]) / overall_std[marker_indices]
                score = z_scores.mean()
            else:
                score = marker_expr.mean()

            scores[cluster][cell_type] = float(score)

            # Record marker expression
            for i, gene in enumerate(present_markers):
                marker_data.append({
                    'cluster': cluster,
                    'cell_type': cell_type,
                    'gene': gene,
                    'expression': float(marker_expr[i]),
                    'z_score': float((marker_expr[i] - overall_mean[marker_indices[i]]) /
                                    overall_std[marker_indices[i]])
                })

    marker_df = pd.DataFrame(marker_data) if marker_data else pd.DataFrame()

    return scores, marker_df


def annotate_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: List[str],
    tissue: str = 'pbmc',
    custom_markers: Optional[Dict[str, List[str]]] = None,
    method: str = 'mean',
    min_confidence: float = 0.1,
    verbose: bool = True
) -> AnnotationResult:
    """
    Annotate clusters with cell type labels.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (cells x genes)
    labels : np.ndarray
        Cluster labels
    gene_names : list
        Gene names
    tissue : str
        Tissue type for marker selection: 'pbmc', 'brain', 'immune', 'tumor', 'all'
    custom_markers : dict, optional
        Custom marker dictionary {cell_type: [genes]}
    method : str
        Scoring method: 'mean', 'percent', 'zscore'
    min_confidence : float
        Minimum confidence threshold for annotation
    verbose : bool
        Print results

    Returns
    -------
    AnnotationResult
        Annotation results
    """
    # Get marker database
    markers = get_marker_database(tissue, custom_markers)

    if verbose:
        print(f"Using {len(markers)} cell type markers from '{tissue}' database")

    # Score clusters
    scores, marker_expr = score_cell_types(X, labels, gene_names, markers, method)

    # Assign cell types
    cluster_annotations = {}
    cluster_confidence = {}

    for cluster in sorted(scores.keys()):
        cluster_scores = scores[cluster]

        if not cluster_scores:
            cluster_annotations[cluster] = 'Unknown'
            cluster_confidence[cluster] = 0.0
            continue

        # Find best match
        best_type = max(cluster_scores.keys(), key=lambda x: cluster_scores[x])
        best_score = cluster_scores[best_type]

        # Compute confidence (normalized score)
        all_scores = list(cluster_scores.values())
        if max(all_scores) > 0:
            confidence = best_score / (sum(all_scores) + 1e-10)
            # Scale confidence
            confidence = min(1.0, confidence * len(all_scores))
        else:
            confidence = 0.0

        if confidence < min_confidence:
            cluster_annotations[cluster] = 'Unknown'
            cluster_confidence[cluster] = confidence
        else:
            cluster_annotations[cluster] = best_type
            cluster_confidence[cluster] = confidence

    # Create per-cell annotations
    cell_types = np.array([cluster_annotations[l] for l in labels])
    cell_confidence = np.array([cluster_confidence[l] for l in labels])

    result = AnnotationResult(
        cluster_annotations=cluster_annotations,
        cluster_scores=scores,
        cluster_confidence=cluster_confidence,
        cell_types=cell_types,
        cell_confidence=cell_confidence,
        marker_expression=marker_expr
    )

    if verbose:
        print(result.summary())

    return result


def annotate_adata(
    adata,
    key: str = 'scgcl_clusters',
    tissue: str = 'pbmc',
    custom_markers: Optional[Dict[str, List[str]]] = None,
    method: str = 'mean',
    key_added: str = 'cell_type'
) -> AnnotationResult:
    """
    Annotate clusters in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    key : str
        Key in adata.obs containing cluster labels
    tissue : str
        Tissue type for markers
    custom_markers : dict, optional
        Custom marker dictionary
    method : str
        Scoring method
    key_added : str
        Key to add to adata.obs for cell types

    Returns
    -------
    AnnotationResult
        Annotation results
    """
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    labels = adata.obs[key].values
    if hasattr(labels, 'cat'):
        labels = labels.cat.codes.values
    labels = np.asarray(labels, dtype=int)

    gene_names = adata.var_names.tolist()

    result = annotate_clusters(
        X, labels, gene_names,
        tissue=tissue,
        custom_markers=custom_markers,
        method=method,
        verbose=False
    )

    # Add to adata
    adata.obs[key_added] = result.cell_types
    adata.obs[f'{key_added}_confidence'] = result.cell_confidence
    adata.uns['cell_type_annotation'] = {
        'cluster_annotations': result.cluster_annotations,
        'cluster_confidence': result.cluster_confidence,
    }

    return result


def plot_annotation(
    result: AnnotationResult,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot annotation results.

    Parameters
    ----------
    result : AnnotationResult
        Annotation results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cell type distribution
    ax = axes[0]
    unique, counts = np.unique(result.cell_types, return_counts=True)
    sorted_idx = np.argsort(-counts)
    ax.barh(range(len(unique)), counts[sorted_idx], color='steelblue')
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique[sorted_idx])
    ax.set_xlabel('Number of Cells')
    ax.set_title('Cell Type Distribution')
    ax.invert_yaxis()

    # Confidence per cluster
    ax = axes[1]
    clusters = sorted(result.cluster_annotations.keys())
    confidences = [result.cluster_confidence[c] for c in clusters]
    colors = ['green' if c > 0.5 else 'orange' if c > 0.2 else 'red' for c in confidences]
    ax.bar(clusters, confidences, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Confidence')
    ax.set_title('Annotation Confidence by Cluster')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_marker_heatmap(
    result: AnnotationResult,
    top_n: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot heatmap of marker gene expression.

    Parameters
    ----------
    result : AnnotationResult
        Annotation results
    top_n : int
        Top N markers per cell type to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return

    if result.marker_expression.empty:
        print("No marker expression data available")
        return

    # Pivot to get clusters x genes
    df = result.marker_expression.copy()

    # Get top markers by z-score for each cell type
    grouped = (df.groupby(['cell_type', 'gene'])['z_score']
               .mean()
               .reset_index())
    top_markers = {}
    for cell_type in grouped['cell_type'].unique():
        ct_df = grouped[grouped['cell_type'] == cell_type]
        top_genes = ct_df.nlargest(top_n, 'z_score')['gene'].tolist()
        top_markers[cell_type] = top_genes

    # Flatten marker list
    all_markers = []
    for markers in top_markers.values():
        all_markers.extend(markers)
    all_markers = list(dict.fromkeys(all_markers))  # Remove duplicates, preserve order

    # Create pivot table
    pivot_df = df[df['gene'].isin(all_markers)].pivot_table(
        index='cluster',
        columns='gene',
        values='z_score',
        aggfunc='mean'
    )

    if pivot_df.empty:
        print("No data for heatmap")
        return

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_df,
        cmap='RdBu_r',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Z-score'}
    )
    ax.set_title('Marker Gene Expression by Cluster')
    ax.set_xlabel('Marker Gene')
    ax.set_ylabel('Cluster')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def quick_annotate(
    X: np.ndarray,
    labels: np.ndarray,
    gene_names: List[str],
    tissue: str = 'pbmc'
) -> Dict[int, str]:
    """
    Quick annotation returning just cluster -> cell type mapping.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix
    labels : np.ndarray
        Cluster labels
    gene_names : list
        Gene names
    tissue : str
        Tissue type

    Returns
    -------
    dict
        {cluster: cell_type}
    """
    result = annotate_clusters(X, labels, gene_names, tissue=tissue, verbose=False)
    return result.cluster_annotations
