"""Gene set enrichment analysis for cluster marker genes."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class EnrichmentResult:
    """Container for enrichment analysis results."""

    term: str
    cluster: int
    pvalue: float
    adjusted_pvalue: float
    odds_ratio: float
    overlap_genes: List[str]
    overlap_count: int
    term_size: int
    query_size: int
    source: str  # GO, KEGG, etc.

    def to_dict(self) -> dict:
        return {
            'term': self.term,
            'cluster': self.cluster,
            'pvalue': self.pvalue,
            'adjusted_pvalue': self.adjusted_pvalue,
            'odds_ratio': self.odds_ratio,
            'overlap_genes': ','.join(self.overlap_genes),
            'overlap_count': self.overlap_count,
            'term_size': self.term_size,
            'query_size': self.query_size,
            'source': self.source
        }


def _hypergeometric_test(
    overlap: int,
    query_size: int,
    term_size: int,
    background_size: int
) -> float:
    """Compute hypergeometric p-value."""
    from scipy.stats import hypergeom

    # P(X >= overlap)
    pval = hypergeom.sf(overlap - 1, background_size, term_size, query_size)
    return pval


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        adjusted[i] = min(cummin, sorted_pvals[i] * n / (i + 1))
        cummin = adjusted[i]

    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return np.clip(result, 0, 1)


def load_gene_sets(
    source: str = 'go_bp',
    organism: str = 'human',
    custom_gmt: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Load gene sets from various sources.

    Parameters
    ----------
    source : str
        Gene set source: 'go_bp', 'go_mf', 'go_cc', 'kegg', 'reactome', 'custom'
    organism : str
        Organism: 'human', 'mouse'
    custom_gmt : str, optional
        Path to custom GMT file

    Returns
    -------
    dict
        Dictionary mapping term names to gene lists
    """
    if custom_gmt is not None:
        return load_gmt(custom_gmt)

    # Try to use gseapy if available
    try:
        import gseapy as gp

        if source == 'go_bp':
            lib = 'GO_Biological_Process_2021'
        elif source == 'go_mf':
            lib = 'GO_Molecular_Function_2021'
        elif source == 'go_cc':
            lib = 'GO_Cellular_Component_2021'
        elif source == 'kegg':
            lib = 'KEGG_2021_Human' if organism == 'human' else 'KEGG_2019_Mouse'
        elif source == 'reactome':
            lib = 'Reactome_2022'
        else:
            lib = source  # Use as-is

        gene_sets = gp.get_library(lib)
        return gene_sets

    except ImportError:
        warnings.warn(
            "gseapy not installed. Using built-in minimal gene sets. "
            "For full functionality, install gseapy: pip install gseapy"
        )
        return _get_builtin_gene_sets(source)


def _get_builtin_gene_sets(source: str) -> Dict[str, List[str]]:
    """Return minimal built-in gene sets for testing."""
    # Minimal example gene sets
    return {
        'Cell Cycle': ['CDK1', 'CDK2', 'CCNA2', 'CCNB1', 'CCNE1', 'MCM2', 'MCM3'],
        'Apoptosis': ['BCL2', 'BAX', 'CASP3', 'CASP9', 'TP53', 'FAS', 'FASLG'],
        'Immune Response': ['IL2', 'IL6', 'TNF', 'IFNG', 'CD4', 'CD8A', 'CD19'],
        'Metabolism': ['GAPDH', 'PKM', 'LDHA', 'HK1', 'HK2', 'PFKM', 'ENO1'],
        'Ribosome': ['RPL5', 'RPL11', 'RPS6', 'RPS3', 'RPL3', 'RPL4', 'RPS2'],
    }


def load_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """
    Load gene sets from GMT file.

    Parameters
    ----------
    gmt_path : str
        Path to GMT file

    Returns
    -------
    dict
        Gene sets dictionary
    """
    gene_sets = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                name = parts[0]
                genes = parts[2:]  # Skip description
                gene_sets[name] = genes
    return gene_sets


def enrich(
    genes: List[str],
    gene_sets: Dict[str, List[str]],
    background: Optional[List[str]] = None,
    min_overlap: int = 3,
    max_term_size: int = 500,
    pval_cutoff: float = 0.05
) -> List[EnrichmentResult]:
    """
    Perform enrichment analysis for a gene list.

    Parameters
    ----------
    genes : list
        Query gene list
    gene_sets : dict
        Gene sets dictionary
    background : list, optional
        Background gene list
    min_overlap : int
        Minimum overlap required
    max_term_size : int
        Maximum term size to consider
    pval_cutoff : float
        P-value cutoff

    Returns
    -------
    list of EnrichmentResult
        Enrichment results
    """
    query_genes = set(g.upper() for g in genes)
    query_size = len(query_genes)

    if background is None:
        # Use all genes in gene sets as background
        all_genes = set()
        for gs in gene_sets.values():
            all_genes.update(g.upper() for g in gs)
        background = all_genes
    else:
        background = set(g.upper() for g in background)

    background_size = len(background)

    results = []
    pvalues = []

    for term, term_genes in gene_sets.items():
        term_genes_upper = set(g.upper() for g in term_genes)
        term_size = len(term_genes_upper & background)

        if term_size > max_term_size or term_size < min_overlap:
            continue

        overlap = query_genes & term_genes_upper & background
        overlap_count = len(overlap)

        if overlap_count < min_overlap:
            continue

        # Hypergeometric test
        pval = _hypergeometric_test(overlap_count, query_size, term_size, background_size)
        pvalues.append(pval)

        # Odds ratio
        a = overlap_count
        b = query_size - overlap_count
        c = term_size - overlap_count
        d = background_size - query_size - term_size + overlap_count
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')

        results.append({
            'term': term,
            'pvalue': pval,
            'odds_ratio': odds_ratio,
            'overlap_genes': list(overlap),
            'overlap_count': overlap_count,
            'term_size': term_size,
            'query_size': query_size
        })

    if not results:
        return []

    # FDR correction
    pvalues = np.array(pvalues)
    adjusted_pvalues = _benjamini_hochberg(pvalues)

    # Filter and create results
    enrichment_results = []
    for i, r in enumerate(results):
        if adjusted_pvalues[i] <= pval_cutoff:
            enrichment_results.append(EnrichmentResult(
                term=r['term'],
                cluster=-1,  # Will be set by caller
                pvalue=r['pvalue'],
                adjusted_pvalue=adjusted_pvalues[i],
                odds_ratio=r['odds_ratio'],
                overlap_genes=r['overlap_genes'],
                overlap_count=r['overlap_count'],
                term_size=r['term_size'],
                query_size=r['query_size'],
                source=''
            ))

    # Sort by adjusted p-value
    enrichment_results.sort(key=lambda x: x.adjusted_pvalue)

    return enrichment_results


def cluster_enrichment(
    markers_df: pd.DataFrame,
    source: str = 'go_bp',
    organism: str = 'human',
    background: Optional[List[str]] = None,
    min_overlap: int = 3,
    pval_cutoff: float = 0.05,
    top_n: int = 10,
    custom_gmt: Optional[str] = None
) -> pd.DataFrame:
    """
    Perform gene set enrichment for each cluster's marker genes.

    Parameters
    ----------
    markers_df : pd.DataFrame
        Marker genes DataFrame with 'cluster' and 'gene' columns
    source : str
        Gene set source
    organism : str
        Organism
    background : list, optional
        Background gene list
    min_overlap : int
        Minimum overlap
    pval_cutoff : float
        Adjusted p-value cutoff
    top_n : int
        Top N terms per cluster
    custom_gmt : str, optional
        Path to custom GMT file

    Returns
    -------
    pd.DataFrame
        Enrichment results for all clusters
    """
    # Load gene sets
    gene_sets = load_gene_sets(source, organism, custom_gmt)

    all_results = []

    for cluster in sorted(markers_df['cluster'].unique()):
        cluster_genes = markers_df[markers_df['cluster'] == cluster]['gene'].tolist()

        results = enrich(
            cluster_genes,
            gene_sets,
            background=background,
            min_overlap=min_overlap,
            pval_cutoff=pval_cutoff
        )

        for r in results[:top_n]:
            r.cluster = cluster
            r.source = source
            all_results.append(r.to_dict())

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    return df.sort_values(['cluster', 'adjusted_pvalue'])


def plot_enrichment(
    enrichment_df: pd.DataFrame,
    top_n: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot enrichment results.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Enrichment results
    top_n : int
        Top N terms per cluster
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    if enrichment_df.empty:
        print("No enrichment results to plot")
        return

    clusters = sorted(enrichment_df['cluster'].unique())
    n_clusters = len(clusters)

    fig, axes = plt.subplots(1, n_clusters, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        cluster_df = enrichment_df[enrichment_df['cluster'] == cluster].head(top_n)

        if cluster_df.empty:
            ax.text(0.5, 0.5, 'No significant\nenrichment',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Cluster {cluster}')
            continue

        # Truncate long term names
        terms = [t[:40] + '...' if len(t) > 40 else t for t in cluster_df['term']]
        pvals = -np.log10(cluster_df['adjusted_pvalue'].values)

        y_pos = np.arange(len(terms))
        ax.barh(y_pos, pvals, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms)
        ax.set_xlabel('-log10(adj. p-value)')
        ax.set_title(f'Cluster {cluster}')
        ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def quick_enrich(
    markers_df: pd.DataFrame,
    source: str = 'go_bp',
    top_n: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Quick enrichment analysis with default settings.

    Parameters
    ----------
    markers_df : pd.DataFrame
        Marker genes with 'cluster' and 'gene' columns
    source : str
        Gene set source
    top_n : int
        Top terms per cluster
    verbose : bool
        Print results

    Returns
    -------
    pd.DataFrame
        Enrichment results
    """
    results = cluster_enrichment(markers_df, source=source, top_n=top_n)

    if verbose and not results.empty:
        print(f"\n{'='*60}")
        print(f"Gene Set Enrichment Results ({source})")
        print(f"{'='*60}")

        for cluster in sorted(results['cluster'].unique()):
            print(f"\nCluster {cluster}:")
            cluster_df = results[results['cluster'] == cluster].head(top_n)
            for _, row in cluster_df.iterrows():
                term = row['term'][:50] + '...' if len(row['term']) > 50 else row['term']
                print(f"  {term}: p={row['adjusted_pvalue']:.2e} ({row['overlap_count']} genes)")

    return results
