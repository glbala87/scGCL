"""Cell cycle scoring for single-cell data."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import warnings


# Default cell cycle genes (from Tirosh et al., 2016)
S_PHASE_GENES = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG',
    'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP',
    'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76',
    'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51',
    'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
    'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
]

G2M_PHASE_GENES = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A',
    'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF',
    'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB',
    'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP',
    'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1',
    'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR',
    'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF',
    'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
]


@dataclass
class CellCycleResult:
    """Container for cell cycle scoring results."""

    s_scores: np.ndarray
    g2m_scores: np.ndarray
    phase: np.ndarray
    s_genes_found: List[str]
    g2m_genes_found: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'S_score': self.s_scores,
            'G2M_score': self.g2m_scores,
            'phase': self.phase
        })

    def summary(self) -> Dict:
        """Get summary statistics."""
        unique, counts = np.unique(self.phase, return_counts=True)
        phase_counts = dict(zip(unique, counts))
        return {
            'n_cells': len(self.phase),
            'phase_counts': phase_counts,
            'phase_proportions': {k: v / len(self.phase) for k, v in phase_counts.items()},
            's_genes_found': len(self.s_genes_found),
            'g2m_genes_found': len(self.g2m_genes_found)
        }


def score_genes(
    expression: np.ndarray,
    gene_names: np.ndarray,
    gene_set: List[str],
    ctrl_size: int = 50,
    n_bins: int = 25
) -> Tuple[np.ndarray, List[str]]:
    """
    Score cells for expression of a gene set.

    Uses the method from Tirosh et al. (2016): average expression of genes
    in the set minus average expression of control genes with similar
    expression levels.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    gene_names : np.ndarray
        Gene names
    gene_set : List[str]
        Genes to score
    ctrl_size : int
        Number of control genes per gene in set
    n_bins : int
        Number of expression bins

    Returns
    -------
    scores : np.ndarray
        Score for each cell
    genes_found : List[str]
        Genes from the set found in the data
    """
    gene_names = np.array([g.upper() for g in gene_names])
    gene_set_upper = [g.upper() for g in gene_set]

    # Find genes in the data
    gene_mask = np.isin(gene_names, gene_set_upper)
    genes_found = [gene_names[i] for i in range(len(gene_names)) if gene_mask[i]]

    if len(genes_found) == 0:
        return np.zeros(expression.shape[0]), []

    # Get mean expression per gene
    mean_expr = np.mean(expression, axis=0)

    # Bin genes by expression
    gene_bins = pd.cut(mean_expr, bins=n_bins, labels=False)

    # Get expression of gene set genes
    gene_set_expr = expression[:, gene_mask]
    gene_set_mean = np.mean(gene_set_expr, axis=1)

    # Get control genes for each gene in set
    gene_set_indices = np.where(gene_mask)[0]
    ctrl_scores = []

    for gene_idx in gene_set_indices:
        gene_bin = gene_bins[gene_idx]

        # Find genes in same bin (excluding gene set genes)
        same_bin = (gene_bins == gene_bin) & ~gene_mask
        same_bin_indices = np.where(same_bin)[0]

        if len(same_bin_indices) < ctrl_size:
            # Use all available genes in this bin
            ctrl_genes = same_bin_indices
        else:
            # Random sample
            ctrl_genes = np.random.choice(same_bin_indices, ctrl_size, replace=False)

        if len(ctrl_genes) > 0:
            ctrl_scores.append(np.mean(expression[:, ctrl_genes], axis=1))

    if len(ctrl_scores) > 0:
        ctrl_mean = np.mean(ctrl_scores, axis=0)
        scores = gene_set_mean - ctrl_mean
    else:
        scores = gene_set_mean

    return scores, genes_found


def score_cell_cycle(
    expression: np.ndarray,
    gene_names: np.ndarray,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    ctrl_size: int = 50,
    n_bins: int = 25
) -> CellCycleResult:
    """
    Score cells for cell cycle phase.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes), log-normalized
    gene_names : np.ndarray
        Gene names
    s_genes : List[str], optional
        S phase genes (default: Tirosh et al. 2016)
    g2m_genes : List[str], optional
        G2/M phase genes (default: Tirosh et al. 2016)
    ctrl_size : int
        Number of control genes per scored gene
    n_bins : int
        Number of expression bins

    Returns
    -------
    CellCycleResult
        Cell cycle scoring results
    """
    if s_genes is None:
        s_genes = S_PHASE_GENES
    if g2m_genes is None:
        g2m_genes = G2M_PHASE_GENES

    # Score S and G2M phases
    s_scores, s_found = score_genes(
        expression, gene_names, s_genes, ctrl_size, n_bins
    )
    g2m_scores, g2m_found = score_genes(
        expression, gene_names, g2m_genes, ctrl_size, n_bins
    )

    if len(s_found) == 0 and len(g2m_found) == 0:
        warnings.warn("No cell cycle genes found in the data")

    # Assign phases - use dtype to ensure G2M is not truncated
    phase = np.array(['G1'] * len(s_scores), dtype='<U3')

    # S phase: S score > 0 and S score > G2M score
    s_mask = (s_scores > 0) & (s_scores > g2m_scores)
    phase[s_mask] = 'S'

    # G2M phase: G2M score > 0 and G2M score > S score
    g2m_mask = (g2m_scores > 0) & (g2m_scores > s_scores)
    phase[g2m_mask] = 'G2M'

    return CellCycleResult(
        s_scores=s_scores,
        g2m_scores=g2m_scores,
        phase=phase,
        s_genes_found=s_found,
        g2m_genes_found=g2m_found
    )


def score_cell_cycle_adata(
    adata,
    s_genes: Optional[List[str]] = None,
    g2m_genes: Optional[List[str]] = None,
    use_raw: bool = False,
    copy: bool = False
):
    """
    Score cell cycle for AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    s_genes : List[str], optional
        S phase genes
    g2m_genes : List[str], optional
        G2/M phase genes
    use_raw : bool
        Use raw counts (will be log-normalized)
    copy : bool
        Return a copy

    Returns
    -------
    adata or None
        If copy=True, returns modified AnnData
    """
    if copy:
        adata = adata.copy()

    if use_raw and adata.raw is not None:
        X = adata.raw.X
        gene_names = np.array(adata.raw.var_names)
    else:
        X = adata.X
        gene_names = np.array(adata.var_names)

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    result = score_cell_cycle(
        X, gene_names,
        s_genes=s_genes,
        g2m_genes=g2m_genes
    )

    # Add to adata
    adata.obs['S_score'] = result.s_scores
    adata.obs['G2M_score'] = result.g2m_scores
    adata.obs['phase'] = result.phase

    adata.uns['cell_cycle'] = {
        's_genes_found': result.s_genes_found,
        'g2m_genes_found': result.g2m_genes_found
    }

    if copy:
        return adata


def plot_cell_cycle(
    result: Union[CellCycleResult, 'AnnData'],
    embedding: Optional[np.ndarray] = None,
    embedding_key: str = 'X_umap',
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot cell cycle results.

    Parameters
    ----------
    result : CellCycleResult or AnnData
        Cell cycle results
    embedding : np.ndarray, optional
        2D embedding coordinates
    embedding_key : str
        Key for embedding in adata.obsm
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

    # Extract data
    if hasattr(result, 's_scores'):
        s_scores = result.s_scores
        g2m_scores = result.g2m_scores
        phase = result.phase
    else:
        # AnnData
        s_scores = result.obs['S_score'].values
        g2m_scores = result.obs['G2M_score'].values
        phase = result.obs['phase'].values
        if embedding is None and embedding_key in result.obsm:
            embedding = result.obsm[embedding_key]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Phase distribution
    ax = axes[0]
    unique, counts = np.unique(phase, return_counts=True)
    colors = {'G1': '#1f77b4', 'S': '#ff7f0e', 'G2M': '#2ca02c'}
    ax.bar(unique, counts, color=[colors.get(p, 'gray') for p in unique])
    ax.set_xlabel('Phase')
    ax.set_ylabel('Number of cells')
    ax.set_title('Cell Cycle Phase Distribution')

    # S vs G2M scores
    ax = axes[1]
    phase_colors = [colors.get(p, 'gray') for p in phase]
    ax.scatter(s_scores, g2m_scores, c=phase_colors, alpha=0.5, s=10)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('S Score')
    ax.set_ylabel('G2M Score')
    ax.set_title('S vs G2M Scores')

    # Embedding colored by phase
    ax = axes[2]
    if embedding is not None:
        for p in ['G1', 'S', 'G2M']:
            mask = phase == p
            if np.any(mask):
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=colors.get(p), label=p, alpha=0.5, s=10
                )
        ax.legend()
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Cell Cycle Phase')
    else:
        ax.text(0.5, 0.5, 'No embedding provided',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Embedding not available')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def regress_cell_cycle(
    expression: np.ndarray,
    s_scores: np.ndarray,
    g2m_scores: np.ndarray,
    regress_s: bool = True,
    regress_g2m: bool = True,
    regress_difference: bool = False
) -> np.ndarray:
    """
    Regress out cell cycle effects from expression data.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    s_scores : np.ndarray
        S phase scores
    g2m_scores : np.ndarray
        G2M phase scores
    regress_s : bool
        Regress S score
    regress_g2m : bool
        Regress G2M score
    regress_difference : bool
        Regress difference only (preserves cycling vs non-cycling)

    Returns
    -------
    np.ndarray
        Expression with cell cycle effects regressed out
    """
    from scipy import linalg

    n_cells, n_genes = expression.shape

    # Build design matrix
    design = [np.ones(n_cells)]

    if regress_difference:
        # Only regress the difference
        design.append(s_scores - g2m_scores)
    else:
        if regress_s:
            design.append(s_scores)
        if regress_g2m:
            design.append(g2m_scores)

    design = np.column_stack(design)

    # Fit and remove effects
    beta = linalg.lstsq(design, expression)[0]

    # Remove effects (keep intercept)
    corrected = expression - design[:, 1:] @ beta[1:]

    return corrected
