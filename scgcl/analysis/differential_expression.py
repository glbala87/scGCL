"""Differential expression analysis for single-cell data."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class DEResult:
    """Container for differential expression results."""

    results: pd.DataFrame
    group1: str
    group2: str
    method: str
    n_group1: int
    n_group2: int
    n_significant: int

    def significant(self, pval_cutoff: float = 0.05, fc_cutoff: float = 0.5) -> pd.DataFrame:
        """Get significant genes."""
        mask = (self.results['padj'] < pval_cutoff) & (abs(self.results['log2FC']) > fc_cutoff)
        return self.results[mask]

    def upregulated(self, pval_cutoff: float = 0.05, fc_cutoff: float = 0.5) -> pd.DataFrame:
        """Get upregulated genes (higher in group2)."""
        mask = (self.results['padj'] < pval_cutoff) & (self.results['log2FC'] > fc_cutoff)
        return self.results[mask]

    def downregulated(self, pval_cutoff: float = 0.05, fc_cutoff: float = 0.5) -> pd.DataFrame:
        """Get downregulated genes (lower in group2)."""
        mask = (self.results['padj'] < pval_cutoff) & (self.results['log2FC'] < -fc_cutoff)
        return self.results[mask]

    def summary(self) -> str:
        """Get summary string."""
        n_up = len(self.upregulated())
        n_down = len(self.downregulated())
        lines = [
            "Differential Expression Results",
            "=" * 40,
            f"Comparison: {self.group2} vs {self.group1}",
            f"Method: {self.method}",
            f"Cells in {self.group1}: {self.n_group1}",
            f"Cells in {self.group2}: {self.n_group2}",
            f"Total genes tested: {len(self.results)}",
            f"Significant genes (padj < 0.05, |log2FC| > 0.5): {self.n_significant}",
            f"  Upregulated: {n_up}",
            f"  Downregulated: {n_down}",
        ]
        return "\n".join(lines)


def differential_expression(
    expression: np.ndarray,
    groups: np.ndarray,
    gene_names: np.ndarray,
    group1: Optional[str] = None,
    group2: Optional[str] = None,
    method: str = 'wilcoxon',
    min_cells: int = 3,
    min_pct: float = 0.1,
    pseudocount: float = 1.0
) -> DEResult:
    """
    Perform differential expression analysis between two groups.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes)
    groups : np.ndarray
        Group labels for each cell
    gene_names : np.ndarray
        Gene names
    group1 : str, optional
        Reference group (default: first unique)
    group2 : str, optional
        Comparison group (default: second unique)
    method : str
        Test method: 'wilcoxon', 't-test', 'negbinom', 'logistic'
    min_cells : int
        Minimum cells expressing gene in either group
    min_pct : float
        Minimum fraction of cells expressing gene
    pseudocount : float
        Pseudocount for log fold change

    Returns
    -------
    DEResult
        Differential expression results
    """
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise ValueError("Need at least 2 groups for differential expression")

    if group1 is None:
        group1 = str(unique_groups[0])
    if group2 is None:
        group2 = str(unique_groups[1])

    # Get cells in each group
    mask1 = groups == group1
    mask2 = groups == group2

    expr1 = expression[mask1]
    expr2 = expression[mask2]

    n_group1 = expr1.shape[0]
    n_group2 = expr2.shape[0]
    n_genes = expression.shape[1]

    results = []

    for i in range(n_genes):
        gene = gene_names[i]
        x1 = expr1[:, i]
        x2 = expr2[:, i]

        # Filter: minimum expression
        n_expr1 = np.sum(x1 > 0)
        n_expr2 = np.sum(x2 > 0)

        if n_expr1 < min_cells and n_expr2 < min_cells:
            continue

        pct1 = n_expr1 / n_group1
        pct2 = n_expr2 / n_group2

        if pct1 < min_pct and pct2 < min_pct:
            continue

        # Mean expression
        mean1 = np.mean(x1)
        mean2 = np.mean(x2)

        # Log2 fold change
        log2fc = np.log2((mean2 + pseudocount) / (mean1 + pseudocount))

        # Statistical test
        if method == 'wilcoxon':
            try:
                stat, pval = stats.mannwhitneyu(x1, x2, alternative='two-sided')
            except Exception:
                pval = 1.0
                stat = 0

        elif method == 't-test':
            try:
                stat, pval = stats.ttest_ind(x1, x2, equal_var=False)
            except Exception:
                pval = 1.0
                stat = 0

        elif method == 'negbinom':
            # Simplified negative binomial test using Poisson approximation
            try:
                # Use Poisson exact test approximation
                lambda1 = mean1 * n_group1
                lambda2 = mean2 * n_group2
                total = lambda1 + lambda2
                if total > 0:
                    p_expected = n_group1 / (n_group1 + n_group2)
                    # Binomial test approximation
                    stat, pval = stats.binom_test(
                        int(lambda1), int(total), p_expected,
                        alternative='two-sided'
                    ) if hasattr(stats, 'binom_test') else (0, 1.0)
                else:
                    pval = 1.0
                    stat = 0
            except Exception:
                pval = 1.0
                stat = 0

        elif method == 'logistic':
            # Logistic regression test
            try:
                from scipy.optimize import minimize

                # Combine data
                X = np.concatenate([x1, x2]).reshape(-1, 1)
                y = np.array([0] * len(x1) + [1] * len(x2))

                # Standardize
                X = (X - X.mean()) / (X.std() + 1e-10)

                # Fit logistic regression
                def neg_log_likelihood(beta):
                    z = beta[0] + beta[1] * X.flatten()
                    p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                    ll = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
                    return -ll

                result = minimize(neg_log_likelihood, [0, 0], method='BFGS')
                beta = result.x

                # Wald test for coefficient
                hessian = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(2)
                if isinstance(hessian, np.ndarray):
                    se = np.sqrt(np.diag(hessian))[1]
                else:
                    se = 1.0
                z_stat = beta[1] / (se + 1e-10)
                pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                stat = z_stat

            except Exception:
                pval = 1.0
                stat = 0

        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            'gene': gene,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'log2FC': log2fc,
            'pct_group1': pct1,
            'pct_group2': pct2,
            'statistic': stat,
            'pvalue': pval
        })

    if not results:
        return DEResult(
            results=pd.DataFrame(),
            group1=str(group1),
            group2=str(group2),
            method=method,
            n_group1=n_group1,
            n_group2=n_group2,
            n_significant=0
        )

    df = pd.DataFrame(results)

    # Multiple testing correction
    df['padj'] = _benjamini_hochberg(df['pvalue'].values)

    # Sort by adjusted p-value
    df = df.sort_values('padj')

    # Count significant
    n_sig = np.sum((df['padj'] < 0.05) & (abs(df['log2FC']) > 0.5))

    # Rename columns with group names
    df = df.rename(columns={
        'mean_group1': f'mean_{group1}',
        'mean_group2': f'mean_{group2}',
        'pct_group1': f'pct_{group1}',
        'pct_group2': f'pct_{group2}'
    })

    return DEResult(
        results=df,
        group1=str(group1),
        group2=str(group2),
        method=method,
        n_group1=n_group1,
        n_group2=n_group2,
        n_significant=int(n_sig)
    )


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


def pairwise_de(
    expression: np.ndarray,
    groups: np.ndarray,
    gene_names: np.ndarray,
    method: str = 'wilcoxon',
    **kwargs
) -> Dict[str, DEResult]:
    """
    Perform pairwise differential expression for all group pairs.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    groups : np.ndarray
        Group labels
    gene_names : np.ndarray
        Gene names
    method : str
        Test method
    **kwargs
        Additional arguments to differential_expression

    Returns
    -------
    Dict[str, DEResult]
        Results for each pair
    """
    unique_groups = np.unique(groups)
    results = {}

    for i, g1 in enumerate(unique_groups):
        for g2 in unique_groups[i + 1:]:
            key = f"{g2}_vs_{g1}"
            results[key] = differential_expression(
                expression, groups, gene_names,
                group1=str(g1), group2=str(g2),
                method=method, **kwargs
            )

    return results


def one_vs_rest_de(
    expression: np.ndarray,
    groups: np.ndarray,
    gene_names: np.ndarray,
    method: str = 'wilcoxon',
    **kwargs
) -> Dict[str, DEResult]:
    """
    Perform one-vs-rest differential expression for each group.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    groups : np.ndarray
        Group labels
    gene_names : np.ndarray
        Gene names
    method : str
        Test method
    **kwargs
        Additional arguments to differential_expression

    Returns
    -------
    Dict[str, DEResult]
        Results for each group vs rest
    """
    unique_groups = np.unique(groups)
    results = {}

    for g in unique_groups:
        # Create binary grouping
        binary_groups = np.where(groups == g, str(g), 'rest')

        results[str(g)] = differential_expression(
            expression, binary_groups, gene_names,
            group1='rest', group2=str(g),
            method=method, **kwargs
        )

    return results


def de_between_conditions(
    expression: np.ndarray,
    condition: np.ndarray,
    cluster: np.ndarray,
    gene_names: np.ndarray,
    condition1: str,
    condition2: str,
    method: str = 'wilcoxon',
    **kwargs
) -> Dict[int, DEResult]:
    """
    Differential expression between conditions within each cluster.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    condition : np.ndarray
        Condition labels (e.g., 'control', 'treatment')
    cluster : np.ndarray
        Cluster labels
    gene_names : np.ndarray
        Gene names
    condition1 : str
        Reference condition
    condition2 : str
        Comparison condition
    method : str
        Test method
    **kwargs
        Additional arguments

    Returns
    -------
    Dict[int, DEResult]
        Results for each cluster
    """
    unique_clusters = np.unique(cluster)
    results = {}

    for c in unique_clusters:
        cluster_mask = cluster == c

        # Get cells in this cluster
        cluster_expr = expression[cluster_mask]
        cluster_cond = condition[cluster_mask]

        # Check if both conditions present
        if condition1 not in cluster_cond or condition2 not in cluster_cond:
            continue

        results[c] = differential_expression(
            cluster_expr, cluster_cond, gene_names,
            group1=condition1, group2=condition2,
            method=method, **kwargs
        )

    return results


def plot_volcano(
    de_result: DEResult,
    pval_cutoff: float = 0.05,
    fc_cutoff: float = 0.5,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create volcano plot of DE results.

    Parameters
    ----------
    de_result : DEResult
        Differential expression results
    pval_cutoff : float
        P-value threshold
    fc_cutoff : float
        Log2 fold change threshold
    top_n : int
        Number of top genes to label
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

    df = de_result.results

    if len(df) == 0:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    log2fc = df['log2FC'].values
    neg_log_p = -np.log10(df['padj'].values + 1e-300)

    # Classify points
    colors = []
    for fc, padj in zip(log2fc, df['padj'].values):
        if padj < pval_cutoff and fc > fc_cutoff:
            colors.append('red')
        elif padj < pval_cutoff and fc < -fc_cutoff:
            colors.append('blue')
        else:
            colors.append('gray')

    ax.scatter(log2fc, neg_log_p, c=colors, alpha=0.6, s=20)

    # Add threshold lines
    ax.axhline(-np.log10(pval_cutoff), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(fc_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-fc_cutoff, color='gray', linestyle='--', alpha=0.5)

    # Label top genes
    sig_mask = (df['padj'] < pval_cutoff) & (abs(df['log2FC']) > fc_cutoff)
    sig_df = df[sig_mask].head(top_n)

    for _, row in sig_df.iterrows():
        ax.annotate(
            row['gene'],
            (row['log2FC'], -np.log10(row['padj'] + 1e-300)),
            fontsize=8, alpha=0.8
        )

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(adjusted p-value)')
    ax.set_title(f'Differential Expression: {de_result.group2} vs {de_result.group1}')

    # Add counts
    n_up = np.sum((df['padj'] < pval_cutoff) & (df['log2FC'] > fc_cutoff))
    n_down = np.sum((df['padj'] < pval_cutoff) & (df['log2FC'] < -fc_cutoff))
    ax.text(0.02, 0.98, f'Up: {n_up}\nDown: {n_down}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_ma(
    de_result: DEResult,
    pval_cutoff: float = 0.05,
    fc_cutoff: float = 0.5,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create MA plot (log2FC vs mean expression).

    Parameters
    ----------
    de_result : DEResult
        Differential expression results
    pval_cutoff : float
        P-value threshold
    fc_cutoff : float
        Log2 fold change threshold
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

    df = de_result.results

    if len(df) == 0:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Get mean columns
    mean_cols = [c for c in df.columns if c.startswith('mean_')]
    if len(mean_cols) >= 2:
        mean_expr = np.log10((df[mean_cols[0]] + df[mean_cols[1]]) / 2 + 1)
    else:
        mean_expr = np.zeros(len(df))

    log2fc = df['log2FC'].values

    # Classify points
    colors = []
    for fc, padj in zip(log2fc, df['padj'].values):
        if padj < pval_cutoff and fc > fc_cutoff:
            colors.append('red')
        elif padj < pval_cutoff and fc < -fc_cutoff:
            colors.append('blue')
        else:
            colors.append('gray')

    ax.scatter(mean_expr, log2fc, c=colors, alpha=0.6, s=20)

    # Add threshold lines
    ax.axhline(fc_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-fc_cutoff, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Log10(Mean Expression)')
    ax.set_ylabel('Log2 Fold Change')
    ax.set_title(f'MA Plot: {de_result.group2} vs {de_result.group1}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_de_heatmap(
    expression: np.ndarray,
    de_result: DEResult,
    groups: np.ndarray,
    gene_names: np.ndarray,
    top_n: int = 20,
    pval_cutoff: float = 0.05,
    fc_cutoff: float = 0.5,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot heatmap of top DE genes.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    de_result : DEResult
        DE results
    groups : np.ndarray
        Group labels
    gene_names : np.ndarray
        Gene names
    top_n : int
        Number of top genes per direction
    pval_cutoff : float
        P-value threshold
    fc_cutoff : float
        Log2 fold change threshold
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

    df = de_result.results

    if len(df) == 0:
        print("No results to plot")
        return

    # Get top up and down genes
    sig = df[(df['padj'] < pval_cutoff) & (abs(df['log2FC']) > fc_cutoff)]

    up_genes = sig[sig['log2FC'] > 0].head(top_n)['gene'].values
    down_genes = sig[sig['log2FC'] < 0].head(top_n)['gene'].values

    selected_genes = np.concatenate([up_genes, down_genes])

    if len(selected_genes) == 0:
        print("No significant genes to plot")
        return

    # Get expression for selected genes
    gene_mask = np.isin(gene_names, selected_genes)
    expr = expression[:, gene_mask]
    selected_gene_names = gene_names[gene_mask]

    # Order cells by group
    order = np.argsort(groups)
    expr = expr[order]
    groups_ordered = groups[order]

    # Z-score normalize
    expr_z = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-10)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(expr_z.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

    # Gene labels
    ax.set_yticks(range(len(selected_gene_names)))
    ax.set_yticklabels(selected_gene_names, fontsize=8)

    # Group separators
    unique_groups = np.unique(groups_ordered)
    boundaries = []
    for g in unique_groups[:-1]:
        idx = np.where(groups_ordered == g)[0][-1]
        boundaries.append(idx + 0.5)
        ax.axvline(idx + 0.5, color='black', linewidth=2)

    ax.set_xlabel('Cells')
    ax.set_ylabel('Genes')
    ax.set_title(f'Top DE Genes: {de_result.group2} vs {de_result.group1}')

    plt.colorbar(im, ax=ax, label='Z-score')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def compare_de_methods(
    expression: np.ndarray,
    groups: np.ndarray,
    gene_names: np.ndarray,
    methods: List[str] = ['wilcoxon', 't-test'],
    **kwargs
) -> pd.DataFrame:
    """
    Compare different DE methods.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix
    groups : np.ndarray
        Group labels
    gene_names : np.ndarray
        Gene names
    methods : List[str]
        Methods to compare
    **kwargs
        Additional arguments

    Returns
    -------
    pd.DataFrame
        Comparison of methods
    """
    results = {}

    for method in methods:
        de_result = differential_expression(
            expression, groups, gene_names,
            method=method, **kwargs
        )
        results[method] = de_result.results.set_index('gene')

    # Merge results
    comparison = pd.DataFrame(index=results[methods[0]].index)

    for method in methods:
        comparison[f'{method}_log2FC'] = results[method]['log2FC']
        comparison[f'{method}_padj'] = results[method]['padj']

    return comparison
