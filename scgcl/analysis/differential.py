"""Differential abundance analysis between conditions."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class DifferentialAbundanceResult:
    """Container for differential abundance results."""

    cluster: int
    condition1_count: int
    condition2_count: int
    condition1_proportion: float
    condition2_proportion: float
    log2_fold_change: float
    pvalue: float
    adjusted_pvalue: float

    def to_dict(self) -> dict:
        return {
            'cluster': self.cluster,
            'condition1_count': self.condition1_count,
            'condition2_count': self.condition2_count,
            'condition1_prop': self.condition1_proportion,
            'condition2_prop': self.condition2_proportion,
            'log2FC': self.log2_fold_change,
            'pvalue': self.pvalue,
            'padj': self.adjusted_pvalue
        }


def differential_abundance(
    labels: np.ndarray,
    condition: np.ndarray,
    condition1: str = None,
    condition2: str = None,
    method: str = 'fisher',
    min_cells: int = 10,
    pseudocount: float = 0.5
) -> pd.DataFrame:
    """
    Test for differential cluster abundance between conditions.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    condition : np.ndarray
        Condition labels for each cell
    condition1 : str, optional
        First condition (default: first unique value)
    condition2 : str, optional
        Second condition (default: second unique value)
    method : str
        Test method: 'fisher', 'chi2', 'permutation'
    min_cells : int
        Minimum cells per cluster to test
    pseudocount : float
        Pseudocount for log fold change

    Returns
    -------
    pd.DataFrame
        Differential abundance results
    """
    unique_conditions = np.unique(condition)

    if len(unique_conditions) < 2:
        raise ValueError("Need at least 2 conditions for differential abundance")

    if condition1 is None:
        condition1 = unique_conditions[0]
    if condition2 is None:
        condition2 = unique_conditions[1]

    # Get masks
    mask1 = condition == condition1
    mask2 = condition == condition2

    labels1 = labels[mask1]
    labels2 = labels[mask2]

    n1_total = len(labels1)
    n2_total = len(labels2)

    results = []
    pvalues = []

    for cluster in sorted(np.unique(labels)):
        n1 = np.sum(labels1 == cluster)
        n2 = np.sum(labels2 == cluster)

        # Skip if too few cells
        if n1 + n2 < min_cells:
            continue

        # Proportions
        prop1 = n1 / n1_total
        prop2 = n2 / n2_total

        # Log2 fold change with pseudocount
        log2fc = np.log2((prop2 + pseudocount) / (prop1 + pseudocount))

        # Statistical test
        if method == 'fisher':
            # Fisher's exact test
            table = [[n1, n1_total - n1], [n2, n2_total - n2]]
            _, pval = stats.fisher_exact(table)
        elif method == 'chi2':
            # Chi-squared test
            table = [[n1, n1_total - n1], [n2, n2_total - n2]]
            _, pval, _, _ = stats.chi2_contingency(table)
        elif method == 'permutation':
            # Permutation test
            pval = _permutation_test(labels, condition, cluster, condition1, condition2)
        else:
            raise ValueError(f"Unknown method: {method}")

        pvalues.append(pval)
        results.append({
            'cluster': cluster,
            'n_condition1': n1,
            'n_condition2': n2,
            'prop_condition1': prop1,
            'prop_condition2': prop2,
            'log2FC': log2fc,
            'pvalue': pval
        })

    if not results:
        return pd.DataFrame()

    # FDR correction
    pvalues = np.array(pvalues)
    adjusted = _benjamini_hochberg(pvalues)

    for i, r in enumerate(results):
        r['padj'] = adjusted[i]

    df = pd.DataFrame(results)
    df = df.sort_values('padj')

    # Rename columns with condition names
    df = df.rename(columns={
        'n_condition1': f'n_{condition1}',
        'n_condition2': f'n_{condition2}',
        'prop_condition1': f'prop_{condition1}',
        'prop_condition2': f'prop_{condition2}'
    })

    return df


def _permutation_test(
    labels: np.ndarray,
    condition: np.ndarray,
    cluster: int,
    condition1: str,
    condition2: str,
    n_permutations: int = 1000
) -> float:
    """Permutation test for differential abundance."""
    mask1 = condition == condition1
    mask2 = condition == condition2

    n1 = np.sum(labels[mask1] == cluster)
    n2 = np.sum(labels[mask2] == cluster)

    n1_total = np.sum(mask1)
    n2_total = np.sum(mask2)

    observed_diff = n1 / n1_total - n2 / n2_total

    combined_mask = mask1 | mask2
    combined_labels = labels[combined_mask]
    n_combined = len(combined_labels)

    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(n_combined)
        perm_labels1 = combined_labels[perm[:n1_total]]
        perm_labels2 = combined_labels[perm[n1_total:]]

        perm_diff = np.mean(perm_labels1 == cluster) - np.mean(perm_labels2 == cluster)

        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return (count + 1) / (n_permutations + 1)


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


def plot_differential_abundance(
    da_results: pd.DataFrame,
    pval_threshold: float = 0.05,
    fc_threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot differential abundance results as volcano plot.

    Parameters
    ----------
    da_results : pd.DataFrame
        Results from differential_abundance()
    pval_threshold : float
        P-value threshold for significance
    fc_threshold : float
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

    fig, ax = plt.subplots(figsize=figsize)

    # Classify points
    log2fc = da_results['log2FC'].values
    pvals = da_results['padj'].values
    neg_log_p = -np.log10(pvals + 1e-300)

    # Colors
    colors = []
    for fc, p in zip(log2fc, pvals):
        if p < pval_threshold and abs(fc) > fc_threshold:
            colors.append('red' if fc > 0 else 'blue')
        else:
            colors.append('gray')

    ax.scatter(log2fc, neg_log_p, c=colors, alpha=0.7)

    # Add cluster labels for significant
    for i, row in da_results.iterrows():
        if row['padj'] < pval_threshold and abs(row['log2FC']) > fc_threshold:
            ax.annotate(
                f"C{row['cluster']}",
                (row['log2FC'], -np.log10(row['padj'] + 1e-300)),
                fontsize=8
            )

    ax.axhline(-np.log10(pval_threshold), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(-fc_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(fc_threshold, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(adjusted p-value)')
    ax.set_title('Differential Abundance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def abundance_barplot(
    labels: np.ndarray,
    condition: np.ndarray,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot cluster abundance by condition as grouped bar plot.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    condition : np.ndarray
        Condition labels
    normalize : bool
        Normalize to proportions
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required")
        return

    unique_clusters = sorted(np.unique(labels))
    unique_conditions = sorted(np.unique(condition))

    # Compute counts/proportions
    data = np.zeros((len(unique_clusters), len(unique_conditions)))

    for j, cond in enumerate(unique_conditions):
        mask = condition == cond
        cond_labels = labels[mask]
        for i, cluster in enumerate(unique_clusters):
            count = np.sum(cond_labels == cluster)
            if normalize:
                data[i, j] = count / len(cond_labels)
            else:
                data[i, j] = count

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(unique_clusters))
    width = 0.8 / len(unique_conditions)

    for j, cond in enumerate(unique_conditions):
        offset = (j - len(unique_conditions) / 2 + 0.5) * width
        ax.bar(x + offset, data[:, j], width, label=str(cond))

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion' if normalize else 'Count')
    ax.set_title('Cluster Abundance by Condition')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in unique_clusters])
    ax.legend(title='Condition')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
