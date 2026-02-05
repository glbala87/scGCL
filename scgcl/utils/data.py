"""Data loading and preprocessing utilities."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional


def load_data(filepath: str, file_format: str = 'auto') -> Tuple[np.ndarray, Optional[np.ndarray], Optional[list]]:
    """
    Load single-cell expression data.

    Parameters
    ----------
    filepath : str
        Path to data file
    file_format : str
        Format ('csv', 'h5ad', 'mtx', 'auto')

    Returns
    -------
    X : np.ndarray
        Expression matrix (cells x genes)
    labels : np.ndarray or None
        Ground truth labels if available
    gene_names : list or None
        Gene names if available
    """
    if file_format == 'auto':
        if filepath.endswith('.h5ad'):
            file_format = 'h5ad'
        elif filepath.endswith('.csv'):
            file_format = 'csv'
        elif filepath.endswith('.tsv'):
            file_format = 'tsv'
        elif filepath.endswith('.mtx'):
            file_format = 'mtx'
        else:
            raise ValueError(f"Cannot auto-detect format for {filepath}")

    labels, gene_names = None, None

    if file_format == 'h5ad':
        import anndata
        adata = anndata.read_h5ad(filepath)
        X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        if 'cell_type' in adata.obs:
            labels = adata.obs['cell_type'].values
        gene_names = adata.var_names.tolist()

    elif file_format in ['csv', 'tsv']:
        sep = ',' if file_format == 'csv' else '\t'
        df = pd.read_csv(filepath, sep=sep, index_col=0)
        X = df.values
        gene_names = df.columns.tolist()

    elif file_format == 'mtx':
        from scipy.io import mmread
        X = mmread(filepath).T
        if sparse.issparse(X):
            X = X.toarray()

    return X.astype(np.float32), labels, gene_names


def preprocess_data(
    X: np.ndarray,
    normalize: bool = True,
    log_transform: bool = True,
    scale: bool = True,
    n_top_genes: Optional[int] = 2000,
    n_pca_components: Optional[int] = 50,
    min_cells: int = 3,
    min_genes: int = 200
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess single-cell expression data.

    Parameters
    ----------
    X : np.ndarray
        Raw expression matrix (cells x genes)
    normalize : bool
        Library size normalization
    log_transform : bool
        Apply log1p transformation
    scale : bool
        Z-score normalization
    n_top_genes : int, optional
        Number of highly variable genes
    n_pca_components : int, optional
        Number of PCA components

    Returns
    -------
    X_processed : np.ndarray
        Processed expression matrix
    X_pca : np.ndarray or None
        PCA-reduced matrix
    """
    X = X.copy()

    gene_counts = (X > 0).sum(axis=0)
    cell_counts = (X > 0).sum(axis=1)
    X = X[cell_counts >= min_genes][:, gene_counts >= min_cells]

    print(f"Filtered to {X.shape[0]} cells and {X.shape[1]} genes")

    if normalize:
        lib_sizes = np.maximum(X.sum(axis=1, keepdims=True), 1)
        X = X / lib_sizes * np.median(lib_sizes)

    if log_transform:
        X = np.log1p(X)

    if n_top_genes and n_top_genes < X.shape[1]:
        top_idx = np.argsort(np.var(X, axis=0))[-n_top_genes:]
        X = X[:, top_idx]
        print(f"Selected top {n_top_genes} variable genes")

    if scale:
        X = np.clip(StandardScaler().fit_transform(X), -10, 10)

    X_pca = None
    if n_pca_components:
        n_comp = min(n_pca_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        print(f"PCA: {n_comp} components, {pca.explained_variance_ratio_.sum():.2%} variance")

    return X.astype(np.float32), X_pca


def simulate_scrna_data(
    n_cells: int = 1000,
    n_genes: int = 2000,
    n_clusters: int = 5,
    dropout_rate: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate single-cell RNA-seq data for testing."""
    np.random.seed(seed)

    labels = np.random.randint(0, n_clusters, n_cells)
    cluster_centers = np.random.exponential(scale=2, size=(n_clusters, n_genes))

    X = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        X[i] = np.random.poisson(cluster_centers[labels[i]])

    X[np.random.random((n_cells, n_genes)) < dropout_rate] = 0

    return X.astype(np.float32), labels
