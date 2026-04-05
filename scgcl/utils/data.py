"""Data loading and preprocessing utilities."""

import logging
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


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
    # Convert sparse matrices to dense
    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32).copy()

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array with shape {X.shape}")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Input matrix is empty: shape {X.shape}")

    # Replace NaN/Inf with 0
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning("Input contains %d NaN and %d Inf values; replacing with 0", nan_count, inf_count)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    gene_counts = (X > 0).sum(axis=0)
    cell_counts = (X > 0).sum(axis=1)
    X = X[cell_counts >= min_genes][:, gene_counts >= min_cells]

    if X.shape[0] == 0:
        raise ValueError(
            f"All cells were filtered out (min_genes={min_genes}). "
            "Lower min_genes or check that input data contains nonzero values."
        )
    if X.shape[1] == 0:
        raise ValueError(
            f"All genes were filtered out (min_cells={min_cells}). "
            "Lower min_cells or check that input data contains nonzero values."
        )

    logger.info("Filtered to %d cells and %d genes", X.shape[0], X.shape[1])

    if normalize:
        lib_sizes = np.maximum(X.sum(axis=1, keepdims=True), 1)
        X = X / lib_sizes * np.median(lib_sizes)

    if log_transform:
        X = np.log1p(X)

    if n_top_genes and n_top_genes < X.shape[1]:
        top_idx = np.argsort(np.var(X, axis=0))[-n_top_genes:]
        X = X[:, top_idx]
        logger.info("Selected top %d variable genes", n_top_genes)

    if scale:
        X = np.clip(StandardScaler().fit_transform(X), -10, 10)

    X_pca = None
    if n_pca_components:
        n_comp = min(n_pca_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        logger.info("PCA: %d components, %.2f%% variance", n_comp, pca.explained_variance_ratio_.sum() * 100)

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
