"""Scanpy integration for seamless workflow compatibility."""

import numpy as np
from typing import Optional, Union, List
import warnings


def _check_anndata():
    """Check if anndata is available."""
    try:
        import anndata
        return True
    except ImportError:
        return False


def from_anndata(
    adata,
    layer: Optional[str] = None,
    use_raw: bool = False
) -> np.ndarray:
    """
    Extract expression matrix from AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    layer : str, optional
        Layer to use. If None, uses .X
    use_raw : bool
        Whether to use .raw

    Returns
    -------
    np.ndarray
        Expression matrix
    """
    if use_raw and adata.raw is not None:
        X = adata.raw.X
    elif layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()

    return np.asarray(X)


def to_anndata(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    soft_assignments: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    cell_names: Optional[List[str]] = None
):
    """
    Create AnnData object from scGCL results.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix
    labels : np.ndarray, optional
        Cluster labels
    embeddings : np.ndarray, optional
        Learned embeddings
    confidence : np.ndarray, optional
        Confidence scores
    soft_assignments : np.ndarray, optional
        Soft cluster assignments
    gene_names : list of str, optional
        Gene names
    cell_names : list of str, optional
        Cell names

    Returns
    -------
    AnnData
        Annotated data object
    """
    try:
        import anndata
        import pandas as pd
    except ImportError:
        raise ImportError("anndata and pandas required for to_anndata()")

    n_cells, n_genes = X.shape

    # Create obs DataFrame
    obs_dict = {}
    if labels is not None:
        obs_dict['scgcl_clusters'] = pd.Categorical(labels)
    if confidence is not None:
        obs_dict['scgcl_confidence'] = confidence

    obs = pd.DataFrame(obs_dict)
    if cell_names is not None:
        obs.index = cell_names
    else:
        obs.index = [f"cell_{i}" for i in range(n_cells)]

    # Create var DataFrame
    if gene_names is not None:
        var = pd.DataFrame(index=gene_names)
    else:
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    # Create AnnData
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Add embeddings to obsm
    if embeddings is not None:
        adata.obsm['X_scgcl'] = embeddings
    if soft_assignments is not None:
        adata.obsm['scgcl_soft'] = soft_assignments

    return adata


def scgcl(
    adata,
    n_clusters: Optional[int] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    key_added: str = 'scgcl',
    copy: bool = False,
    **kwargs
):
    """
    Run scGCL clustering on AnnData object.

    This function integrates scGCL into scanpy-style workflows.
    Results are stored in:
    - adata.obs['{key_added}_clusters']: Cluster labels
    - adata.obs['{key_added}_confidence']: Confidence scores
    - adata.obsm['X_{key_added}']: Learned embeddings
    - adata.obsm['{key_added}_soft']: Soft assignments
    - adata.uns['{key_added}']: Model parameters and metadata

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    n_clusters : int, optional
        Number of clusters. Auto-detected if None.
    layer : str, optional
        Layer to use for clustering
    use_raw : bool
        Whether to use raw data
    key_added : str
        Key prefix for storing results
    copy : bool
        Return a copy instead of modifying in place
    **kwargs
        Additional arguments passed to ScGCL

    Returns
    -------
    AnnData or None
        Returns AnnData if copy=True, otherwise modifies in place

    Example
    -------
    >>> import scanpy as sc
    >>> from scgcl.integration import scgcl
    >>>
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> sc.pp.normalize_total(adata)
    >>> sc.pp.log1p(adata)
    >>> sc.pp.pca(adata)
    >>>
    >>> # Run scGCL clustering
    >>> scgcl(adata, n_clusters=10)
    >>>
    >>> # Results are in adata.obs['scgcl_clusters']
    >>> sc.pl.umap(adata, color='scgcl_clusters')
    """
    try:
        import anndata
    except ImportError:
        raise ImportError("anndata required for scanpy integration")

    from ..model import ScGCL

    if copy:
        adata = adata.copy()

    # Extract expression matrix
    X = from_anndata(adata, layer=layer, use_raw=use_raw)

    # Check if PCA is available
    if 'X_pca' in adata.obsm:
        X_input = adata.obsm['X_pca']
        preprocess = False
    else:
        X_input = X
        preprocess = True

    # Separate fit kwargs from model kwargs
    verbose = kwargs.pop('verbose', True)
    preprocess_kwarg = kwargs.pop('preprocess', preprocess)

    # Initialize and fit model
    model = ScGCL(n_clusters=n_clusters, **kwargs)
    model.fit(X_input, preprocess=preprocess_kwarg, verbose=verbose)

    # Store results
    adata.obs[f'{key_added}_clusters'] = model.labels_.astype(str)
    adata.obs[f'{key_added}_clusters'] = adata.obs[f'{key_added}_clusters'].astype('category')

    adata.obs[f'{key_added}_confidence'] = model.get_confidence_scores()

    adata.obsm[f'X_{key_added}'] = model.get_embeddings()
    adata.obsm[f'{key_added}_soft'] = model.get_soft_assignments()

    # Store metadata
    adata.uns[key_added] = {
        'params': {
            'n_clusters': model.n_clusters,
            'hidden_dim': model.hidden_dim,
            'encoder_type': model.encoder_type,
        },
        'pretrain_losses': model.pretrain_losses,
        'ssc_losses': model.ssc_losses,
    }

    if copy:
        return adata


def scgcl_markers(
    adata,
    cluster_key: str = 'scgcl_clusters',
    layer: Optional[str] = None,
    use_raw: bool = True,
    n_genes: int = 10,
    method: str = 'wilcoxon',
    key_added: str = 'scgcl_markers',
    **kwargs
):
    """
    Find marker genes for scGCL clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data with scGCL clustering results
    cluster_key : str
        Key in adata.obs containing cluster labels
    layer : str, optional
        Layer to use for differential expression
    use_raw : bool
        Whether to use raw data for DE
    n_genes : int
        Number of top markers per cluster
    method : str
        Statistical method ('wilcoxon' or 't-test')
    key_added : str
        Key for storing results in adata.uns
    **kwargs
        Additional arguments for find_marker_genes

    Returns
    -------
    pd.DataFrame
        Marker gene results
    """
    from ..analysis.markers import find_marker_genes

    # Get expression matrix
    X = from_anndata(adata, layer=layer, use_raw=use_raw)

    # Get labels
    labels = adata.obs[cluster_key].astype(int).values

    # Get gene names
    if use_raw and adata.raw is not None:
        gene_names = adata.raw.var_names.tolist()
    else:
        gene_names = adata.var_names.tolist()

    # Find markers
    markers_df = find_marker_genes(
        X, labels, gene_names,
        n_markers=n_genes,
        method=method,
        **kwargs
    )

    # Store in adata
    adata.uns[key_added] = markers_df

    return markers_df


# Register with scanpy if available
def _register_scanpy_functions():
    """Register functions with scanpy.tl namespace."""
    try:
        import scanpy as sc

        # Add to scanpy.tl namespace
        if not hasattr(sc.tl, 'scgcl'):
            sc.tl.scgcl = scgcl
            sc.tl.scgcl_markers = scgcl_markers

    except ImportError:
        pass


# Try to register on import
_register_scanpy_functions()
