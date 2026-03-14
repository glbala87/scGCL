"""Interactive visualization using Plotly."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple
import warnings


def interactive_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_by: Optional[np.ndarray] = None,
    hover_data: Optional[Dict[str, np.ndarray]] = None,
    cell_names: Optional[List[str]] = None,
    title: str = "Interactive UMAP",
    color_label: str = "Cluster",
    point_size: int = 5,
    opacity: float = 0.8,
    width: int = 900,
    height: int = 700,
    colorscale: str = 'Viridis',
    show_legend: bool = True,
    save_path: Optional[str] = None
) -> "plotly.graph_objects.Figure":
    """
    Create an interactive UMAP scatter plot with Plotly.

    Parameters
    ----------
    embeddings : np.ndarray
        2D embeddings (n_cells x 2), e.g., UMAP coordinates
    labels : np.ndarray, optional
        Cluster labels for coloring (categorical)
    color_by : np.ndarray, optional
        Continuous values for coloring (e.g., gene expression)
    hover_data : dict, optional
        Additional data to show on hover {name: values}
    cell_names : list, optional
        Cell names/IDs
    title : str
        Plot title
    color_label : str
        Label for the color legend
    point_size : int
        Size of scatter points
    opacity : float
        Point opacity (0-1)
    width : int
        Figure width in pixels
    height : int
        Figure height in pixels
    colorscale : str
        Plotly colorscale for continuous coloring
    show_legend : bool
        Whether to show legend
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required for interactive plots. Install with: pip install plotly")

    n_cells = embeddings.shape[0]

    # Build DataFrame for plotting
    df = pd.DataFrame({
        'UMAP1': embeddings[:, 0],
        'UMAP2': embeddings[:, 1],
    })

    if cell_names is not None:
        df['Cell'] = cell_names
    else:
        df['Cell'] = [f'Cell_{i}' for i in range(n_cells)]

    # Add labels
    if labels is not None:
        df['Cluster'] = labels.astype(str)

    # Add hover data
    if hover_data is not None:
        for name, values in hover_data.items():
            df[name] = values

    # Determine coloring
    if color_by is not None:
        df['ColorValue'] = color_by
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='ColorValue',
            color_continuous_scale=colorscale,
            hover_name='Cell',
            hover_data={col: True for col in df.columns if col not in ['UMAP1', 'UMAP2']},
            title=title,
            labels={'ColorValue': color_label}
        )
    elif labels is not None:
        # Categorical coloring by cluster
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='Cluster',
            hover_name='Cell',
            hover_data={col: True for col in df.columns if col not in ['UMAP1', 'UMAP2', 'Cluster']},
            title=title,
            category_orders={'Cluster': sorted(df['Cluster'].unique(), key=lambda x: int(x) if x.isdigit() else x)}
        )
    else:
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            hover_name='Cell',
            hover_data={col: True for col in df.columns if col not in ['UMAP1', 'UMAP2']},
            title=title
        )

    # Update layout
    fig.update_traces(
        marker=dict(size=point_size, opacity=opacity),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        width=width,
        height=height,
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white"
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive plot to {save_path}")

    return fig


def interactive_embedding(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'umap',
    n_components: int = 2,
    **kwargs
) -> "plotly.graph_objects.Figure":
    """
    Compute embedding and create interactive plot.

    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embeddings (n_cells x n_dims)
    labels : np.ndarray, optional
        Cluster labels
    method : str
        Dimensionality reduction: 'umap', 'tsne', 'pca'
    n_components : int
        Number of components (2 or 3)
    **kwargs
        Additional arguments passed to interactive_umap

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    # Compute 2D embedding if needed
    if embeddings.shape[1] > n_components:
        if method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=n_components, random_state=42)
                coords = reducer.fit_transform(embeddings)
            except ImportError:
                raise ImportError("umap-learn required. Install with: pip install umap-learn")
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(embeddings)
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        coords = embeddings[:, :n_components]

    if n_components == 3:
        return interactive_3d(coords, labels=labels, **kwargs)
    else:
        return interactive_umap(coords, labels=labels, **kwargs)


def interactive_3d(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_by: Optional[np.ndarray] = None,
    hover_data: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Interactive 3D Plot",
    point_size: int = 3,
    opacity: float = 0.8,
    width: int = 900,
    height: int = 700,
    save_path: Optional[str] = None
) -> "plotly.graph_objects.Figure":
    """
    Create an interactive 3D scatter plot.

    Parameters
    ----------
    embeddings : np.ndarray
        3D coordinates (n_cells x 3)
    labels : np.ndarray, optional
        Cluster labels
    color_by : np.ndarray, optional
        Continuous values for coloring
    hover_data : dict, optional
        Additional hover data
    title : str
        Plot title
    point_size : int
        Point size
    opacity : float
        Point opacity
    width : int
        Figure width
    height : int
        Figure height
    save_path : str, optional
        Path to save HTML

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 3D Plotly figure
    """
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly required. Install with: pip install plotly")

    df = pd.DataFrame({
        'X': embeddings[:, 0],
        'Y': embeddings[:, 1],
        'Z': embeddings[:, 2],
    })

    if labels is not None:
        df['Cluster'] = labels.astype(str)

    if hover_data:
        for name, values in hover_data.items():
            df[name] = values

    if color_by is not None:
        df['Color'] = color_by
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Color', title=title)
    elif labels is not None:
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Cluster', title=title)
    else:
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', title=title)

    fig.update_traces(marker=dict(size=point_size, opacity=opacity))
    fig.update_layout(width=width, height=height)

    if save_path:
        fig.write_html(save_path)

    return fig


def interactive_gene_expression(
    embeddings: np.ndarray,
    expression: np.ndarray,
    gene_names: List[str],
    genes: List[str],
    labels: Optional[np.ndarray] = None,
    ncols: int = 2,
    point_size: int = 4,
    width: int = 400,
    height: int = 350,
    save_path: Optional[str] = None
) -> "plotly.graph_objects.Figure":
    """
    Create interactive gene expression plots.

    Parameters
    ----------
    embeddings : np.ndarray
        2D embeddings (UMAP coordinates)
    expression : np.ndarray
        Expression matrix (cells x genes)
    gene_names : list
        All gene names
    genes : list
        Genes to plot
    labels : np.ndarray, optional
        Cluster labels
    ncols : int
        Number of columns in subplot grid
    point_size : int
        Point size
    width : int
        Width per subplot
    height : int
        Height per subplot
    save_path : str, optional
        Path to save HTML

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive figure with gene expression
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required. Install with: pip install plotly")

    gene_names_upper = [g.upper() for g in gene_names]
    gene_to_idx = {g: i for i, g in enumerate(gene_names_upper)}

    # Filter to available genes
    available_genes = [g for g in genes if g.upper() in gene_to_idx]
    if not available_genes:
        raise ValueError(f"None of the specified genes found in data")

    nrows = int(np.ceil(len(available_genes) / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=available_genes,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    for i, gene in enumerate(available_genes):
        row = i // ncols + 1
        col = i % ncols + 1

        gene_idx = gene_to_idx[gene.upper()]
        expr = expression[:, gene_idx]

        fig.add_trace(
            go.Scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=expr,
                    colorscale='Viridis',
                    showscale=(i == 0),
                    colorbar=dict(title="Expression") if i == 0 else None
                ),
                text=[f"Expr: {e:.2f}" for e in expr],
                hovertemplate="UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>%{text}<extra></extra>",
                showlegend=False
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="Gene Expression",
        width=width * ncols,
        height=height * nrows,
        template="plotly_white"
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def interactive_comparison(
    embeddings: np.ndarray,
    labels1: np.ndarray,
    labels2: np.ndarray,
    title1: str = "Clustering 1",
    title2: str = "Clustering 2",
    point_size: int = 5,
    width: int = 500,
    height: int = 450,
    save_path: Optional[str] = None
) -> "plotly.graph_objects.Figure":
    """
    Compare two clusterings side by side.

    Parameters
    ----------
    embeddings : np.ndarray
        2D embeddings
    labels1 : np.ndarray
        First set of labels
    labels2 : np.ndarray
        Second set of labels
    title1 : str
        Title for first plot
    title2 : str
        Title for second plot
    point_size : int
        Point size
    width : int
        Width per subplot
    height : int
        Height
    save_path : str, optional
        Path to save HTML

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Side-by-side comparison figure
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[title1, title2],
        horizontal_spacing=0.1
    )

    # Get unique labels and assign colors
    all_labels = np.unique(np.concatenate([labels1, labels2]))
    colors = dict(zip(all_labels, range(len(all_labels))))

    for col, (labels, title) in enumerate([(labels1, title1), (labels2, title2)], 1):
        for label in np.unique(labels):
            mask = labels == label
            fig.add_trace(
                go.Scatter(
                    x=embeddings[mask, 0],
                    y=embeddings[mask, 1],
                    mode='markers',
                    marker=dict(size=point_size),
                    name=f"Cluster {label}",
                    legendgroup=str(label),
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

    fig.update_layout(
        width=width * 2,
        height=height,
        template="plotly_white"
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def interactive_cluster_composition(
    labels: np.ndarray,
    metadata: pd.DataFrame,
    group_by: str,
    normalize: bool = True,
    title: str = "Cluster Composition",
    width: int = 800,
    height: int = 500,
    save_path: Optional[str] = None
) -> "plotly.graph_objects.Figure":
    """
    Interactive stacked bar chart of cluster composition.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    metadata : pd.DataFrame
        Cell metadata with grouping column
    group_by : str
        Column name to group by (e.g., 'batch', 'sample')
    normalize : bool
        Whether to normalize to proportions
    title : str
        Plot title
    width : int
        Figure width
    height : int
        Figure height
    save_path : str, optional
        Path to save HTML

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive bar chart
    """
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly required")

    df = pd.DataFrame({
        'Cluster': labels.astype(str),
        'Group': metadata[group_by].values
    })

    # Count cells per cluster per group
    counts = df.groupby(['Cluster', 'Group']).size().reset_index(name='Count')

    if normalize:
        totals = counts.groupby('Cluster')['Count'].transform('sum')
        counts['Proportion'] = counts['Count'] / totals
        y_col = 'Proportion'
    else:
        y_col = 'Count'

    fig = px.bar(
        counts,
        x='Cluster',
        y=y_col,
        color='Group',
        title=title,
        barmode='stack'
    )

    fig.update_layout(
        width=width,
        height=height,
        xaxis_title="Cluster",
        yaxis_title="Proportion" if normalize else "Cell Count",
        template="plotly_white"
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def interactive_violin(
    values: np.ndarray,
    labels: np.ndarray,
    title: str = "Distribution by Cluster",
    ylabel: str = "Value",
    width: int = 800,
    height: int = 500,
    save_path: Optional[str] = None
) -> "plotly.graph_objects.Figure":
    """
    Interactive violin plot of values by cluster.

    Parameters
    ----------
    values : np.ndarray
        Values to plot (e.g., confidence scores)
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    ylabel : str
        Y-axis label
    width : int
        Figure width
    height : int
        Figure height
    save_path : str, optional
        Path to save HTML

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive violin plot
    """
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly required")

    df = pd.DataFrame({
        'Cluster': labels.astype(str),
        'Value': values
    })

    fig = px.violin(
        df,
        x='Cluster',
        y='Value',
        box=True,
        points="outliers",
        title=title
    )

    fig.update_layout(
        width=width,
        height=height,
        yaxis_title=ylabel,
        template="plotly_white"
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def create_dashboard(
    embeddings: np.ndarray,
    labels: np.ndarray,
    expression: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    confidence: Optional[np.ndarray] = None,
    cell_types: Optional[np.ndarray] = None,
    marker_genes: Optional[List[str]] = None,
    title: str = "scGCL Analysis Dashboard",
    save_path: str = "dashboard.html"
) -> str:
    """
    Create a comprehensive interactive dashboard.

    Parameters
    ----------
    embeddings : np.ndarray
        2D UMAP embeddings
    labels : np.ndarray
        Cluster labels
    expression : np.ndarray, optional
        Expression matrix
    gene_names : list, optional
        Gene names
    confidence : np.ndarray, optional
        Confidence scores
    cell_types : np.ndarray, optional
        Cell type annotations
    marker_genes : list, optional
        Marker genes to plot
    title : str
        Dashboard title
    save_path : str
        Path to save HTML dashboard

    Returns
    -------
    str
        Path to saved dashboard
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required")

    # Determine layout
    n_panels = 2  # Cluster plot + one more
    if confidence is not None:
        n_panels += 1
    if expression is not None and marker_genes:
        n_panels += 1

    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    subplot_titles = ["Clusters"]
    if cell_types is not None:
        subplot_titles.append("Cell Types")
    else:
        subplot_titles.append("Cluster Sizes")

    if confidence is not None:
        subplot_titles.append("Confidence Scores")

    if expression is not None and marker_genes:
        subplot_titles.append(f"Expression: {marker_genes[0]}")

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scatter"}, {"type": "scatter" if cell_types is not None else "bar"}],
               [{"type": "scatter"} if confidence is not None else None,
                {"type": "scatter"} if expression is not None else None]][:nrows],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # Panel 1: Clusters
    for label in np.unique(labels):
        mask = labels == label
        fig.add_trace(
            go.Scatter(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                mode='markers',
                marker=dict(size=4),
                name=f"Cluster {label}",
                legendgroup=str(label)
            ),
            row=1, col=1
        )

    # Panel 2: Cell types or cluster sizes
    if cell_types is not None:
        for ct in np.unique(cell_types):
            mask = cell_types == ct
            fig.add_trace(
                go.Scatter(
                    x=embeddings[mask, 0],
                    y=embeddings[mask, 1],
                    mode='markers',
                    marker=dict(size=4),
                    name=ct,
                    showlegend=False
                ),
                row=1, col=2
            )
    else:
        unique, counts = np.unique(labels, return_counts=True)
        fig.add_trace(
            go.Bar(x=[str(l) for l in unique], y=counts, showlegend=False),
            row=1, col=2
        )

    # Panel 3: Confidence
    if confidence is not None and nrows > 1:
        fig.add_trace(
            go.Scatter(
                x=embeddings[:, 0],
                y=embeddings[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=confidence,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Confidence", x=0.45)
                ),
                showlegend=False
            ),
            row=2, col=1
        )

    # Panel 4: Gene expression
    if expression is not None and marker_genes and gene_names and nrows > 1:
        gene_upper = marker_genes[0].upper()
        gene_names_upper = [g.upper() for g in gene_names]
        if gene_upper in gene_names_upper:
            gene_idx = gene_names_upper.index(gene_upper)
            expr = expression[:, gene_idx]
            fig.add_trace(
                go.Scatter(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=expr,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Expression", x=1.0)
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )

    fig.update_layout(
        title=title,
        width=1000,
        height=500 * nrows,
        template="plotly_white",
        showlegend=True
    )

    fig.write_html(save_path)
    print(f"Dashboard saved to {save_path}")

    return save_path
