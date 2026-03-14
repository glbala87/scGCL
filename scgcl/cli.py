"""Command-line interface for scGCL."""

import argparse
import sys
import os
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='scgcl',
        description='scGCL: Single-Cell Graph Contrastive Learning for Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic clustering
  scgcl cluster data.h5ad -o results/

  # Specify number of clusters
  scgcl cluster data.csv --n-clusters 10 -o results/

  # With marker gene detection
  scgcl cluster data.h5ad --markers -o results/

  # Hyperparameter tuning
  scgcl tune data.h5ad --n-trials 50

  # Quick info about the package
  scgcl info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Run clustering')
    cluster_parser.add_argument('input', help='Input file (H5AD, CSV, or MTX)')
    cluster_parser.add_argument('-o', '--output', default='.', help='Output directory')
    cluster_parser.add_argument('-n', '--n-clusters', type=int, default=None,
                                help='Number of clusters (auto-detect if not specified)')
    cluster_parser.add_argument('--hidden-dim', type=int, default=64,
                                help='Hidden dimension (default: 64)')
    cluster_parser.add_argument('--pretrain-epochs', type=int, default=100,
                                help='Pretraining epochs (default: 100)')
    cluster_parser.add_argument('--ssc-epochs', type=int, default=500,
                                help='SSC epochs (default: 500)')
    cluster_parser.add_argument('--encoder', choices=['gcn', 'gat'], default='gcn',
                                help='Encoder type (default: gcn)')
    cluster_parser.add_argument('--k-neighbors', type=int, default=15,
                                help='Number of neighbors for graph (default: 15)')
    cluster_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed (default: 42)')
    cluster_parser.add_argument('--markers', action='store_true',
                                help='Find marker genes for each cluster')
    cluster_parser.add_argument('--n-markers', type=int, default=10,
                                help='Number of marker genes per cluster (default: 10)')
    cluster_parser.add_argument('--no-preprocess', action='store_true',
                                help='Skip preprocessing')
    cluster_parser.add_argument('--save-model', action='store_true',
                                help='Save trained model')
    cluster_parser.add_argument('-q', '--quiet', action='store_true',
                                help='Suppress output')

    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Hyperparameter tuning')
    tune_parser.add_argument('input', help='Input file')
    tune_parser.add_argument('-o', '--output', default='.', help='Output directory')
    tune_parser.add_argument('-n', '--n-clusters', type=int, default=None,
                             help='Number of clusters')
    tune_parser.add_argument('--n-trials', type=int, default=50,
                             help='Number of tuning trials (default: 50)')
    tune_parser.add_argument('--metric', default='silhouette',
                             choices=['silhouette', 'calinski', 'davies'],
                             help='Optimization metric (default: silhouette)')
    tune_parser.add_argument('--timeout', type=int, default=None,
                             help='Timeout in seconds')
    tune_parser.add_argument('--no-preprocess', action='store_true',
                             help='Skip preprocessing')
    tune_parser.add_argument('-q', '--quiet', action='store_true',
                             help='Suppress output')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')

    # Version
    parser.add_argument('--version', action='store_true', help='Show version')

    return parser


def load_data(filepath: str):
    """Load data from various formats."""
    import numpy as np

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.h5ad':
        try:
            import anndata
            adata = anndata.read_h5ad(filepath)
            X = adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
            gene_names = adata.var_names.tolist()
            return np.asarray(X), gene_names, adata
        except ImportError:
            raise ImportError("anndata required for H5AD files. Install with: pip install anndata")

    elif ext == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath, index_col=0)
        return df.values, df.columns.tolist(), None

    elif ext == '.tsv':
        import pandas as pd
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        return df.values, df.columns.tolist(), None

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def cmd_cluster(args):
    """Run clustering command."""
    import numpy as np

    from . import ScGCL, set_seed
    from .analysis.markers import find_marker_genes

    verbose = not args.quiet

    if verbose:
        print(f"Loading data from {args.input}...")

    X, gene_names, adata = load_data(args.input)

    if verbose:
        print(f"Data shape: {X.shape[0]} cells x {X.shape[1]} genes")

    # Set seed
    set_seed(args.seed)

    # Create model
    model = ScGCL(
        n_clusters=args.n_clusters,
        hidden_dim=args.hidden_dim,
        pretrain_epochs=args.pretrain_epochs,
        ssc_epochs=args.ssc_epochs,
        encoder_type=args.encoder,
        k_neighbors=args.k_neighbors,
        seed=args.seed
    )

    # Fit model
    if verbose:
        print("\nRunning scGCL clustering...")

    model.fit(X, preprocess=not args.no_preprocess, verbose=verbose)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save labels
    labels_path = os.path.join(args.output, 'labels.csv')
    import pandas as pd
    labels_df = pd.DataFrame({
        'cell': range(len(model.labels_)),
        'cluster': model.labels_,
        'confidence': model.get_confidence_scores()
    })
    labels_df.to_csv(labels_path, index=False)

    if verbose:
        print(f"\nCluster labels saved to {labels_path}")
        print(f"Found {len(np.unique(model.labels_))} clusters")

    # Save embeddings
    emb_path = os.path.join(args.output, 'embeddings.csv')
    embeddings = model.get_embeddings()
    emb_df = pd.DataFrame(embeddings, columns=[f'dim_{i}' for i in range(embeddings.shape[1])])
    emb_df.to_csv(emb_path, index=False)

    if verbose:
        print(f"Embeddings saved to {emb_path}")

    # Find markers if requested
    if args.markers:
        if verbose:
            print("\nFinding marker genes...")

        markers_df = find_marker_genes(
            X, model.labels_, gene_names,
            n_markers=args.n_markers
        )

        markers_path = os.path.join(args.output, 'markers.csv')
        markers_df.to_csv(markers_path, index=False)

        if verbose:
            print(f"Marker genes saved to {markers_path}")

            # Print top markers per cluster
            print("\nTop markers per cluster:")
            for cluster in sorted(markers_df['cluster'].unique()):
                cluster_markers = markers_df[markers_df['cluster'] == cluster]['gene'].head(5).tolist()
                print(f"  Cluster {cluster}: {', '.join(cluster_markers)}")

    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output, 'model.pt')
        model.save(model_path)
        if verbose:
            print(f"Model saved to {model_path}")

    # Update AnnData if available
    if adata is not None:
        adata.obs['scgcl_clusters'] = model.labels_.astype(str)
        adata.obs['scgcl_confidence'] = model.get_confidence_scores()
        adata.obsm['X_scgcl'] = embeddings

        adata_path = os.path.join(args.output, 'result.h5ad')
        adata.write_h5ad(adata_path)
        if verbose:
            print(f"AnnData saved to {adata_path}")

    if verbose:
        print("\nDone!")


def cmd_tune(args):
    """Run hyperparameter tuning command."""
    from .tuning import HyperparameterTuner
    import json

    verbose = not args.quiet

    if verbose:
        print(f"Loading data from {args.input}...")

    X, gene_names, _ = load_data(args.input)

    if verbose:
        print(f"Data shape: {X.shape[0]} cells x {X.shape[1]} genes")
        print(f"\nStarting hyperparameter tuning with {args.n_trials} trials...")

    tuner = HyperparameterTuner(
        n_trials=args.n_trials,
        metric=args.metric,
        timeout=args.timeout,
        verbose=verbose
    )

    result = tuner.tune(X, n_clusters=args.n_clusters, preprocess=not args.no_preprocess)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save results
    results_path = os.path.join(args.output, 'tuning_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': result.best_params,
            'best_score': result.best_score,
            'best_trial': result.best_trial,
            'metric': args.metric,
            'n_trials': args.n_trials
        }, f, indent=2)

    if verbose:
        print(result.summary())
        print(f"\nResults saved to {results_path}")


def cmd_info(args):
    """Show package information."""
    from . import __version__

    print(f"""
scGCL: Single-Cell Graph Contrastive Learning
Version: {__version__}

A framework for clustering single-cell RNA sequencing data using
graph neural networks and contrastive learning.

Features:
  - Graph-based clustering with GCN/GAT encoders
  - Self-supervised contrastive pretraining
  - Automatic cluster number estimation
  - Confidence scores for predictions
  - Marker gene detection
  - Scanpy integration

Usage:
  scgcl cluster <input> [options]   Run clustering
  scgcl tune <input> [options]      Hyperparameter tuning
  scgcl info                        Show this information

For more help:
  scgcl <command> --help

Documentation: https://github.com/your-org/scgcl
    """)


def main(argv: Optional[list] = None):
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(f"scgcl {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == 'cluster':
            cmd_cluster(args)
        elif args.command == 'tune':
            cmd_tune(args)
        elif args.command == 'info':
            cmd_info(args)
        else:
            parser.print_help()
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
