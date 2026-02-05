#!/usr/bin/env python
"""
scGCL Tutorial: Single-cell clustering with graph contrastive learning.

Run: python examples/tutorial.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def example_basic():
    """Basic usage with simulated data."""
    from scgcl import ScGCL
    from scgcl.utils import simulate_scrna_data

    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Generate simulated scRNA-seq data
    print("\nGenerating simulated data...")
    X, y = simulate_scrna_data(
        n_cells=1000,
        n_genes=2000,
        n_clusters=5,
        dropout_rate=0.5
    )
    print(f"Data shape: {X.shape}, Clusters: {len(np.unique(y))}")

    # Create and train model
    print("\nTraining scGCL...")
    model = ScGCL(
        n_clusters=5,
        hidden_dim=64,
        pretrain_epochs=50,
        ssc_epochs=100,
        device='cpu'
    )

    labels = model.fit_predict(X, y=y)
    print(f"\nPredicted {len(np.unique(labels))} clusters")

    return model


def example_auto_k():
    """Automatic cluster number estimation."""
    from scgcl import ScGCL
    from scgcl.utils import simulate_scrna_data

    print("\n" + "=" * 60)
    print("Example 2: Automatic Cluster Estimation")
    print("=" * 60)

    X, y = simulate_scrna_data(n_cells=800, n_genes=1500, n_clusters=7)

    # Don't specify n_clusters - let scGCL estimate it
    model = ScGCL(
        n_clusters=None,  # Auto-estimate
        pretrain_epochs=30,
        ssc_epochs=50
    )

    labels = model.fit_predict(X, y=y, verbose=True)
    print(f"\nEstimated and found {len(np.unique(labels))} clusters (true: 7)")

    return model


def example_visualization():
    """Visualize results."""
    from scgcl import ScGCL
    from scgcl.utils import simulate_scrna_data, plot_umap, plot_clusters, plot_training_curves
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("Example 3: Visualization")
    print("=" * 60)

    X, y = simulate_scrna_data(n_cells=1000, n_genes=2000, n_clusters=5)

    model = ScGCL(n_clusters=5, pretrain_epochs=50, ssc_epochs=100)
    labels = model.fit_predict(X, y=y, verbose=False)

    embeddings = model.get_embeddings()

    # Create visualizations
    print("\nCreating visualizations...")

    fig = plot_clusters(embeddings, y, labels)
    fig.savefig("clusters_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved: clusters_comparison.png")

    fig = plot_training_curves(model.pretrain_losses)
    fig.savefig("training_loss.png", dpi=150, bbox_inches='tight')
    print("Saved: training_loss.png")

    plt.close('all')
    return model


def example_gpu():
    """GPU-accelerated clustering."""
    import torch

    print("\n" + "=" * 60)
    print("Example 4: GPU Acceleration")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU example.")
        print("To use GPU, install PyTorch with CUDA support.")
        return None

    from scgcl import ScGCLGPU
    from scgcl.utils import simulate_scrna_data

    # Larger dataset for GPU
    X, y = simulate_scrna_data(n_cells=5000, n_genes=2000, n_clusters=10)

    model = ScGCLGPU(
        n_clusters=10,
        pretrain_epochs=50,
        ssc_epochs=100,
        use_amp=True,  # Mixed precision
        batch_size=256
    )

    labels = model.fit_predict(X, y=y)
    return model


def example_experiments():
    """Run multiple experiments for statistics."""
    from scgcl import run_experiment
    from scgcl.utils import simulate_scrna_data

    print("\n" + "=" * 60)
    print("Example 5: Multiple Runs with Statistics")
    print("=" * 60)

    X, y = simulate_scrna_data(n_cells=500, n_genes=1000, n_clusters=5)

    results = run_experiment(
        X, y,
        n_runs=3,  # Use more runs in practice
        n_clusters=5,
        pretrain_epochs=30,
        ssc_epochs=50
    )

    print(f"\nARI: {results['ARI']['mean']:.4f} +/- {results['ARI']['std']:.4f}")
    print(f"NMI: {results['NMI']['mean']:.4f} +/- {results['NMI']['std']:.4f}")

    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("scGCL Tutorial")
    print("=" * 60 + "\n")

    example_basic()
    example_auto_k()
    example_visualization()
    example_gpu()
    example_experiments()

    print("\n" + "=" * 60)
    print("Tutorial completed!")
    print("=" * 60)
