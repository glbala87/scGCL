"""Basic tests for scGCL."""

import pytest
import numpy as np


def test_simulate_data():
    """Test data simulation."""
    from scgcl.utils import simulate_scrna_data

    X, y = simulate_scrna_data(n_cells=100, n_genes=200, n_clusters=3)

    assert X.shape == (100, 200)
    assert len(y) == 100
    assert len(np.unique(y)) == 3


def test_graph_construction():
    """Test graph building."""
    from scgcl.utils import build_knn_graph, build_adaptive_knn_graph

    X = np.random.randn(50, 20).astype(np.float32)

    edge_index, edge_weight = build_knn_graph(X, k=5)
    assert edge_index.shape[0] == 2
    assert len(edge_weight) == edge_index.shape[1]

    edge_index, edge_weight = build_adaptive_knn_graph(X, k_min=3, k_max=10)
    assert edge_index.shape[0] == 2


def test_model_creation():
    """Test model instantiation."""
    from scgcl import ScGCL

    model = ScGCL(n_clusters=5, hidden_dim=32, pretrain_epochs=5, ssc_epochs=5)
    assert model.n_clusters == 5
    assert model.hidden_dim == 32


def test_model_fit():
    """Test model fitting on small data."""
    from scgcl import ScGCL
    from scgcl.utils import simulate_scrna_data

    X, y = simulate_scrna_data(n_cells=50, n_genes=100, n_clusters=3)

    model = ScGCL(
        n_clusters=3,
        hidden_dim=16,
        pretrain_epochs=2,
        ssc_epochs=2
    )

    labels = model.fit_predict(X, preprocess=False, verbose=False)

    assert len(labels) == 50
    assert len(np.unique(labels)) <= 3


def test_metrics():
    """Test evaluation metrics."""
    from scgcl.utils import clustering_metrics

    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    metrics = clustering_metrics(y_true, y_pred)

    assert metrics['ARI'] == 1.0
    assert metrics['NMI'] == 1.0
    assert metrics['ACC'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
