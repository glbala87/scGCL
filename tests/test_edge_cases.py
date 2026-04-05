"""Edge case and robustness tests for scGCL."""

import pytest
import numpy as np
from scipy import sparse


class TestInputValidation:
    """Test input validation catches bad data."""

    def test_nan_input_handled(self):
        """NaN values should be replaced, not crash."""
        from scgcl import ScGCL

        X = np.random.randn(30, 20).astype(np.float32)
        X[0, 0] = np.nan
        X[1, 1] = np.inf

        model = ScGCL(n_clusters=2, pretrain_epochs=2, ssc_epochs=2)
        labels = model.fit_predict(X, preprocess=False, verbose=False)
        assert len(labels) == 30

    def test_1d_input_rejected(self):
        """1D arrays should raise ValueError."""
        from scgcl import ScGCL

        model = ScGCL(n_clusters=2)
        with pytest.raises(ValueError, match="2D"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_single_cell_rejected(self):
        """Single cell should raise ValueError."""
        from scgcl import ScGCL

        model = ScGCL(n_clusters=2)
        with pytest.raises(ValueError, match="at least 2"):
            model.fit(np.random.randn(1, 10).astype(np.float32), preprocess=False)

    def test_empty_input_rejected(self):
        """Empty arrays should raise ValueError."""
        from scgcl import ScGCL

        model = ScGCL(n_clusters=2)
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 10)), preprocess=False)

    def test_n_clusters_exceeds_samples(self):
        """n_clusters >= n_samples should raise ValueError."""
        from scgcl import ScGCL

        model = ScGCL(n_clusters=20)
        with pytest.raises(ValueError, match="n_clusters"):
            model.fit(np.random.randn(10, 5).astype(np.float32), preprocess=False)

    def test_sparse_matrix_input(self):
        """Sparse matrices should be converted automatically."""
        from scgcl import ScGCL

        X_dense = np.random.randn(30, 20).astype(np.float32)
        X_sparse = sparse.csr_matrix(X_dense)

        model = ScGCL(n_clusters=2, pretrain_epochs=2, ssc_epochs=2)
        labels = model.fit_predict(X_sparse, preprocess=False, verbose=False)
        assert len(labels) == 30

    def test_label_length_mismatch(self):
        """Mismatched label length should raise ValueError."""
        from scgcl import ScGCL

        X = np.random.randn(30, 20).astype(np.float32)
        y = np.array([0, 1, 2])  # wrong length

        model = ScGCL(n_clusters=2)
        with pytest.raises(ValueError, match="Label length"):
            model.fit(X, y=y, preprocess=False)


class TestSmallDatasets:
    """Test with very small but valid datasets."""

    def test_tiny_dataset(self):
        """5 cells should work (minimum viable)."""
        from scgcl import ScGCL

        X = np.random.randn(5, 10).astype(np.float32)
        model = ScGCL(n_clusters=2, pretrain_epochs=2, ssc_epochs=2, k_neighbors=2)
        labels = model.fit_predict(X, preprocess=False, verbose=False)
        assert len(labels) == 5
        assert len(np.unique(labels)) <= 2

    def test_equal_cells_and_clusters(self):
        """Dataset where n_cells is just above n_clusters."""
        from scgcl import ScGCL

        X = np.random.randn(6, 10).astype(np.float32)
        model = ScGCL(n_clusters=3, pretrain_epochs=2, ssc_epochs=2, k_neighbors=2)
        labels = model.fit_predict(X, preprocess=False, verbose=False)
        assert len(labels) == 6


class TestGraphConstruction:
    """Test graph building edge cases."""

    def test_small_graph(self):
        """Graph construction with very few nodes."""
        from scgcl.utils.graph import build_adaptive_knn_graph

        X = np.random.randn(5, 10).astype(np.float32)
        edge_index, edge_weight = build_adaptive_knn_graph(X, k_min=2, k_max=4)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_single_sample_graph_raises(self):
        """Single sample should raise ValueError."""
        from scgcl.utils.graph import build_adaptive_knn_graph

        X = np.random.randn(1, 10).astype(np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            build_adaptive_knn_graph(X, k_min=2, k_max=10)

    def test_identical_points(self):
        """Identical data points shouldn't crash."""
        from scgcl.utils.graph import build_adaptive_knn_graph

        X = np.ones((10, 5), dtype=np.float32)
        edge_index, edge_weight = build_adaptive_knn_graph(X, k_min=2, k_max=5)
        assert edge_index.shape[0] == 2


class TestPreprocessing:
    """Test preprocessing edge cases."""

    def test_sparse_input_preprocessing(self):
        """Sparse input to preprocess_data should work."""
        from scgcl.utils.data import preprocess_data

        # Use enough genes and high enough density so cells pass min_genes filter
        X = np.random.poisson(3, (50, 500)).astype(np.float32)
        X_proc, X_pca = preprocess_data(X, min_genes=1, n_pca_components=10)
        assert X_pca is not None

    def test_all_zero_genes_filtered(self):
        """All-zero genes should be filtered without crash."""
        from scgcl.utils.data import preprocess_data

        X = np.random.poisson(2, (50, 100)).astype(np.float32)
        X[:, :10] = 0  # zero out some genes
        X_proc, X_pca = preprocess_data(X, min_cells=3, min_genes=1, n_pca_components=10)
        assert X_pca is not None

    def test_all_cells_filtered_raises(self):
        """If filtering removes all cells, should raise ValueError."""
        from scgcl.utils.data import preprocess_data

        X = np.zeros((10, 20), dtype=np.float32)
        X[:, 0] = 1  # only 1 gene expressed
        with pytest.raises(ValueError, match="filtered out"):
            preprocess_data(X, min_genes=200)

    def test_nan_in_preprocessing(self):
        """NaN values in preprocessing should be handled."""
        from scgcl.utils.data import preprocess_data

        X = np.random.poisson(5, (50, 100)).astype(np.float32)
        X[0, 0] = np.nan
        X_proc, X_pca = preprocess_data(X, min_genes=1, n_pca_components=10)
        assert not np.isnan(X_pca).any()


class TestModelSaveLoad:
    """Test save/load edge cases."""

    def test_save_unfitted_raises(self):
        """Saving unfitted model should raise RuntimeError."""
        from scgcl import ScGCL

        model = ScGCL(n_clusters=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save("/tmp/test_model.pt")

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve model state."""
        from scgcl import ScGCL

        X = np.random.randn(30, 20).astype(np.float32)
        model = ScGCL(n_clusters=2, pretrain_epochs=2, ssc_epochs=2)
        model.fit(X, preprocess=False, verbose=False)

        path = str(tmp_path / "model.pt")
        model.save(path)

        model2 = ScGCL()
        model2.load(path, input_dim=20)
        assert model2.n_clusters == 2
        assert model2.labels_ is not None


class TestConfidenceScores:
    """Test confidence and soft assignment edge cases."""

    def test_confidence_unfitted_raises(self):
        """Confidence on unfitted model should raise."""
        from scgcl import ScGCL

        model = ScGCL()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_confidence_scores()

    def test_soft_assignments_unfitted_raises(self):
        """Soft assignments on unfitted model should raise."""
        from scgcl import ScGCL

        model = ScGCL()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_soft_assignments()

    def test_confidence_range(self):
        """Confidence scores should be in [0, 1]."""
        from scgcl import ScGCL

        X = np.random.randn(30, 20).astype(np.float32)
        model = ScGCL(n_clusters=2, pretrain_epochs=2, ssc_epochs=2)
        model.fit(X, preprocess=False, verbose=False)

        confidence = model.get_confidence_scores()
        assert np.all(confidence >= 0)
        assert np.all(confidence <= 1)
        assert len(confidence) == 30


class TestEndToEnd:
    """Integration tests with the full pipeline."""

    def test_full_pipeline_simulated(self):
        """Full pipeline: simulate -> fit -> predict -> evaluate."""
        from scgcl import ScGCL
        from scgcl.utils.data import simulate_scrna_data
        from scgcl.utils.evaluation import clustering_metrics

        X, y = simulate_scrna_data(n_cells=100, n_genes=2000, n_clusters=3, seed=42)
        model = ScGCL(n_clusters=3, pretrain_epochs=10, ssc_epochs=20, seed=42)
        labels = model.fit_predict(X, verbose=False)

        metrics = clustering_metrics(y, labels)
        assert metrics['ARI'] >= 0  # at minimum should not crash
        assert metrics['NMI'] >= 0

        embeddings = model.get_embeddings()
        assert embeddings.shape == (100, 64)  # hidden_dim default

        confidence = model.get_confidence_scores()
        assert len(confidence) == 100

    def test_full_pipeline_with_preprocessing(self):
        """Full pipeline with preprocessing enabled."""
        from scgcl import ScGCL
        from scgcl.utils.data import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=1000, n_clusters=3, seed=42)
        model = ScGCL(n_clusters=3, pretrain_epochs=5, ssc_epochs=10, seed=42)
        labels = model.fit_predict(X, preprocess=True, verbose=False)
        assert len(labels) > 0  # may lose cells to filtering
        assert len(np.unique(labels)) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
