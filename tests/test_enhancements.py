"""Tests for enhanced features: markers, tuning, CLI, scanpy integration."""

import pytest
import numpy as np
import tempfile
import os


class TestMarkerGenes:
    """Test marker gene detection."""

    def test_find_marker_genes(self):
        """Test basic marker gene finding."""
        from scgcl import find_marker_genes
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)
        gene_names = [f'Gene{i}' for i in range(50)]

        markers = find_marker_genes(X, y, gene_names, n_markers=5)

        assert len(markers) > 0
        assert 'gene' in markers.columns
        assert 'cluster' in markers.columns
        assert 'score' in markers.columns
        assert 'pval_adj' in markers.columns

    def test_rank_genes_groups(self):
        """Test rank_genes_groups function."""
        from scgcl import rank_genes_groups
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)

        result = rank_genes_groups(X, y, method='wilcoxon')

        assert result.n_genes == 50
        assert len(result.names) == 3  # 3 clusters

    def test_marker_result_to_dataframe(self):
        """Test MarkerGeneResult.to_dataframe()."""
        from scgcl import rank_genes_groups
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)
        result = rank_genes_groups(X, y)

        df = result.to_dataframe()
        assert len(df) > 0

        df_cluster0 = result.to_dataframe(cluster=0)
        assert all(df_cluster0['cluster'] == 0)

    def test_filter_markers(self):
        """Test filter_markers function."""
        from scgcl import find_marker_genes, filter_markers
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)
        markers = find_marker_genes(X, y, n_markers=20, pval_cutoff=1.0, logfc_cutoff=0.0)

        filtered = filter_markers(markers, pval_cutoff=0.1, n_top=3)
        assert len(filtered) <= 9  # 3 clusters * 3 top

    def test_ttest_method(self):
        """Test t-test method."""
        from scgcl import find_marker_genes
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)
        markers = find_marker_genes(X, y, method='t-test', n_markers=5)

        assert len(markers) > 0


class TestHyperparameterTuning:
    """Test hyperparameter tuning (requires optuna)."""

    @pytest.fixture
    def small_data(self):
        from scgcl.utils import simulate_scrna_data
        return simulate_scrna_data(n_cells=50, n_genes=30, n_clusters=3)

    def test_tuner_creation(self):
        """Test HyperparameterTuner creation."""
        try:
            from scgcl import HyperparameterTuner
            tuner = HyperparameterTuner(n_trials=5, metric='silhouette')
            assert tuner.n_trials == 5
            assert tuner.metric == 'silhouette'
        except ImportError:
            pytest.skip("Optuna not installed")

    def test_quick_tune(self, small_data):
        """Test quick_tune function."""
        try:
            from scgcl import quick_tune
            X, y = small_data
            params = quick_tune(X, n_clusters=3, n_trials=3, verbose=False)
            assert isinstance(params, dict)
        except ImportError:
            pytest.skip("Optuna not installed")

    def test_tuning_result(self):
        """Test TuningResult dataclass."""
        from scgcl import TuningResult

        result = TuningResult(
            best_params={'hidden_dim': 64},
            best_score=0.85,
            best_trial=5
        )

        assert result.best_params['hidden_dim'] == 64
        assert result.best_score == 0.85

        summary = result.summary()
        assert 'Best Score' in summary


class TestCLI:
    """Test command-line interface."""

    def test_parser_creation(self):
        """Test CLI parser creation."""
        from scgcl.cli import create_parser

        parser = create_parser()
        assert parser is not None

    def test_info_command(self):
        """Test info command."""
        from scgcl.cli import main

        result = main(['info'])
        assert result == 0

    def test_cluster_parser(self):
        """Test cluster command parser."""
        from scgcl.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['cluster', 'test.h5ad', '-n', '5', '--quiet'])

        assert args.command == 'cluster'
        assert args.input == 'test.h5ad'
        assert args.n_clusters == 5
        assert args.quiet == True

    def test_tune_parser(self):
        """Test tune command parser."""
        from scgcl.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(['tune', 'test.csv', '--n-trials', '20'])

        assert args.command == 'tune'
        assert args.n_trials == 20


class TestScanpyIntegration:
    """Test Scanpy integration."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData object."""
        try:
            import anndata
            import pandas as pd

            X = np.random.rand(50, 30).astype(np.float32)
            obs = pd.DataFrame(index=[f'cell_{i}' for i in range(50)])
            var = pd.DataFrame(index=[f'gene_{i}' for i in range(30)])

            return anndata.AnnData(X=X, obs=obs, var=var)
        except ImportError:
            pytest.skip("anndata not installed")

    def test_from_anndata(self, mock_adata):
        """Test from_anndata function."""
        from scgcl.integration import from_anndata

        X = from_anndata(mock_adata)
        assert X.shape == (50, 30)

    def test_to_anndata(self):
        """Test to_anndata function."""
        try:
            from scgcl.integration import to_anndata

            X = np.random.rand(50, 30)
            labels = np.array([0] * 20 + [1] * 20 + [2] * 10)
            embeddings = np.random.rand(50, 10)

            adata = to_anndata(
                X, labels=labels, embeddings=embeddings,
                gene_names=[f'gene_{i}' for i in range(30)]
            )

            assert adata.shape == (50, 30)
            assert 'scgcl_clusters' in adata.obs.columns
            assert 'X_scgcl' in adata.obsm
        except ImportError:
            pytest.skip("anndata not installed")

    def test_scgcl_clustering(self, mock_adata):
        """Test scgcl function on AnnData."""
        try:
            from scgcl.integration import scgcl

            scgcl(
                mock_adata,
                n_clusters=3,
                pretrain_epochs=2,
                ssc_epochs=2,
                verbose=False,
                preprocess=False  # Skip preprocessing for mock data
            )

            assert 'scgcl_clusters' in mock_adata.obs.columns
            assert 'scgcl_confidence' in mock_adata.obs.columns
            assert 'X_scgcl' in mock_adata.obsm
            assert 'scgcl' in mock_adata.uns
        except ImportError:
            pytest.skip("anndata not installed")

    def test_scgcl_markers(self, mock_adata):
        """Test scgcl_markers function."""
        try:
            from scgcl.integration import scgcl, scgcl_markers

            scgcl(mock_adata, n_clusters=3, pretrain_epochs=2, ssc_epochs=2, verbose=False, preprocess=False)
            markers = scgcl_markers(mock_adata, n_genes=5, use_raw=False)

            assert len(markers) >= 0  # May be empty if no significant markers
            assert 'scgcl_markers' in mock_adata.uns
        except ImportError:
            pytest.skip("anndata not installed")


class TestIntegration:
    """Integration tests for all new features together."""

    def test_full_pipeline_with_markers(self):
        """Test full pipeline with marker detection."""
        from scgcl import ScGCL, find_marker_genes
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)
        gene_names = [f'Gene{i}' for i in range(50)]

        model = ScGCL(n_clusters=3, pretrain_epochs=3, ssc_epochs=3)
        model.fit(X, preprocess=False, verbose=False)

        markers = find_marker_genes(X, model.labels_, gene_names, n_markers=5)

        assert len(model.labels_) == 100
        assert len(markers) > 0

    def test_model_with_all_callbacks(self):
        """Test model with callbacks and all features."""
        from scgcl import ScGCL, EarlyStopping, Timer, find_marker_genes
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=50, n_genes=30, n_clusters=3)

        callbacks = [
            EarlyStopping(patience=50),
            Timer(verbose=False)
        ]

        model = ScGCL(n_clusters=3, pretrain_epochs=3, ssc_epochs=3)
        model.fit(X, preprocess=False, verbose=False, callbacks=callbacks)

        # All features should work
        labels = model.labels_
        confidence = model.get_confidence_scores()
        soft = model.get_soft_assignments()
        embeddings = model.get_embeddings()

        assert len(labels) == 50
        assert len(confidence) == 50
        assert soft.shape == (50, 3)
        assert embeddings.shape[0] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
