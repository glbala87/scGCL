"""Tests for analysis module: stability, visualization, export, enrichment."""

import pytest
import numpy as np
import tempfile
import os
import pandas as pd


class TestClusterStability:
    """Test cluster stability analysis."""

    @pytest.fixture
    def stability_data(self):
        """Create data for stability testing."""
        from scgcl.utils import simulate_scrna_data
        X, y = simulate_scrna_data(n_cells=100, n_genes=50, n_clusters=3)
        # Simulate embeddings
        embeddings = np.random.rand(100, 32).astype(np.float32)
        return embeddings, y

    def test_cluster_stability(self, stability_data):
        """Test cluster_stability function."""
        from scgcl import cluster_stability

        embeddings, labels = stability_data
        result = cluster_stability(
            embeddings, labels,
            n_bootstrap=10,
            sample_fraction=0.7,
            verbose=False
        )

        assert result.mean_ari >= 0
        assert result.std_ari >= 0
        assert len(result.cluster_stability) == 3
        assert len(result.cell_stability) == 100
        assert result.n_bootstrap == 10
        assert result.n_clusters == 3

    def test_stability_result_summary(self, stability_data):
        """Test StabilityResult.summary()."""
        from scgcl import cluster_stability

        embeddings, labels = stability_data
        result = cluster_stability(
            embeddings, labels,
            n_bootstrap=5,
            verbose=False
        )

        summary = result.summary()
        assert "Cluster Stability Analysis" in summary
        assert "Overall ARI" in summary
        assert "Per-cluster stability" in summary

    def test_consensus_clustering(self, stability_data):
        """Test consensus clustering."""
        from scgcl import consensus_clustering

        embeddings, _ = stability_data
        labels, consensus_matrix = consensus_clustering(
            embeddings,
            n_clusters=3,
            n_iterations=10,
            sample_fraction=0.7
        )

        assert len(labels) == 100
        assert consensus_matrix.shape == (100, 100)
        assert np.allclose(consensus_matrix, consensus_matrix.T)  # Should be symmetric

    def test_custom_clustering_func(self, stability_data):
        """Test with custom clustering function."""
        from scgcl import cluster_stability
        from sklearn.cluster import AgglomerativeClustering

        embeddings, labels = stability_data

        def custom_cluster(X):
            return AgglomerativeClustering(n_clusters=3).fit_predict(X)

        result = cluster_stability(
            embeddings, labels,
            n_bootstrap=5,
            clustering_func=custom_cluster,
            verbose=False
        )

        assert result.mean_ari >= 0


class TestVisualization:
    """Test visualization functions."""

    @pytest.fixture
    def viz_data(self):
        """Create data for visualization testing."""
        embeddings = np.random.rand(50, 32).astype(np.float32)
        labels = np.array([0]*20 + [1]*15 + [2]*15)
        confidence = np.random.rand(50)
        return embeddings, labels, confidence

    def test_silhouette_plot_returns_values(self, viz_data):
        """Test silhouette_plot returns silhouette values."""
        from scgcl import silhouette_plot
        import matplotlib
        matplotlib.use('Agg')

        embeddings, labels, _ = viz_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            result = silhouette_plot(embeddings, labels, save_path=f.name)
            os.unlink(f.name)

        assert result is not None
        assert len(result) == 50

    def test_cluster_dendrogram(self, viz_data):
        """Test cluster_dendrogram function."""
        from scgcl import cluster_dendrogram
        import matplotlib
        matplotlib.use('Agg')

        embeddings, labels, _ = viz_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            result = cluster_dendrogram(embeddings, labels, save_path=f.name)
            os.unlink(f.name)

        assert isinstance(result, dict)

    def test_cluster_heatmap(self, viz_data):
        """Test cluster_heatmap function."""
        from scgcl import cluster_heatmap
        import matplotlib
        matplotlib.use('Agg')

        embeddings, labels, _ = viz_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cluster_heatmap(embeddings, labels, n_genes=10, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_plot_confidence_distribution(self, viz_data):
        """Test plot_confidence_distribution function."""
        from scgcl import plot_confidence_distribution
        import matplotlib
        matplotlib.use('Agg')

        _, labels, confidence = viz_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_confidence_distribution(confidence, labels, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_plot_cluster_composition(self, viz_data):
        """Test plot_cluster_composition function."""
        from scgcl import plot_cluster_composition
        import matplotlib
        matplotlib.use('Agg')

        _, labels, _ = viz_data
        batch = np.array([0]*25 + [1]*25)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_cluster_composition(labels, batch=batch, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestExport:
    """Test export functions."""

    @pytest.fixture
    def export_data(self):
        """Create data for export testing."""
        X = np.random.rand(50, 30).astype(np.float32)
        labels = np.array([0]*20 + [1]*15 + [2]*15)
        embeddings = np.random.rand(50, 16)
        confidence = np.random.rand(50)
        gene_names = [f'Gene{i}' for i in range(30)]
        cell_names = [f'Cell{i}' for i in range(50)]
        return X, labels, embeddings, confidence, gene_names, cell_names

    def test_to_seurat(self, export_data):
        """Test Seurat export."""
        from scgcl import to_seurat

        X, labels, embeddings, confidence, gene_names, cell_names = export_data

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'seurat_export')
            to_seurat(
                X, labels, embeddings,
                confidence=confidence,
                gene_names=gene_names,
                cell_names=cell_names,
                output_dir=output_dir
            )

            assert os.path.exists(os.path.join(output_dir, 'counts.csv'))
            assert os.path.exists(os.path.join(output_dir, 'metadata.csv'))
            assert os.path.exists(os.path.join(output_dir, 'embeddings.csv'))
            assert os.path.exists(os.path.join(output_dir, 'load_seurat.R'))

            # Check metadata content
            metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'), index_col=0)
            assert 'scgcl_clusters' in metadata.columns
            assert 'scgcl_confidence' in metadata.columns

    def test_to_cellxgene(self, export_data):
        """Test cellxgene export."""
        try:
            import anndata
            from scgcl import to_cellxgene

            X, labels, embeddings, _, gene_names, cell_names = export_data

            adata = anndata.AnnData(X=X)
            adata.obs['scgcl_clusters'] = labels
            adata.obs_names = cell_names
            adata.var_names = gene_names

            with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as f:
                to_cellxgene(adata, output_path=f.name)
                assert os.path.exists(f.name)

                # Verify file can be read back
                adata_read = anndata.read_h5ad(f.name)
                assert adata_read.shape == (50, 30)
                os.unlink(f.name)
        except ImportError:
            pytest.skip("anndata not installed")

    def test_export_markers_to_gmt(self, export_data):
        """Test GMT export."""
        from scgcl import export_markers_to_gmt

        markers_df = pd.DataFrame({
            'cluster': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'gene': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        })

        with tempfile.NamedTemporaryFile(suffix='.gmt', delete=False, mode='w') as f:
            export_markers_to_gmt(markers_df, output_path=f.name)
            assert os.path.exists(f.name)

            # Read and verify
            with open(f.name, 'r') as gmt:
                lines = gmt.readlines()
                assert len(lines) == 3  # 3 clusters
                assert 'Cluster_0' in lines[0]
            os.unlink(f.name)


class TestEnrichment:
    """Test gene set enrichment analysis."""

    @pytest.fixture
    def enrichment_data(self):
        """Create data for enrichment testing."""
        markers_df = pd.DataFrame({
            'cluster': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            'gene': ['CDK1', 'CDK2', 'CCNA2', 'CCNB1', 'MCM2',
                    'BCL2', 'BAX', 'CASP3', 'TP53', 'FAS']
        })
        return markers_df

    def test_load_gene_sets_builtin(self):
        """Test loading built-in gene sets."""
        from scgcl import load_gene_sets

        gene_sets = load_gene_sets('go_bp')
        assert isinstance(gene_sets, dict)
        assert len(gene_sets) > 0

    def test_load_gmt(self):
        """Test loading GMT file."""
        from scgcl import load_gmt

        gmt_content = "Term1\tdesc\tGene1\tGene2\tGene3\nTerm2\tdesc\tGene4\tGene5"

        with tempfile.NamedTemporaryFile(suffix='.gmt', delete=False, mode='w') as f:
            f.write(gmt_content)
            f.flush()

            gene_sets = load_gmt(f.name)
            assert 'Term1' in gene_sets
            assert 'Term2' in gene_sets
            assert gene_sets['Term1'] == ['Gene1', 'Gene2', 'Gene3']
            os.unlink(f.name)

    def test_enrich_function(self):
        """Test basic enrichment function."""
        from scgcl import enrich, load_gene_sets

        genes = ['CDK1', 'CDK2', 'CCNA2', 'CCNB1', 'MCM2', 'MCM3']
        gene_sets = load_gene_sets('go_bp')

        results = enrich(genes, gene_sets, pval_cutoff=1.0)

        # May have results if genes overlap with gene sets
        assert isinstance(results, list)

    def test_cluster_enrichment(self, enrichment_data):
        """Test cluster_enrichment function."""
        from scgcl import cluster_enrichment

        result = cluster_enrichment(
            enrichment_data,
            source='go_bp',
            pval_cutoff=1.0,
            min_overlap=2
        )

        assert isinstance(result, pd.DataFrame)

    def test_enrichment_result_dataclass(self):
        """Test EnrichmentResult dataclass."""
        from scgcl import EnrichmentResult

        result = EnrichmentResult(
            term='Cell Cycle',
            cluster=0,
            pvalue=0.001,
            adjusted_pvalue=0.01,
            odds_ratio=5.0,
            overlap_genes=['CDK1', 'CDK2'],
            overlap_count=2,
            term_size=10,
            query_size=5,
            source='go_bp'
        )

        assert result.term == 'Cell Cycle'
        d = result.to_dict()
        assert d['cluster'] == 0
        assert d['overlap_genes'] == 'CDK1,CDK2'

    def test_quick_enrich(self, enrichment_data):
        """Test quick_enrich function."""
        from scgcl import quick_enrich

        result = quick_enrich(
            enrichment_data,
            source='go_bp',
            top_n=3,
            verbose=False
        )

        assert isinstance(result, pd.DataFrame)


class TestAnalysisIntegration:
    """Integration tests combining analysis features."""

    def test_full_analysis_pipeline(self):
        """Test full analysis pipeline."""
        from scgcl import ScGCL, find_marker_genes, cluster_stability, to_seurat
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=80, n_genes=50, n_clusters=3)
        gene_names = [f'Gene{i}' for i in range(50)]

        # Fit model
        model = ScGCL(n_clusters=3, pretrain_epochs=3, ssc_epochs=3)
        model.fit(X, preprocess=False, verbose=False)

        labels = model.labels_
        embeddings = model.get_embeddings()
        confidence = model.get_confidence_scores()

        # Find markers
        markers = find_marker_genes(X, labels, gene_names, n_markers=5)
        assert len(markers) > 0

        # Cluster stability
        stability = cluster_stability(
            embeddings, labels,
            n_bootstrap=5,
            verbose=False
        )
        assert stability.mean_ari >= 0

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            to_seurat(
                X, labels, embeddings,
                confidence=confidence,
                gene_names=gene_names,
                output_dir=os.path.join(tmpdir, 'seurat')
            )
            assert os.path.exists(os.path.join(tmpdir, 'seurat', 'counts.csv'))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
