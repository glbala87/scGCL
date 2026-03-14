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
        # Check that diagonal is all 1s (cell always co-clusters with itself)
        assert np.allclose(np.diag(consensus_matrix), 1.0)
        # Check values are in valid range [0, 1] (with tolerance for floating point)
        assert consensus_matrix.min() >= -1e-10 and consensus_matrix.max() <= 1 + 1e-10

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


class TestRefinement:
    """Test subclustering and cluster merging."""

    @pytest.fixture
    def refinement_data(self):
        """Create data for refinement testing."""
        from scgcl.utils import simulate_scrna_data
        X, y = simulate_scrna_data(n_cells=150, n_genes=50, n_clusters=3)
        embeddings = np.random.rand(150, 32).astype(np.float32)
        return X, y, embeddings

    def test_subcluster_kmeans(self, refinement_data):
        """Test basic subclustering with kmeans."""
        from scgcl import subcluster

        X, labels, embeddings = refinement_data

        result = subcluster(
            X, labels,
            cluster_id=0,
            n_subclusters=2,
            method='kmeans',
            embeddings=embeddings
        )

        assert result.original_cluster == 0
        assert result.n_subclusters == 2
        assert len(result.subcluster_labels) == np.sum(labels == 0)
        assert len(result.full_labels) == len(labels)
        assert len(result.subcluster_sizes) == 2

    def test_subcluster_hierarchical(self, refinement_data):
        """Test subclustering with hierarchical method."""
        from scgcl import subcluster

        X, labels, embeddings = refinement_data

        result = subcluster(
            X, labels,
            cluster_id=1,
            n_subclusters=2,
            method='hierarchical',
            embeddings=embeddings
        )

        assert result.n_subclusters == 2
        assert len(np.unique(result.full_labels)) >= 3  # Original 3 + 1 new

    def test_subcluster_result_summary(self, refinement_data):
        """Test SubclusterResult.summary()."""
        from scgcl import subcluster

        X, labels, embeddings = refinement_data

        result = subcluster(X, labels, cluster_id=0, n_subclusters=2, embeddings=embeddings)
        summary = result.summary()

        assert "Subclustering Results" in summary
        assert "Cluster 0" in summary
        assert "Subcluster sizes" in summary

    def test_subcluster_recursive(self, refinement_data):
        """Test recursive subclustering."""
        from scgcl import subcluster_recursive

        X, labels, embeddings = refinement_data

        final_labels = subcluster_recursive(
            X, labels,
            min_cluster_size=20,
            max_depth=2,
            n_subclusters=2,
            embeddings=embeddings,
            verbose=False
        )

        assert len(final_labels) == len(labels)
        assert len(np.unique(final_labels)) >= len(np.unique(labels))

    def test_merge_clusters_by_threshold(self, refinement_data):
        """Test merging clusters by distance threshold."""
        from scgcl import merge_clusters

        _, labels, embeddings = refinement_data

        result = merge_clusters(
            labels, embeddings,
            threshold=5.0,  # Large threshold to ensure some merging
            method='centroid'
        )

        assert result.original_n_clusters == 3
        assert result.merged_n_clusters <= 3
        assert len(result.merged_labels) == len(labels)
        assert isinstance(result.cluster_mapping, dict)

    def test_merge_clusters_by_count(self, refinement_data):
        """Test merging clusters to target count."""
        from scgcl import merge_clusters

        _, labels, embeddings = refinement_data

        result = merge_clusters(
            labels, embeddings,
            n_clusters=2,  # Merge to 2 clusters
            method='centroid'
        )

        assert result.merged_n_clusters == 2
        assert len(np.unique(result.merged_labels)) == 2

    def test_merge_result_summary(self, refinement_data):
        """Test MergeResult.summary()."""
        from scgcl import merge_clusters

        _, labels, embeddings = refinement_data

        result = merge_clusters(labels, embeddings, n_clusters=2)
        summary = result.summary()

        assert "Cluster Merge Results" in summary
        assert "Original clusters" in summary
        assert "Merged clusters" in summary

    def test_auto_merge(self, refinement_data):
        """Test automatic merging of small clusters."""
        from scgcl import auto_merge

        _, labels, embeddings = refinement_data

        # Create a small cluster artificially
        small_labels = labels.copy()
        small_labels[:5] = 99  # Very small cluster

        result = auto_merge(
            small_labels, embeddings,
            min_cluster_size=10
        )

        # Small cluster should be merged
        assert 99 not in result.merged_labels

    def test_merge_by_markers(self, refinement_data):
        """Test merging by marker gene similarity."""
        from scgcl import merge_by_markers

        X, labels, _ = refinement_data

        result = merge_by_markers(
            X, labels,
            correlation_threshold=0.5,
            n_top_markers=10
        )

        assert result.merged_n_clusters <= result.original_n_clusters
        assert len(result.merged_labels) == len(labels)

    def test_plot_merge_dendrogram(self, refinement_data):
        """Test merge dendrogram plotting."""
        from scgcl import plot_merge_dendrogram
        import matplotlib
        matplotlib.use('Agg')

        _, labels, embeddings = refinement_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_merge_dendrogram(
                labels, embeddings,
                threshold=2.0,
                save_path=f.name
            )
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_subcluster_then_merge(self, refinement_data):
        """Test subclustering followed by merging."""
        from scgcl import subcluster, merge_clusters

        X, labels, embeddings = refinement_data

        # First subcluster
        sub_result = subcluster(
            X, labels,
            cluster_id=0,
            n_subclusters=3,
            embeddings=embeddings
        )

        # Then merge back
        merge_result = merge_clusters(
            sub_result.full_labels, embeddings,
            n_clusters=3  # Back to original count
        )

        assert merge_result.merged_n_clusters == 3


class TestAnnotation:
    """Test cell type annotation."""

    @pytest.fixture
    def annotation_data(self):
        """Create data with known marker genes for testing."""
        # Create expression matrix with marker-like patterns
        np.random.seed(42)
        n_cells = 150
        n_genes = 100

        # Gene names including some known markers
        gene_names = ['CD3D', 'CD3E', 'CD4', 'CD8A', 'NKG7', 'GNLY',
                     'CD19', 'MS4A1', 'CD14', 'LYZ'] + [f'Gene{i}' for i in range(90)]

        X = np.random.rand(n_cells, n_genes).astype(np.float32) * 0.1

        # Create 3 clusters with different marker expression
        labels = np.array([0]*50 + [1]*50 + [2]*50)

        # Cluster 0: T cell markers high
        X[:50, 0:4] = np.random.rand(50, 4) * 2 + 1  # CD3D, CD3E, CD4, CD8A

        # Cluster 1: NK cell markers high
        X[50:100, 4:6] = np.random.rand(50, 2) * 2 + 1  # NKG7, GNLY

        # Cluster 2: B cell markers high
        X[100:150, 6:8] = np.random.rand(50, 2) * 2 + 1  # CD19, MS4A1

        return X, labels, gene_names

    def test_annotate_clusters(self, annotation_data):
        """Test basic cluster annotation."""
        from scgcl import annotate_clusters

        X, labels, gene_names = annotation_data

        result = annotate_clusters(
            X, labels, gene_names,
            tissue='pbmc',
            verbose=False
        )

        assert len(result.cluster_annotations) == 3
        assert len(result.cell_types) == 150
        assert len(result.cell_confidence) == 150

    def test_annotation_result_summary(self, annotation_data):
        """Test AnnotationResult.summary()."""
        from scgcl import annotate_clusters

        X, labels, gene_names = annotation_data

        result = annotate_clusters(X, labels, gene_names, verbose=False)
        summary = result.summary()

        assert "Cell Type Annotation Results" in summary
        assert "Cluster Annotations" in summary

    def test_annotation_result_to_dataframe(self, annotation_data):
        """Test AnnotationResult.to_dataframe()."""
        from scgcl import annotate_clusters

        X, labels, gene_names = annotation_data

        result = annotate_clusters(X, labels, gene_names, verbose=False)
        df = result.to_dataframe()

        assert len(df) == 3
        assert 'cluster' in df.columns
        assert 'cell_type' in df.columns
        assert 'confidence' in df.columns

    def test_get_marker_database(self):
        """Test getting marker databases."""
        from scgcl import get_marker_database, PBMC_MARKERS, BRAIN_MARKERS

        pbmc = get_marker_database('pbmc')
        assert 'CD4+ T cells' in pbmc or 'T cells' in pbmc

        brain = get_marker_database('brain')
        assert len(brain) > 0

        all_markers = get_marker_database('all')
        assert len(all_markers) > len(pbmc)

    def test_custom_markers(self, annotation_data):
        """Test annotation with custom markers."""
        from scgcl import annotate_clusters

        X, labels, gene_names = annotation_data

        custom_markers = {
            'Type A': ['CD3D', 'CD3E', 'CD4'],
            'Type B': ['NKG7', 'GNLY'],
            'Type C': ['CD19', 'MS4A1']
        }

        result = annotate_clusters(
            X, labels, gene_names,
            custom_markers=custom_markers,
            verbose=False
        )

        assert len(result.cluster_annotations) == 3

    def test_quick_annotate(self, annotation_data):
        """Test quick_annotate function."""
        from scgcl import quick_annotate

        X, labels, gene_names = annotation_data

        annotations = quick_annotate(X, labels, gene_names, tissue='pbmc')

        assert isinstance(annotations, dict)
        assert len(annotations) == 3

    def test_score_cell_types(self, annotation_data):
        """Test score_cell_types function."""
        from scgcl import score_cell_types, PBMC_MARKERS

        X, labels, gene_names = annotation_data

        scores, marker_expr = score_cell_types(
            X, labels, gene_names,
            markers=PBMC_MARKERS,
            method='mean'
        )

        assert len(scores) == 3
        assert isinstance(marker_expr, pd.DataFrame)

    def test_plot_annotation(self, annotation_data):
        """Test plot_annotation function."""
        from scgcl import annotate_clusters, plot_annotation
        import matplotlib
        matplotlib.use('Agg')

        X, labels, gene_names = annotation_data
        result = annotate_clusters(X, labels, gene_names, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_annotation(result, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_plot_marker_heatmap(self, annotation_data):
        """Test plot_marker_heatmap function."""
        from scgcl import annotate_clusters, plot_marker_heatmap
        import matplotlib
        matplotlib.use('Agg')

        X, labels, gene_names = annotation_data
        result = annotate_clusters(X, labels, gene_names, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_marker_heatmap(result, top_n=3, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_different_tissues(self, annotation_data):
        """Test annotation with different tissue types."""
        from scgcl import annotate_clusters

        X, labels, gene_names = annotation_data

        for tissue in ['pbmc', 'brain', 'immune', 'tumor']:
            result = annotate_clusters(
                X, labels, gene_names,
                tissue=tissue,
                verbose=False
            )
            assert len(result.cluster_annotations) == 3

    def test_different_methods(self, annotation_data):
        """Test different scoring methods."""
        from scgcl import annotate_clusters

        X, labels, gene_names = annotation_data

        for method in ['mean', 'percent', 'zscore']:
            result = annotate_clusters(
                X, labels, gene_names,
                method=method,
                verbose=False
            )
            assert len(result.cluster_annotations) == 3


class TestInteractive:
    """Test interactive visualization."""

    @pytest.fixture
    def interactive_data(self):
        """Create data for interactive testing."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 2).astype(np.float32)
        labels = np.array([0]*40 + [1]*30 + [2]*30)
        confidence = np.random.rand(100)
        expression = np.random.rand(100, 50).astype(np.float32)
        gene_names = [f'Gene{i}' for i in range(50)]
        return embeddings, labels, confidence, expression, gene_names

    def test_interactive_umap(self, interactive_data):
        """Test interactive_umap function."""
        pytest.importorskip("plotly")
        from scgcl import interactive_umap

        embeddings, labels, confidence, _, _ = interactive_data

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            fig = interactive_umap(
                embeddings, labels,
                hover_data={'Confidence': confidence},
                save_path=f.name
            )
            assert os.path.exists(f.name)
            assert os.path.getsize(f.name) > 0
            os.unlink(f.name)

    def test_interactive_umap_continuous(self, interactive_data):
        """Test interactive_umap with continuous coloring."""
        pytest.importorskip("plotly")
        from scgcl import interactive_umap

        embeddings, _, confidence, _, _ = interactive_data

        fig = interactive_umap(
            embeddings,
            color_by=confidence,
            color_label='Confidence'
        )
        assert fig is not None

    def test_interactive_3d(self, interactive_data):
        """Test interactive_3d function."""
        pytest.importorskip("plotly")
        from scgcl import interactive_3d

        embeddings_3d = np.random.rand(100, 3).astype(np.float32)
        labels = interactive_data[1]

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            fig = interactive_3d(embeddings_3d, labels, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_interactive_embedding(self, interactive_data):
        """Test interactive_embedding with dimensionality reduction."""
        pytest.importorskip("plotly")
        from scgcl import interactive_embedding

        high_dim = np.random.rand(100, 32).astype(np.float32)
        labels = interactive_data[1]

        fig = interactive_embedding(
            high_dim, labels,
            method='pca',
            n_components=2
        )
        assert fig is not None

    def test_interactive_gene_expression(self, interactive_data):
        """Test interactive_gene_expression function."""
        pytest.importorskip("plotly")
        from scgcl import interactive_gene_expression

        embeddings, labels, _, expression, gene_names = interactive_data

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            fig = interactive_gene_expression(
                embeddings, expression, gene_names,
                genes=['Gene0', 'Gene1', 'Gene2', 'Gene3'],
                save_path=f.name
            )
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_interactive_comparison(self, interactive_data):
        """Test interactive_comparison function."""
        pytest.importorskip("plotly")
        from scgcl import interactive_comparison

        embeddings, labels, _, _, _ = interactive_data
        labels2 = np.array([0]*50 + [1]*50)  # Different clustering

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            fig = interactive_comparison(
                embeddings, labels, labels2,
                title1="Clustering A",
                title2="Clustering B",
                save_path=f.name
            )
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_interactive_cluster_composition(self, interactive_data):
        """Test interactive_cluster_composition function."""
        pytest.importorskip("plotly")
        from scgcl import interactive_cluster_composition

        _, labels, _, _, _ = interactive_data
        metadata = pd.DataFrame({
            'batch': ['A']*50 + ['B']*50
        })

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            fig = interactive_cluster_composition(
                labels, metadata, group_by='batch',
                save_path=f.name
            )
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_interactive_violin(self, interactive_data):
        """Test interactive_violin function."""
        pytest.importorskip("plotly")
        from scgcl import interactive_violin

        _, labels, confidence, _, _ = interactive_data

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            fig = interactive_violin(
                confidence, labels,
                title="Confidence by Cluster",
                save_path=f.name
            )
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_create_dashboard(self, interactive_data):
        """Test create_dashboard function."""
        pytest.importorskip("plotly")
        from scgcl import create_dashboard

        embeddings, labels, confidence, expression, gene_names = interactive_data

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = create_dashboard(
                embeddings, labels,
                expression=expression,
                gene_names=gene_names,
                confidence=confidence,
                marker_genes=['Gene0'],
                save_path=f.name
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            os.unlink(f.name)


class TestDifferentialAbundance:
    """Test differential abundance analysis."""

    @pytest.fixture
    def da_data(self):
        """Create data for differential abundance testing."""
        np.random.seed(42)
        # 100 cells in 3 clusters
        labels = np.array([0]*40 + [1]*30 + [2]*30)
        # Two conditions with different cluster proportions
        condition = np.array(['A']*50 + ['B']*50)
        return labels, condition

    def test_differential_abundance_fisher(self, da_data):
        """Test differential abundance with Fisher's exact test."""
        from scgcl import differential_abundance

        labels, condition = da_data
        result = differential_abundance(
            labels, condition,
            method='fisher',
            min_cells=5
        )

        assert isinstance(result, pd.DataFrame)
        assert 'cluster' in result.columns
        assert 'log2FC' in result.columns
        assert 'pvalue' in result.columns
        assert 'padj' in result.columns

    def test_differential_abundance_chi2(self, da_data):
        """Test differential abundance with chi-squared test."""
        from scgcl import differential_abundance

        labels, condition = da_data
        result = differential_abundance(
            labels, condition,
            method='chi2',
            min_cells=5
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # May have 0 clusters passing filter

    def test_differential_abundance_permutation(self, da_data):
        """Test differential abundance with permutation test."""
        from scgcl import differential_abundance

        labels, condition = da_data
        result = differential_abundance(
            labels, condition,
            method='permutation',
            min_cells=5
        )

        assert isinstance(result, pd.DataFrame)

    def test_differential_abundance_result_dataclass(self):
        """Test DifferentialAbundanceResult dataclass."""
        from scgcl import DifferentialAbundanceResult

        result = DifferentialAbundanceResult(
            cluster=0,
            condition1_count=30,
            condition2_count=20,
            condition1_proportion=0.6,
            condition2_proportion=0.4,
            log2_fold_change=-0.58,
            pvalue=0.05,
            adjusted_pvalue=0.1
        )

        d = result.to_dict()
        assert d['cluster'] == 0
        assert d['log2FC'] == -0.58

    def test_plot_differential_abundance(self, da_data):
        """Test volcano plot."""
        from scgcl import differential_abundance, plot_differential_abundance
        import matplotlib
        matplotlib.use('Agg')

        labels, condition = da_data
        da_results = differential_abundance(labels, condition, min_cells=5)

        if len(da_results) > 0:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plot_differential_abundance(da_results, save_path=f.name)
                assert os.path.exists(f.name)
                os.unlink(f.name)

    def test_abundance_barplot(self, da_data):
        """Test abundance barplot."""
        from scgcl import abundance_barplot
        import matplotlib
        matplotlib.use('Agg')

        labels, condition = da_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            abundance_barplot(labels, condition, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestCellCycle:
    """Test cell cycle scoring."""

    @pytest.fixture
    def cell_cycle_data(self):
        """Create data with cell cycle genes."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 150

        # Gene names including some S and G2M phase genes
        from scgcl import S_PHASE_GENES, G2M_PHASE_GENES
        gene_names = list(S_PHASE_GENES[:20]) + list(G2M_PHASE_GENES[:20]) + [f'Gene{i}' for i in range(110)]

        X = np.random.rand(n_cells, n_genes).astype(np.float32)

        # Make some cells express S phase genes
        X[:30, :20] = np.random.rand(30, 20) * 2 + 1

        # Make some cells express G2M phase genes
        X[30:60, 20:40] = np.random.rand(30, 20) * 2 + 1

        return X, gene_names

    def test_score_cell_cycle(self, cell_cycle_data):
        """Test basic cell cycle scoring."""
        from scgcl import score_cell_cycle

        X, gene_names = cell_cycle_data
        result = score_cell_cycle(X, np.array(gene_names))

        assert len(result.s_scores) == 100
        assert len(result.g2m_scores) == 100
        assert len(result.phase) == 100
        # Convert to string for comparison since numpy array may have different string type
        valid_phases = {'G1', 'S', 'G2M'}
        assert all(str(p) in valid_phases for p in result.phase)

    def test_cell_cycle_result_dataframe(self, cell_cycle_data):
        """Test CellCycleResult.to_dataframe()."""
        from scgcl import score_cell_cycle

        X, gene_names = cell_cycle_data
        result = score_cell_cycle(X, np.array(gene_names))
        df = result.to_dataframe()

        assert 'S_score' in df.columns
        assert 'G2M_score' in df.columns
        assert 'phase' in df.columns

    def test_cell_cycle_result_summary(self, cell_cycle_data):
        """Test CellCycleResult.summary()."""
        from scgcl import score_cell_cycle

        X, gene_names = cell_cycle_data
        result = score_cell_cycle(X, np.array(gene_names))
        summary = result.summary()

        assert 'n_cells' in summary
        assert 'phase_counts' in summary
        assert 's_genes_found' in summary

    def test_plot_cell_cycle(self, cell_cycle_data):
        """Test cell cycle plotting."""
        from scgcl import score_cell_cycle, plot_cell_cycle
        import matplotlib
        matplotlib.use('Agg')

        X, gene_names = cell_cycle_data
        result = score_cell_cycle(X, np.array(gene_names))
        embedding = np.random.rand(100, 2)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_cell_cycle(result, embedding=embedding, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_regress_cell_cycle(self, cell_cycle_data):
        """Test regressing out cell cycle effects."""
        from scgcl import score_cell_cycle, regress_cell_cycle

        X, gene_names = cell_cycle_data
        result = score_cell_cycle(X, np.array(gene_names))

        corrected = regress_cell_cycle(
            X, result.s_scores, result.g2m_scores
        )

        assert corrected.shape == X.shape


class TestDoubletDetection:
    """Test doublet detection."""

    @pytest.fixture
    def doublet_data(self):
        """Create data for doublet detection testing."""
        np.random.seed(42)
        n_cells = 100
        n_dims = 30

        # PCA-like embeddings
        X = np.random.rand(n_cells, n_dims).astype(np.float32)
        return X

    def test_detect_doublets(self, doublet_data):
        """Test basic doublet detection."""
        from scgcl import detect_doublets

        result = detect_doublets(
            doublet_data,
            expected_doublet_rate=0.05,
            n_neighbors=20
        )

        assert len(result.doublet_scores) == 100
        assert len(result.predicted_doublets) == 100
        assert result.doublet_rate >= 0 and result.doublet_rate <= 1
        assert result.threshold > 0

    def test_doublet_result_dataframe(self, doublet_data):
        """Test DoubletResult.to_dataframe()."""
        from scgcl import detect_doublets

        result = detect_doublets(doublet_data)
        df = result.to_dataframe()

        assert 'doublet_score' in df.columns
        assert 'is_doublet' in df.columns

    def test_doublet_result_summary(self, doublet_data):
        """Test DoubletResult.summary()."""
        from scgcl import detect_doublets

        result = detect_doublets(doublet_data)
        summary = result.summary()

        assert 'n_cells' in summary
        assert 'n_doublets' in summary
        assert 'doublet_rate' in summary
        assert 'threshold' in summary

    def test_detect_doublets_scrublet(self):
        """Test Scrublet-like detection."""
        from scgcl import detect_doublets_scrublet

        np.random.seed(42)
        counts = np.random.poisson(5, (100, 50)).astype(np.float32)

        result = detect_doublets_scrublet(
            counts,
            expected_doublet_rate=0.05,
            min_counts=1,
            min_cells=1
        )

        assert len(result.doublet_scores) == 100

    def test_plot_doublet_scores(self, doublet_data):
        """Test doublet score plotting."""
        from scgcl import detect_doublets, plot_doublet_scores
        import matplotlib
        matplotlib.use('Agg')

        result = detect_doublets(doublet_data)
        embedding = np.random.rand(100, 2)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_doublet_scores(result, embedding=embedding, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_filter_doublets(self, doublet_data):
        """Test filtering out doublets."""
        from scgcl import detect_doublets, filter_doublets

        result = detect_doublets(doublet_data)
        filtered = filter_doublets(doublet_data, result)

        expected_singlets = 100 - result.n_doublets
        assert len(filtered) == expected_singlets


class TestClusterQC:
    """Test cluster QC metrics and batch visualization."""

    @pytest.fixture
    def qc_data(self):
        """Create data for QC testing."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        X = np.random.rand(n_cells, n_genes).astype(np.float32) * 100
        labels = np.array([0]*40 + [1]*30 + [2]*30)
        batch = np.array(['A']*50 + ['B']*50)
        gene_names = ['MT-' + str(i) for i in range(5)] + ['RPL' + str(i) for i in range(5)] + [f'Gene{i}' for i in range(40)]

        return X, labels, batch, gene_names

    def test_compute_cluster_qc(self, qc_data):
        """Test cluster QC computation."""
        from scgcl import compute_cluster_qc

        X, labels, _, gene_names = qc_data
        result = compute_cluster_qc(X, labels, np.array(gene_names))

        assert len(result.cluster_stats) == 3
        assert len(result.cell_stats) == 100
        assert 'n_cells' in result.overall_stats

    def test_cluster_qc_with_mt_genes(self, qc_data):
        """Test QC with mitochondrial genes."""
        from scgcl import compute_cluster_qc

        X, labels, _, gene_names = qc_data
        result = compute_cluster_qc(X, labels, np.array(gene_names))

        assert 'pct_mt' in result.cell_stats.columns
        assert 'mean_pct_mt' in result.cluster_stats.columns

    def test_compute_cluster_purity(self, qc_data):
        """Test cluster purity computation."""
        from scgcl import compute_cluster_purity

        _, labels, batch, _ = qc_data
        result = compute_cluster_purity(labels, batch)

        assert len(result) == 3
        assert 'purity' in result.columns
        assert all(result['purity'] >= 0) and all(result['purity'] <= 1)

    def test_compute_batch_mixing(self, qc_data):
        """Test batch mixing computation."""
        from scgcl import compute_batch_mixing

        embedding = np.random.rand(100, 2)
        _, _, batch, _ = qc_data

        result = compute_batch_mixing(embedding, batch, n_neighbors=20)

        assert 'overall_mixing' in result
        assert 'mixing_scores' in result
        assert len(result['mixing_scores']) == 100

    def test_plot_cluster_qc(self, qc_data):
        """Test cluster QC plotting."""
        from scgcl import compute_cluster_qc, plot_cluster_qc
        import matplotlib
        matplotlib.use('Agg')

        X, labels, _, gene_names = qc_data
        result = compute_cluster_qc(X, labels, np.array(gene_names))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_cluster_qc(result, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_plot_batch_distribution(self, qc_data):
        """Test batch distribution plotting."""
        from scgcl import plot_batch_distribution
        import matplotlib
        matplotlib.use('Agg')

        _, labels, batch, _ = qc_data

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_batch_distribution(labels, batch, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_plot_batch_umap(self, qc_data):
        """Test batch UMAP plotting."""
        from scgcl import plot_batch_umap
        import matplotlib
        matplotlib.use('Agg')

        _, labels, batch, _ = qc_data
        embedding = np.random.rand(100, 2)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_batch_umap(embedding, batch, labels=labels, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_batch_effect_test(self, qc_data):
        """Test batch effect statistical test."""
        from scgcl import batch_effect_test

        embedding = np.random.rand(100, 2)
        _, labels, batch, _ = qc_data

        result = batch_effect_test(embedding, batch, labels, n_permutations=100)

        assert 'statistic' in result
        assert 'pvalue' in result
        assert 'significant' in result


class TestHTMLReport:
    """Test HTML report generation."""

    @pytest.fixture
    def report_data(self):
        """Create data for report testing."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        labels = np.array([0]*40 + [1]*30 + [2]*30)
        embedding = np.random.rand(n_cells, 2)
        gene_names = [f'Gene{i}' for i in range(n_genes)]

        return X, labels, embedding, gene_names

    def test_html_report_generator(self):
        """Test HTMLReportGenerator class."""
        from scgcl import HTMLReportGenerator

        report = HTMLReportGenerator(title="Test Report")
        report.add_summary({'Cells': 100, 'Clusters': 3})
        report.add_text("Section 1", "This is a test section")

        html = report.generate()

        assert "Test Report" in html
        assert "Cells" in html
        assert "This is a test section" in html

    def test_report_add_table(self):
        """Test adding tables to report."""
        from scgcl import HTMLReportGenerator

        report = HTMLReportGenerator()
        df = pd.DataFrame({
            'Cluster': [0, 1, 2],
            'Size': [40, 30, 30]
        })
        report.add_table("Cluster Sizes", df)

        html = report.generate()
        assert "Cluster Sizes" in html

    def test_report_add_figure(self, report_data):
        """Test adding figures to report."""
        from scgcl import HTMLReportGenerator
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        report = HTMLReportGenerator()

        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [1, 2, 3])

        report.add_figure("Test Plot", fig)
        html = report.generate()

        assert "Test Plot" in html
        assert "data:image/png;base64" in html

    def test_generate_clustering_report(self, report_data):
        """Test generate_clustering_report function."""
        from scgcl import generate_clustering_report

        X, labels, embedding, gene_names = report_data

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = generate_clustering_report(
                X, labels,
                embedding=embedding,
                gene_names=np.array(gene_names),
                title="Test Clustering Report",
                output_path=f.name
            )

            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            # Check content
            with open(path, 'r') as rf:
                content = rf.read()
                assert "Test Clustering Report" in content
                assert "Summary Statistics" in content

            os.unlink(f.name)

    def test_report_with_metrics(self, report_data):
        """Test report with clustering metrics."""
        from scgcl import generate_clustering_report

        X, labels, embedding, gene_names = report_data
        metrics = {'ARI': 0.85, 'NMI': 0.78, 'Silhouette': 0.42}

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = generate_clustering_report(
                X, labels,
                embedding=embedding,
                metrics=metrics,
                output_path=f.name
            )

            with open(path, 'r') as rf:
                content = rf.read()
                assert "ARI" in content or "Ari" in content

            os.unlink(f.name)

    def test_generate_comparison_report(self, report_data):
        """Test comparison report generation."""
        from scgcl import generate_comparison_report

        _, labels, embedding, _ = report_data
        labels2 = np.array([0]*50 + [1]*50)

        results = [
            {'ARI': 0.85, 'NMI': 0.78},
            {'ARI': 0.72, 'NMI': 0.65}
        ]

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = generate_comparison_report(
                results,
                [labels, labels2],
                ['Method A', 'Method B'],
                embedding=embedding,
                output_path=f.name
            )

            assert os.path.exists(path)
            os.unlink(f.name)


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

    def test_qc_to_report_pipeline(self):
        """Test QC to report pipeline."""
        from scgcl import compute_cluster_qc, generate_clustering_report
        from scgcl.utils import simulate_scrna_data

        X, labels = simulate_scrna_data(n_cells=80, n_genes=50, n_clusters=3)
        gene_names = [f'Gene{i}' for i in range(50)]

        # Compute QC
        qc_result = compute_cluster_qc(X, labels, np.array(gene_names))

        # Generate report
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = generate_clustering_report(
                X, labels,
                gene_names=np.array(gene_names),
                metrics=qc_result.overall_stats,
                output_path=f.name
            )
            assert os.path.exists(path)
            os.unlink(f.name)


class TestTrajectory:
    """Test trajectory analysis."""

    @pytest.fixture
    def trajectory_data(self):
        """Create data for trajectory testing."""
        np.random.seed(42)
        n_cells = 100

        # Create a simple trajectory in embedding space
        t = np.linspace(0, 1, n_cells)
        embedding = np.zeros((n_cells, 2))
        embedding[:, 0] = t + np.random.randn(n_cells) * 0.1
        embedding[:, 1] = np.sin(t * np.pi) + np.random.randn(n_cells) * 0.1

        labels = np.array([0]*30 + [1]*40 + [2]*30)

        return embedding, labels

    def test_diffusion_pseudotime(self, trajectory_data):
        """Test diffusion pseudotime."""
        from scgcl import diffusion_pseudotime

        embedding, _ = trajectory_data
        result = diffusion_pseudotime(embedding, root=0)

        assert len(result.pseudotime) == 100
        assert result.pseudotime.min() >= 0
        assert result.pseudotime.max() <= 1
        assert result.root_cell == 0
        assert result.method == 'diffusion_pseudotime'

    def test_diffusion_pseudotime_auto_root(self, trajectory_data):
        """Test DPT with auto root detection."""
        from scgcl import diffusion_pseudotime

        embedding, _ = trajectory_data
        result = diffusion_pseudotime(embedding)

        assert len(result.pseudotime) == 100
        assert result.root_cell >= 0

    def test_principal_curve(self, trajectory_data):
        """Test principal curve fitting."""
        from scgcl import principal_curve

        embedding, _ = trajectory_data
        pseudotime, curve = principal_curve(embedding, n_points=50, max_iter=10)

        assert len(pseudotime) == 100
        assert curve.shape == (50, 2)

    def test_slingshot(self, trajectory_data):
        """Test slingshot trajectory inference."""
        from scgcl import slingshot

        embedding, labels = trajectory_data
        result = slingshot(embedding, labels, start_cluster=0)

        assert len(result.pseudotime) == 100
        assert result.branch_labels is not None
        assert result.n_branches >= 1

    def test_infer_trajectory_dpt(self, trajectory_data):
        """Test unified trajectory interface with DPT."""
        from scgcl import infer_trajectory

        embedding, _ = trajectory_data
        result = infer_trajectory(embedding, method='dpt', root=0)

        assert len(result.pseudotime) == 100
        assert result.method == 'diffusion_pseudotime'

    def test_infer_trajectory_slingshot(self, trajectory_data):
        """Test unified trajectory interface with slingshot."""
        from scgcl import infer_trajectory

        embedding, labels = trajectory_data
        result = infer_trajectory(embedding, labels, method='slingshot')

        assert len(result.pseudotime) == 100

    def test_trajectory_result_dataframe(self, trajectory_data):
        """Test TrajectoryResult.to_dataframe()."""
        from scgcl import diffusion_pseudotime

        embedding, _ = trajectory_data
        result = diffusion_pseudotime(embedding)
        df = result.to_dataframe()

        assert 'pseudotime' in df.columns
        assert len(df) == 100

    def test_trajectory_result_summary(self, trajectory_data):
        """Test TrajectoryResult.summary()."""
        from scgcl import diffusion_pseudotime

        embedding, _ = trajectory_data
        result = diffusion_pseudotime(embedding)
        summary = result.summary()

        assert "Trajectory Analysis Results" in summary
        assert "Pseudotime range" in summary

    def test_paga(self, trajectory_data):
        """Test PAGA graph abstraction."""
        from scgcl import paga

        embedding, labels = trajectory_data
        connectivity, edges = paga(embedding, labels, n_neighbors=10)

        assert connectivity.shape == (3, 3)
        assert isinstance(edges, pd.DataFrame)

    def test_find_trajectory_genes(self, trajectory_data):
        """Test finding trajectory-associated genes."""
        from scgcl import diffusion_pseudotime, find_trajectory_genes

        embedding, _ = trajectory_data
        expression = np.random.rand(100, 50)
        gene_names = np.array([f'Gene{i}' for i in range(50)])

        result = diffusion_pseudotime(embedding)
        genes_df = find_trajectory_genes(
            expression, result.pseudotime, gene_names,
            n_genes=10
        )

        assert len(genes_df) == 10
        assert 'gene' in genes_df.columns
        assert 'correlation' in genes_df.columns

    def test_plot_trajectory(self, trajectory_data):
        """Test trajectory plotting."""
        from scgcl import diffusion_pseudotime, plot_trajectory
        import matplotlib
        matplotlib.use('Agg')

        embedding, labels = trajectory_data
        result = diffusion_pseudotime(embedding)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_trajectory(embedding, result, labels=labels, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestBatchIntegration:
    """Test batch integration methods."""

    @pytest.fixture
    def integration_data(self):
        """Create data for integration testing."""
        np.random.seed(42)

        # Two batches with different offsets
        batch1 = np.random.rand(50, 30) + np.array([1, 0, 0] + [0]*27)
        batch2 = np.random.rand(50, 30) + np.array([0, 1, 0] + [0]*27)

        expression = np.vstack([batch1, batch2])
        batch = np.array(['A']*50 + ['B']*50)

        return expression, batch

    def test_harmony(self, integration_data):
        """Test Harmony integration."""
        from scgcl import harmony

        expression, batch = integration_data
        result = harmony(expression, batch, n_clusters=10, max_iter=5)

        assert result.corrected.shape == expression.shape
        assert result.method == 'harmony'
        assert 'mixing_score' in result.metrics

    def test_combat(self, integration_data):
        """Test ComBat integration."""
        from scgcl import combat

        expression, batch = integration_data
        result = combat(expression, batch)

        assert result.corrected.shape == expression.shape
        assert result.method == 'combat'

    def test_regress_batch(self, integration_data):
        """Test regression-based batch correction."""
        from scgcl import regress_batch

        expression, batch = integration_data
        result = regress_batch(expression, batch)

        assert result.corrected.shape == expression.shape
        assert result.method == 'regression'

    def test_mnn_correct(self):
        """Test MNN correction."""
        from scgcl import mnn_correct

        np.random.seed(42)
        batch1 = np.random.rand(30, 20)
        batch2 = np.random.rand(40, 20) + 0.5

        result = mnn_correct([batch1, batch2], ['A', 'B'], k=5)

        assert len(result.corrected) == 70
        assert result.method == 'mnn'

    def test_integrate_harmony(self, integration_data):
        """Test unified integrate interface with Harmony."""
        from scgcl import integrate

        expression, batch = integration_data
        result = integrate(expression, batch, method='harmony', max_iter=3)

        assert result.corrected.shape == expression.shape

    def test_integrate_combat(self, integration_data):
        """Test unified integrate interface with ComBat."""
        from scgcl import integrate

        expression, batch = integration_data
        result = integrate(expression, batch, method='combat')

        assert result.corrected.shape == expression.shape

    def test_integration_result_summary(self, integration_data):
        """Test IntegrationResult.summary()."""
        from scgcl import harmony

        expression, batch = integration_data
        result = harmony(expression, batch, max_iter=3)
        summary = result.summary()

        assert "Batch Integration Results" in summary
        assert "harmony" in summary

    def test_compute_lisi(self, integration_data):
        """Test LISI computation."""
        from scgcl import compute_lisi

        expression, batch = integration_data
        lisi = compute_lisi(expression, batch, n_neighbors=20)

        assert len(lisi) == 100
        assert all(lisi >= 1)  # LISI >= 1

    def test_plot_integration(self, integration_data):
        """Test integration plotting."""
        from scgcl import harmony, plot_integration
        import matplotlib
        matplotlib.use('Agg')

        expression, batch = integration_data
        result = harmony(expression, batch, max_iter=3)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_integration(
                expression, result.corrected, batch,
                method='pca', save_path=f.name
            )
            assert os.path.exists(f.name)
            os.unlink(f.name)


class TestDifferentialExpression:
    """Test differential expression analysis."""

    @pytest.fixture
    def de_data(self):
        """Create data for DE testing."""
        np.random.seed(42)
        n_cells = 100
        n_genes = 50

        # Two groups with some differentially expressed genes
        expression = np.random.rand(n_cells, n_genes)

        # Make first 10 genes higher in group2
        expression[50:, :10] += 1.5

        groups = np.array(['control']*50 + ['treatment']*50)
        gene_names = np.array([f'Gene{i}' for i in range(n_genes)])

        return expression, groups, gene_names

    def test_differential_expression_wilcoxon(self, de_data):
        """Test DE with Wilcoxon test."""
        from scgcl import differential_expression

        expression, groups, gene_names = de_data
        result = differential_expression(
            expression, groups, gene_names,
            method='wilcoxon'
        )

        assert len(result.results) > 0
        assert 'gene' in result.results.columns
        assert 'log2FC' in result.results.columns
        assert 'padj' in result.results.columns
        assert result.method == 'wilcoxon'

    def test_differential_expression_ttest(self, de_data):
        """Test DE with t-test."""
        from scgcl import differential_expression

        expression, groups, gene_names = de_data
        result = differential_expression(
            expression, groups, gene_names,
            method='t-test'
        )

        assert len(result.results) > 0
        assert result.method == 't-test'

    def test_de_result_significant(self, de_data):
        """Test DEResult.significant()."""
        from scgcl import differential_expression

        expression, groups, gene_names = de_data
        result = differential_expression(expression, groups, gene_names)

        sig = result.significant(pval_cutoff=0.05, fc_cutoff=0.5)
        assert isinstance(sig, pd.DataFrame)

    def test_de_result_upregulated(self, de_data):
        """Test DEResult.upregulated()."""
        from scgcl import differential_expression

        expression, groups, gene_names = de_data
        result = differential_expression(expression, groups, gene_names)

        up = result.upregulated()
        if len(up) > 0:
            assert all(up['log2FC'] > 0)

    def test_de_result_downregulated(self, de_data):
        """Test DEResult.downregulated()."""
        from scgcl import differential_expression

        expression, groups, gene_names = de_data
        result = differential_expression(expression, groups, gene_names)

        down = result.downregulated()
        if len(down) > 0:
            assert all(down['log2FC'] < 0)

    def test_de_result_summary(self, de_data):
        """Test DEResult.summary()."""
        from scgcl import differential_expression

        expression, groups, gene_names = de_data
        result = differential_expression(expression, groups, gene_names)
        summary = result.summary()

        assert "Differential Expression Results" in summary
        assert "treatment vs control" in summary

    def test_pairwise_de(self, de_data):
        """Test pairwise DE."""
        from scgcl import pairwise_de

        expression, groups, gene_names = de_data
        results = pairwise_de(expression, groups, gene_names)

        assert len(results) >= 1
        assert 'treatment_vs_control' in results

    def test_one_vs_rest_de(self):
        """Test one-vs-rest DE."""
        from scgcl import one_vs_rest_de

        np.random.seed(42)
        expression = np.random.rand(90, 30)
        groups = np.array(['A']*30 + ['B']*30 + ['C']*30)
        gene_names = np.array([f'Gene{i}' for i in range(30)])

        results = one_vs_rest_de(expression, groups, gene_names)

        assert len(results) == 3
        assert 'A' in results
        assert 'B' in results
        assert 'C' in results

    def test_de_between_conditions(self):
        """Test DE between conditions within clusters."""
        from scgcl import de_between_conditions

        np.random.seed(42)
        expression = np.random.rand(100, 30)
        condition = np.array(['ctrl']*50 + ['treat']*50)
        cluster = np.array([0]*25 + [1]*25 + [0]*25 + [1]*25)
        gene_names = np.array([f'Gene{i}' for i in range(30)])

        results = de_between_conditions(
            expression, condition, cluster, gene_names,
            condition1='ctrl', condition2='treat'
        )

        assert len(results) == 2  # Two clusters
        assert 0 in results
        assert 1 in results

    def test_plot_volcano(self, de_data):
        """Test volcano plot."""
        from scgcl import differential_expression, plot_volcano
        import matplotlib
        matplotlib.use('Agg')

        expression, groups, gene_names = de_data
        result = differential_expression(expression, groups, gene_names)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_volcano(result, top_n=5, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_plot_ma(self, de_data):
        """Test MA plot."""
        from scgcl import differential_expression, plot_ma
        import matplotlib
        matplotlib.use('Agg')

        expression, groups, gene_names = de_data
        result = differential_expression(expression, groups, gene_names)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_ma(result, save_path=f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_compare_de_methods(self, de_data):
        """Test comparing DE methods."""
        from scgcl import compare_de_methods

        expression, groups, gene_names = de_data
        comparison = compare_de_methods(
            expression, groups, gene_names,
            methods=['wilcoxon', 't-test']
        )

        assert 'wilcoxon_log2FC' in comparison.columns
        assert 't-test_log2FC' in comparison.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
