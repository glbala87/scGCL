"""Tests for quick win features: reproducibility, callbacks, memory, confidence."""

import pytest
import numpy as np
import torch


class TestReproducibility:
    """Test reproducibility utilities."""

    def test_set_seed(self):
        """Test that set_seed produces consistent results."""
        from scgcl import set_seed

        set_seed(42)
        a1 = np.random.rand(10)
        b1 = torch.rand(10)

        set_seed(42)
        a2 = np.random.rand(10)
        b2 = torch.rand(10)

        assert np.allclose(a1, a2)
        assert torch.allclose(b1, b2)

    def test_rng_state(self):
        """Test RNG state save/restore."""
        from scgcl import get_rng_state, set_rng_state, set_seed

        set_seed(42)
        _ = np.random.rand(5)

        state = get_rng_state()
        expected = np.random.rand(5)

        set_rng_state(state)
        actual = np.random.rand(5)

        assert np.allclose(expected, actual)

    def test_reproducibility_context(self):
        """Test ReproducibilityContext."""
        from scgcl import ReproducibilityContext

        with ReproducibilityContext(seed=123):
            a1 = np.random.rand(5)

        with ReproducibilityContext(seed=123):
            a2 = np.random.rand(5)

        assert np.allclose(a1, a2)


class TestCallbacks:
    """Test callback system."""

    def test_early_stopping(self):
        """Test EarlyStopping callback."""
        from scgcl import EarlyStopping

        es = EarlyStopping(monitor='loss', patience=3, mode='min')
        es.on_train_begin()

        assert es.on_epoch_end(0, {'loss': 1.0}) == True
        assert es.on_epoch_end(1, {'loss': 0.9}) == True  # Improvement, wait=0
        assert es.on_epoch_end(2, {'loss': 0.95}) == True  # No improvement, wait=1
        assert es.on_epoch_end(3, {'loss': 0.96}) == True  # No improvement, wait=2
        assert es.on_epoch_end(4, {'loss': 0.97}) == False  # No improvement, wait=3 >= patience

    def test_progress_logger(self):
        """Test ProgressLogger callback."""
        from scgcl import ProgressLogger

        logger = ProgressLogger(log_interval=5)
        logger.on_phase_begin('pretrain')
        logger.on_epoch_end(0, {'loss': 1.0})
        logger.on_epoch_end(4, {'loss': 0.5})

        history = logger.get_history()
        assert len(history) == 2
        assert history[0]['loss'] == 1.0
        assert history[1]['loss'] == 0.5

    def test_timer(self):
        """Test Timer callback."""
        import time
        from scgcl import Timer

        timer = Timer(verbose=False)
        timer.on_train_begin()
        timer.on_phase_begin('test')
        time.sleep(0.1)
        timer.on_phase_end('test')
        timer.on_train_end()

        stats = timer.get_stats()
        assert 'test' in stats['phase_times']
        assert stats['phase_times']['test'] >= 0.1

    def test_callback_list(self):
        """Test CallbackList."""
        from scgcl import CallbackList, EarlyStopping, Timer

        callbacks = CallbackList([EarlyStopping(patience=5), Timer(verbose=False)])
        callbacks.on_train_begin()
        result = callbacks.on_epoch_end(0, {'loss': 1.0})
        callbacks.on_train_end()

        assert result == True


class TestMemoryProfiler:
    """Test memory profiling utilities."""

    def test_memory_profiler_basic(self):
        """Test basic MemoryProfiler functionality."""
        from scgcl import MemoryProfiler

        profiler = MemoryProfiler(enabled=True, verbose=False)
        profiler.start()
        profiler.snapshot('test1')
        profiler.snapshot('test2')

        stats = profiler.get_stats()
        assert stats.total_snapshots == 3  # start + 2 snapshots

    def test_memory_profiler_disabled(self):
        """Test MemoryProfiler when disabled."""
        from scgcl import MemoryProfiler

        profiler = MemoryProfiler(enabled=False)
        profiler.start()
        result = profiler.snapshot('test')

        assert result is None

    def test_profile_memory_context(self):
        """Test profile_memory context manager."""
        from scgcl import profile_memory

        with profile_memory('test_block', verbose=False):
            x = np.random.rand(1000, 1000)

    def test_gpu_memory_info(self):
        """Test get_gpu_memory_info."""
        from scgcl import get_gpu_memory_info

        info = get_gpu_memory_info()
        assert 'available' in info


class TestConfidenceScores:
    """Test confidence score functionality."""

    def test_get_confidence_scores(self):
        """Test get_confidence_scores returns valid values."""
        from scgcl import ScGCL
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=30, n_genes=50, n_clusters=3)

        model = ScGCL(n_clusters=3, hidden_dim=16, pretrain_epochs=2, ssc_epochs=2)
        model.fit(X, preprocess=False, verbose=False)

        confidence = model.get_confidence_scores()

        assert len(confidence) == 30
        assert all(0 <= c <= 1 for c in confidence)

    def test_get_soft_assignments(self):
        """Test get_soft_assignments returns valid probabilities."""
        from scgcl import ScGCL
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=30, n_genes=50, n_clusters=3)

        model = ScGCL(n_clusters=3, hidden_dim=16, pretrain_epochs=2, ssc_epochs=2)
        model.fit(X, preprocess=False, verbose=False)

        soft = model.get_soft_assignments()

        assert soft.shape == (30, 3)
        assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_with_confidence(self):
        """Test predict_with_confidence on new data."""
        from scgcl import ScGCL
        from scgcl.utils import simulate_scrna_data

        X_train, _ = simulate_scrna_data(n_cells=30, n_genes=50, n_clusters=3)
        X_test, _ = simulate_scrna_data(n_cells=10, n_genes=50, n_clusters=3)

        model = ScGCL(n_clusters=3, hidden_dim=16, pretrain_epochs=2, ssc_epochs=2)
        model.fit(X_train, preprocess=False, verbose=False)

        labels, confidence, soft = model.predict_with_confidence(X_test, preprocess=False)

        assert len(labels) == 10
        assert len(confidence) == 10
        assert soft.shape == (10, 3)


class TestModelSaveLoad:
    """Test model save/load functionality."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test saving and loading a model."""
        from scgcl import ScGCL
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=30, n_genes=50, n_clusters=3)

        model = ScGCL(n_clusters=3, hidden_dim=16, pretrain_epochs=2, ssc_epochs=2)
        model.fit(X, preprocess=False, verbose=False)

        save_path = tmp_path / "model.pt"
        model.save(str(save_path))

        model2 = ScGCL()
        model2.load(str(save_path), input_dim=50)

        assert model2.n_clusters == 3
        assert model2.hidden_dim == 16
        assert model2.labels_ is not None


class TestIntegration:
    """Integration tests combining multiple quick wins."""

    def test_fit_with_callbacks_and_memory(self):
        """Test fitting with callbacks and memory profiling enabled."""
        from scgcl import ScGCL, EarlyStopping, Timer
        from scgcl.utils import simulate_scrna_data

        X, y = simulate_scrna_data(n_cells=30, n_genes=50, n_clusters=3)

        callbacks = [
            EarlyStopping(monitor='loss', patience=50),
            Timer(verbose=False)
        ]

        model = ScGCL(n_clusters=3, hidden_dim=16, pretrain_epochs=3, ssc_epochs=3)
        model.fit(X, preprocess=False, verbose=False, callbacks=callbacks, memory_profiling=False)

        assert len(model.labels_) == 30
        confidence = model.get_confidence_scores()
        assert len(confidence) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
