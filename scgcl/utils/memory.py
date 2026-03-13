"""Memory profiling utilities for tracking resource usage."""

import gc
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import time


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    label: str
    cpu_allocated_mb: float
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_max_allocated_mb: float = 0.0


@dataclass
class MemoryStats:
    """Aggregated memory statistics."""
    peak_cpu_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    avg_gpu_mb: float = 0.0
    total_snapshots: int = 0
    snapshots: List[MemorySnapshot] = field(default_factory=list)


class MemoryProfiler:
    """
    Track memory usage during training.

    Parameters
    ----------
    enabled : bool
        Whether profiling is active
    track_gpu : bool
        Whether to track GPU memory (requires CUDA)
    verbose : bool
        Print memory updates when snapshotting

    Example
    -------
    >>> profiler = MemoryProfiler(enabled=True)
    >>> profiler.snapshot("before_training")
    >>> model.fit(X)
    >>> profiler.snapshot("after_training")
    >>> profiler.report()
    """

    def __init__(self, enabled: bool = True, track_gpu: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.verbose = verbose

        self._snapshots: List[MemorySnapshot] = []
        self._start_time: Optional[float] = None

    def start(self) -> 'MemoryProfiler':
        """Start profiling session."""
        if not self.enabled:
            return self

        self._start_time = time.time()
        self._snapshots = []

        gc.collect()
        if self.track_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.snapshot("start")
        return self

    def snapshot(self, label: str = "") -> Optional[MemorySnapshot]:
        """Take a memory snapshot."""
        if not self.enabled:
            return None

        import sys
        import os

        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            cpu_mb = 0.0

        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        gpu_max_mb = 0.0

        if self.track_gpu:
            gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_max_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        snapshot = MemorySnapshot(
            timestamp=time.time() - (self._start_time or time.time()),
            label=label,
            cpu_allocated_mb=cpu_mb,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_max_allocated_mb=gpu_max_mb
        )

        self._snapshots.append(snapshot)

        if self.verbose:
            self._print_snapshot(snapshot)

        return snapshot

    def _print_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Print a single snapshot."""
        msg = f"[Memory:{snapshot.label}] CPU: {snapshot.cpu_allocated_mb:.1f}MB"
        if self.track_gpu:
            msg += f" | GPU: {snapshot.gpu_allocated_mb:.1f}MB (peak: {snapshot.gpu_max_allocated_mb:.1f}MB)"
        print(msg)

    def get_stats(self) -> MemoryStats:
        """Get aggregated statistics."""
        if not self._snapshots:
            return MemoryStats()

        peak_cpu = max(s.cpu_allocated_mb for s in self._snapshots)
        peak_gpu = max(s.gpu_allocated_mb for s in self._snapshots) if self.track_gpu else 0.0
        avg_gpu = sum(s.gpu_allocated_mb for s in self._snapshots) / len(self._snapshots) if self.track_gpu else 0.0

        return MemoryStats(
            peak_cpu_mb=peak_cpu,
            peak_gpu_mb=peak_gpu,
            avg_gpu_mb=avg_gpu,
            total_snapshots=len(self._snapshots),
            snapshots=self._snapshots.copy()
        )

    def report(self) -> str:
        """Generate a human-readable report."""
        stats = self.get_stats()

        lines = [
            "=" * 50,
            "Memory Profile Report",
            "=" * 50,
            f"Total snapshots: {stats.total_snapshots}",
            f"Peak CPU memory: {stats.peak_cpu_mb:.1f} MB",
        ]

        if self.track_gpu:
            lines.extend([
                f"Peak GPU memory: {stats.peak_gpu_mb:.1f} MB",
                f"Avg GPU memory: {stats.avg_gpu_mb:.1f} MB",
            ])

        if self._snapshots:
            lines.append("\nSnapshots:")
            for s in self._snapshots:
                gpu_info = f" | GPU: {s.gpu_allocated_mb:.1f}MB" if self.track_gpu else ""
                lines.append(f"  [{s.timestamp:.2f}s] {s.label}: CPU={s.cpu_allocated_mb:.1f}MB{gpu_info}")

        lines.append("=" * 50)

        report = "\n".join(lines)
        print(report)
        return report

    def reset(self) -> None:
        """Reset all profiling data."""
        self._snapshots = []
        self._start_time = None
        if self.track_gpu:
            torch.cuda.reset_peak_memory_stats()

    def to_dict(self) -> Dict[str, Any]:
        """Export profiling data as dictionary."""
        stats = self.get_stats()
        return {
            'peak_cpu_mb': stats.peak_cpu_mb,
            'peak_gpu_mb': stats.peak_gpu_mb,
            'avg_gpu_mb': stats.avg_gpu_mb,
            'snapshots': [
                {
                    'timestamp': s.timestamp,
                    'label': s.label,
                    'cpu_mb': s.cpu_allocated_mb,
                    'gpu_mb': s.gpu_allocated_mb,
                    'gpu_reserved_mb': s.gpu_reserved_mb,
                    'gpu_peak_mb': s.gpu_max_allocated_mb
                }
                for s in self._snapshots
            ]
        }


@contextmanager
def profile_memory(label: str = "block", verbose: bool = True):
    """
    Context manager for profiling memory of a code block.

    Example
    -------
    >>> with profile_memory("training"):
    ...     model.fit(X)
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        start_cpu = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        start_cpu = 0.0

    start_gpu = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    yield

    gc.collect()

    try:
        end_cpu = process.memory_info().rss / (1024 * 1024)
    except:
        end_cpu = 0.0

    end_gpu = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    if verbose:
        print(f"[Memory:{label}] CPU: {start_cpu:.1f}MB -> {end_cpu:.1f}MB (delta: {end_cpu - start_cpu:+.1f}MB)")
        if torch.cuda.is_available():
            print(f"[Memory:{label}] GPU: {start_gpu:.1f}MB -> {end_gpu:.1f}MB (peak: {peak_gpu:.1f}MB)")


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information in MB."""
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
        'reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
        'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
        'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    }


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
