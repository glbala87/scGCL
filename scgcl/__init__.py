"""
scGCL: Single-Cell Graph Contrastive Learning

A framework for clustering single-cell RNA sequencing data using
graph neural networks and contrastive learning.
"""

__version__ = "0.2.0"
__author__ = "scGCL Team"

from .model import ScGCL, run_experiment
from .model_gpu import ScGCLGPU

# Utilities
from .utils.reproducibility import set_seed, get_rng_state, set_rng_state, ReproducibilityContext
from .utils.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ProgressLogger,
    Timer,
    LambdaCallback,
    CheckpointCallback
)
from .utils.memory import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryStats,
    profile_memory,
    get_gpu_memory_info,
    clear_gpu_memory
)

# Analysis
from .analysis.markers import (
    find_marker_genes,
    rank_genes_groups,
    filter_markers,
    MarkerGeneResult
)

# Tuning
from .tuning import (
    HyperparameterTuner,
    TuningResult,
    auto_tune,
    quick_tune
)

# Integration (lazy import to avoid dependency issues)
def _import_integration():
    from .integration import scgcl as scanpy_scgcl, scgcl_markers, to_anndata, from_anndata
    return scanpy_scgcl, scgcl_markers, to_anndata, from_anndata

__all__ = [
    # Main classes
    'ScGCL',
    'ScGCLGPU',
    'run_experiment',
    # Reproducibility
    'set_seed',
    'get_rng_state',
    'set_rng_state',
    'ReproducibilityContext',
    # Callbacks
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'ProgressLogger',
    'Timer',
    'LambdaCallback',
    'CheckpointCallback',
    # Memory
    'MemoryProfiler',
    'MemorySnapshot',
    'MemoryStats',
    'profile_memory',
    'get_gpu_memory_info',
    'clear_gpu_memory',
    # Analysis
    'find_marker_genes',
    'rank_genes_groups',
    'filter_markers',
    'MarkerGeneResult',
    # Tuning
    'HyperparameterTuner',
    'TuningResult',
    'auto_tune',
    'quick_tune',
]
