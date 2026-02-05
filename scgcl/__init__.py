"""
scGCL: Single-Cell Graph Contrastive Learning

A framework for clustering single-cell RNA sequencing data using
graph neural networks and contrastive learning.
"""

__version__ = "0.1.0"
__author__ = "scGCL Team"

from .model import ScGCL, run_experiment
from .model_gpu import ScGCLGPU

__all__ = ['ScGCL', 'ScGCLGPU', 'run_experiment']
