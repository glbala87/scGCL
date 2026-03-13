"""Reproducibility utilities for consistent results across runs."""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value
    deterministic : bool
        If True, enables CUDA deterministic operations (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_rng_state() -> dict:
    """
    Capture current RNG state for all libraries.

    Returns
    -------
    dict
        Dictionary containing RNG states for random, numpy, and torch
    """
    state = {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: dict) -> None:
    """
    Restore RNG state for all libraries.

    Parameters
    ----------
    state : dict
        Dictionary containing RNG states (from get_rng_state)
    """
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.

    Example
    -------
    >>> with ReproducibilityContext(seed=42):
    ...     # All random operations here are reproducible
    ...     result = model.fit_predict(X)
    """

    def __init__(self, seed: int = 42, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self._saved_state: Optional[dict] = None
        self._saved_cudnn_deterministic: Optional[bool] = None
        self._saved_cudnn_benchmark: Optional[bool] = None

    def __enter__(self):
        self._saved_state = get_rng_state()

        if torch.cuda.is_available():
            self._saved_cudnn_deterministic = torch.backends.cudnn.deterministic
            self._saved_cudnn_benchmark = torch.backends.cudnn.benchmark

        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._saved_state is not None:
            set_rng_state(self._saved_state)

        if torch.cuda.is_available():
            if self._saved_cudnn_deterministic is not None:
                torch.backends.cudnn.deterministic = self._saved_cudnn_deterministic
            if self._saved_cudnn_benchmark is not None:
                torch.backends.cudnn.benchmark = self._saved_cudnn_benchmark

        return False
