"""Hyperparameter auto-tuning using Optuna."""

import numpy as np
from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""

    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    study: Any = None  # Optuna study object

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 50,
            "Hyperparameter Tuning Results",
            "=" * 50,
            f"Best Score: {self.best_score:.4f}",
            f"Best Trial: {self.best_trial}",
            "",
            "Best Parameters:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


class HyperparameterTuner:
    """
    Automated hyperparameter tuning for scGCL using Optuna.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials
    metric : str
        Metric to optimize: 'silhouette', 'calinski', 'davies', 'ari', 'nmi'
    direction : str
        Optimization direction: 'maximize' or 'minimize'
    timeout : int, optional
        Maximum tuning time in seconds
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    seed : int
        Random seed
    verbose : bool
        Print progress

    Example
    -------
    >>> from scgcl import HyperparameterTuner
    >>> from scgcl.utils import simulate_scrna_data
    >>>
    >>> X, y = simulate_scrna_data(n_cells=500, n_clusters=5)
    >>>
    >>> tuner = HyperparameterTuner(n_trials=50, metric='silhouette')
    >>> result = tuner.tune(X, y_true=y)
    >>>
    >>> print(result.summary())
    >>> best_model = tuner.get_best_model(X)
    """

    # Default search space
    DEFAULT_SEARCH_SPACE = {
        'hidden_dim': {'type': 'categorical', 'choices': [32, 64, 128]},
        'proj_dim': {'type': 'categorical', 'choices': [16, 32, 64]},
        'num_layers': {'type': 'int', 'low': 1, 'high': 3},
        'k_neighbors': {'type': 'int', 'low': 5, 'high': 30},
        'temperature': {'type': 'float', 'low': 0.1, 'high': 1.0, 'log': True},
        'lambda_align': {'type': 'float', 'low': 0.1, 'high': 2.0},
        'lambda_uniform': {'type': 'float', 'low': 0.1, 'high': 2.0},
        'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
    }

    def __init__(
        self,
        n_trials: int = 100,
        metric: str = 'silhouette',
        direction: str = 'maximize',
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        seed: int = 42,
        verbose: bool = True
    ):
        self.n_trials = n_trials
        self.metric = metric
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        # Set direction based on metric
        if metric == 'davies':
            self.direction = 'minimize'
        elif direction:
            self.direction = direction
        else:
            self.direction = 'maximize'

        self._study = None
        self._best_params = None
        self._search_space = self.DEFAULT_SEARCH_SPACE.copy()

    def set_search_space(self, search_space: Dict[str, Dict]) -> 'HyperparameterTuner':
        """
        Set custom search space.

        Parameters
        ----------
        search_space : dict
            Dictionary mapping parameter names to search configurations.
            Each config should have 'type' and relevant bounds.

        Returns
        -------
        self
        """
        self._search_space.update(search_space)
        return self

    def _sample_params(self, trial) -> Dict[str, Any]:
        """Sample parameters from search space using Optuna trial."""
        params = {}

        for name, config in self._search_space.items():
            param_type = config['type']

            if param_type == 'categorical':
                params[name] = trial.suggest_categorical(name, config['choices'])
            elif param_type == 'int':
                params[name] = trial.suggest_int(name, config['low'], config['high'])
            elif param_type == 'float':
                log = config.get('log', False)
                params[name] = trial.suggest_float(name, config['low'], config['high'], log=log)

        return params

    def _compute_metric(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> float:
        """Compute the optimization metric."""
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            adjusted_rand_score,
            normalized_mutual_info_score
        )

        n_unique = len(np.unique(labels))
        if n_unique < 2:
            return float('-inf') if self.direction == 'maximize' else float('inf')

        if self.metric == 'silhouette':
            return silhouette_score(embeddings, labels)
        elif self.metric == 'calinski':
            return calinski_harabasz_score(embeddings, labels)
        elif self.metric == 'davies':
            return davies_bouldin_score(embeddings, labels)
        elif self.metric == 'ari':
            if y_true is None:
                raise ValueError("y_true required for ARI metric")
            return adjusted_rand_score(y_true, labels)
        elif self.metric == 'nmi':
            if y_true is None:
                raise ValueError("y_true required for NMI metric")
            return normalized_mutual_info_score(y_true, labels)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def tune(
        self,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        preprocess: bool = True,
        pretrain_epochs: int = 50,
        ssc_epochs: int = 100
    ) -> TuningResult:
        """
        Run hyperparameter tuning.

        Parameters
        ----------
        X : np.ndarray
            Input expression matrix
        y_true : np.ndarray, optional
            Ground truth labels (required for ARI/NMI metrics)
        n_clusters : int, optional
            Number of clusters (auto-detected if None)
        fixed_params : dict, optional
            Parameters to fix (not tuned)
        preprocess : bool
            Whether to preprocess data
        pretrain_epochs : int
            Pretraining epochs (reduced for speed)
        ssc_epochs : int
            SSC epochs (reduced for speed)

        Returns
        -------
        TuningResult
            Tuning results with best parameters
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError(
                "Optuna required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        from ..model import ScGCL

        fixed_params = fixed_params or {}

        def objective(trial):
            # Sample parameters
            params = self._sample_params(trial)
            params.update(fixed_params)

            # Add fixed training params
            params['pretrain_epochs'] = pretrain_epochs
            params['ssc_epochs'] = ssc_epochs
            params['seed'] = self.seed + trial.number

            if n_clusters is not None:
                params['n_clusters'] = n_clusters

            try:
                # Train model
                model = ScGCL(**params)
                model.fit(X, preprocess=preprocess, verbose=False)

                # Compute metric
                embeddings = model.get_embeddings()
                score = self._compute_metric(model.labels_, embeddings, y_true)

                return score

            except Exception as e:
                if self.verbose:
                    print(f"Trial {trial.number} failed: {e}")
                return float('-inf') if self.direction == 'maximize' else float('inf')

        # Create study
        sampler = TPESampler(seed=self.seed)
        self._study = optuna.create_study(
            direction=self.direction,
            sampler=sampler
        )

        # Configure logging
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.verbose
        )

        # Collect results
        self._best_params = self._study.best_params
        all_trials = [
            {
                'number': t.number,
                'params': t.params,
                'value': t.value,
                'state': str(t.state)
            }
            for t in self._study.trials
        ]

        return TuningResult(
            best_params=self._best_params,
            best_score=self._study.best_value,
            best_trial=self._study.best_trial.number,
            all_trials=all_trials,
            study=self._study
        )

    def get_best_model(
        self,
        X: np.ndarray,
        n_clusters: Optional[int] = None,
        pretrain_epochs: int = 100,
        ssc_epochs: int = 500,
        **kwargs
    ):
        """
        Get a model trained with the best parameters.

        Parameters
        ----------
        X : np.ndarray
            Input data
        n_clusters : int, optional
            Number of clusters
        pretrain_epochs : int
            Full training epochs
        ssc_epochs : int
            Full SSC epochs
        **kwargs
            Additional parameters

        Returns
        -------
        ScGCL
            Trained model with best parameters
        """
        if self._best_params is None:
            raise RuntimeError("No tuning results. Call tune() first.")

        from ..model import ScGCL

        params = self._best_params.copy()
        params['pretrain_epochs'] = pretrain_epochs
        params['ssc_epochs'] = ssc_epochs
        params.update(kwargs)

        if n_clusters is not None:
            params['n_clusters'] = n_clusters

        model = ScGCL(**params)
        model.fit(X, **kwargs)

        return model


def auto_tune(
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    n_trials: int = 50,
    metric: str = 'silhouette',
    verbose: bool = True,
    **kwargs
) -> TuningResult:
    """
    Convenience function for quick hyperparameter tuning.

    Parameters
    ----------
    X : np.ndarray
        Input expression matrix
    y_true : np.ndarray, optional
        Ground truth labels
    n_clusters : int, optional
        Number of clusters
    n_trials : int
        Number of trials
    metric : str
        Optimization metric
    verbose : bool
        Print progress
    **kwargs
        Additional arguments for tuning

    Returns
    -------
    TuningResult
        Tuning results
    """
    tuner = HyperparameterTuner(
        n_trials=n_trials,
        metric=metric,
        verbose=verbose
    )

    return tuner.tune(X, y_true=y_true, n_clusters=n_clusters, **kwargs)


def quick_tune(
    X: np.ndarray,
    n_clusters: Optional[int] = None,
    n_trials: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick tuning with minimal trials for fast iteration.

    Parameters
    ----------
    X : np.ndarray
        Input data
    n_clusters : int, optional
        Number of clusters
    n_trials : int
        Number of trials (default 20)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Best parameters
    """
    tuner = HyperparameterTuner(
        n_trials=n_trials,
        metric='silhouette',
        verbose=verbose
    )

    # Use reduced search space for speed
    tuner.set_search_space({
        'hidden_dim': {'type': 'categorical', 'choices': [32, 64]},
        'k_neighbors': {'type': 'int', 'low': 10, 'high': 20},
        'temperature': {'type': 'float', 'low': 0.3, 'high': 0.7},
    })

    result = tuner.tune(
        X,
        n_clusters=n_clusters,
        pretrain_epochs=20,
        ssc_epochs=50
    )

    if verbose:
        print(result.summary())

    return result.best_params
