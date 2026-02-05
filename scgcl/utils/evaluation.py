"""Evaluation metrics for clustering."""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score,
    homogeneity_score, completeness_score, v_measure_score
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute clustering accuracy using Hungarian algorithm."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    cost_matrix = np.zeros((n_classes, n_classes))

    for i in range(len(y_true)):
        cost_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(y_true)


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray, X: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive clustering metrics."""
    metrics = {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'ACC': clustering_accuracy(y_true, y_pred),
        'Homogeneity': homogeneity_score(y_true, y_pred),
        'Completeness': completeness_score(y_true, y_pred),
        'V-measure': v_measure_score(y_true, y_pred)
    }

    if X is not None and len(np.unique(y_pred)) > 1:
        try:
            metrics['Silhouette'] = silhouette_score(X, y_pred)
            metrics['Calinski-Harabasz'] = calinski_harabasz_score(X, y_pred)
            metrics['Davies-Bouldin'] = davies_bouldin_score(X, y_pred)
        except Exception:
            pass

    return metrics


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray, X: Optional[np.ndarray] = None, verbose: bool = True) -> Dict[str, float]:
    """Evaluate clustering with optional printing."""
    metrics = clustering_metrics(y_true, y_pred, X)

    if verbose:
        print("\nClustering Evaluation:")
        print("-" * 40)
        for name, value in metrics.items():
            print(f"  {name:<18} {value:.4f}")
        print("-" * 40)

    return metrics


class ClusterStabilityAnalyzer:
    """Analyze clustering stability across multiple runs."""

    def __init__(self, n_runs: int = 10):
        self.n_runs = n_runs
        self.results = []

    def add_result(self, y_true: np.ndarray, y_pred: np.ndarray, X: Optional[np.ndarray] = None):
        self.results.append(clustering_metrics(y_true, y_pred, X))

    def summarize(self) -> Dict[str, Dict[str, float]]:
        if not self.results:
            return {}

        summary = {}
        for name in self.results[0]:
            values = [r[name] for r in self.results if name in r]
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return summary

    def print_summary(self):
        summary = self.summarize()
        print(f"\nStability Analysis ({len(self.results)} runs):")
        print("-" * 50)
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
        print("-" * 50)
        for name, stats in summary.items():
            print(f"{name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f}")
        print("-" * 50)
