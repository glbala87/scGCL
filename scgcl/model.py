"""scGCL: Main CPU implementation."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from tqdm import tqdm

from .models.encoder import ContrastiveEncoder
from .utils.data import preprocess_data
from .utils.graph import build_adaptive_knn_graph, compute_snn_weights
from .utils.augmentation import ContrastiveAugmentation
from .utils.evaluation import evaluate_clustering, ClusterStabilityAnalyzer
from .utils.reproducibility import set_seed
from .utils.callbacks import Callback, CallbackList
from .utils.memory import MemoryProfiler
from .losses.contrastive import CombinedContrastiveLoss
from .clustering.ssc import SelfSupervisedClustering, ClusterRefiner, tune_n_clusters


class ScGCL:
    """
    Single-cell Graph Contrastive Learning for clustering.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters (auto-estimated if None)
    hidden_dim : int
        Hidden dimension for encoder
    proj_dim : int
        Projection dimension
    num_layers : int
        Number of GCN layers
    encoder_type : str
        Encoder type ('gcn' or 'gat')
    k_neighbors : int
        Number of neighbors for kNN
    temperature : float
        Contrastive loss temperature
    tau_plus : float
        Debiasing parameter
    lambda_align : float
        Alignment loss weight
    lambda_uniform : float
        Uniformity loss weight
    pretrain_epochs : int
        Pretraining epochs
    ssc_epochs : int
        SSC refinement epochs
    ssc_batch_size : int
        Batch size for SSC
    lr : float
        Learning rate
    weight_decay : float
        Weight decay
    device : str
        Device ('cpu' or 'cuda')
    seed : int
        Random seed
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        hidden_dim: int = 64,
        proj_dim: int = 32,
        num_layers: int = 2,
        encoder_type: str = 'gcn',
        k_neighbors: int = 15,
        temperature: float = 0.5,
        tau_plus: float = 0.1,
        lambda_align: float = 1.0,
        lambda_uniform: float = 1.0,
        pretrain_epochs: int = 100,
        ssc_epochs: int = 500,
        ssc_batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cpu',
        seed: int = 42
    ):
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.lambda_align = lambda_align
        self.lambda_uniform = lambda_uniform
        self.pretrain_epochs = pretrain_epochs
        self.ssc_epochs = ssc_epochs
        self.ssc_batch_size = ssc_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed

        # Set global seed for reproducibility
        set_seed(seed, deterministic=True)

        self.encoder = None
        self.ssc = None
        self.x = None
        self.edge_index = None
        self.edge_weight = None
        self.pretrain_losses = []
        self.ssc_losses = []
        self._soft_assignments = None  # Store soft cluster assignments for confidence

    def _build_graph(self, X: np.ndarray):
        edge_index, edge_weight = build_adaptive_knn_graph(X, k_min=5, k_max=self.k_neighbors * 2)
        edge_weight = compute_snn_weights(edge_index, X.shape[0])
        return edge_index, edge_weight

    def _pretrain(self, x, edge_index, edge_weight, verbose=True, callbacks: Optional[CallbackList] = None):
        self.encoder.train()

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = CombinedContrastiveLoss(
            self.temperature, self.tau_plus, self.lambda_align, self.lambda_uniform
        )
        augmentor = ContrastiveAugmentation()

        losses = []
        warmup_epochs = max(1, min(10, self.pretrain_epochs // 5))
        iterator = tqdm(range(self.pretrain_epochs), desc="Pretraining") if verbose else range(self.pretrain_epochs)

        for epoch in iterator:
            if callbacks:
                callbacks.on_epoch_begin(epoch)

            optimizer.zero_grad()

            view1, view2 = augmentor(x, edge_index, edge_weight)
            x1, ei1, ew1 = view1
            x2, ei2, ew2 = view2

            _, _, _, z1, z2 = self.encoder.contrastive_forward(x, ei1, ei2, ew1, ew2, x1, x2)

            warmup = min(1.0, epoch / warmup_epochs)
            loss_dict = criterion(z1, z2, warmup)
            loss = loss_dict['total']

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if verbose and (epoch + 1) % 20 == 0:
                tqdm.write(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

            if callbacks:
                continue_training = callbacks.on_epoch_end(epoch, {'loss': loss.item(), **loss_dict})
                if not continue_training:
                    if verbose:
                        tqdm.write(f"Early stopping at epoch {epoch + 1}")
                    break

        return losses

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        preprocess: bool = True,
        n_pca: int = 50,
        verbose: bool = True,
        callbacks: Optional[List[Callback]] = None,
        memory_profiling: bool = False
    ) -> 'ScGCL':
        """
        Fit the clustering model.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (cells x genes)
        y : np.ndarray, optional
            Ground truth labels for evaluation
        preprocess : bool
            Whether to preprocess the data
        n_pca : int
            Number of PCA components
        verbose : bool
            Print progress information
        callbacks : list of Callback, optional
            Training callbacks for logging, early stopping, etc.
        memory_profiling : bool
            Enable memory usage tracking

        Returns
        -------
        self
        """
        # Initialize callbacks and memory profiler
        cb_list = CallbackList(callbacks)
        profiler = MemoryProfiler(enabled=memory_profiling, verbose=verbose)
        profiler.start()

        cb_list.on_train_begin({'n_samples': X.shape[0]})

        if preprocess:
            if verbose:
                print("Preprocessing data...")
            _, X_pca = preprocess_data(X, n_pca_components=n_pca)
            X_input = X_pca
        else:
            X_input = X

        n_samples, n_features = X_input.shape
        if verbose:
            print(f"Data: {n_samples} cells x {n_features} features")
            print("Building graph...")

        edge_index, edge_weight = self._build_graph(X_input)
        if verbose:
            print(f"Graph: {edge_index.shape[1]} edges")

        self.x = torch.tensor(X_input, dtype=torch.float32, device=self.device)
        self.edge_index = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)

        self.encoder = ContrastiveEncoder(
            n_features, self.hidden_dim, self.proj_dim, self.encoder_type, self.num_layers
        ).to(self.device)

        if verbose:
            print("\nPhase 1: Contrastive pretraining...")
        profiler.snapshot("pretrain_start")
        cb_list.on_phase_begin('pretrain')
        self.pretrain_losses = self._pretrain(self.x, self.edge_index, self.edge_weight, verbose, cb_list)
        cb_list.on_phase_end('pretrain', {'final_loss': self.pretrain_losses[-1] if self.pretrain_losses else 0})
        profiler.snapshot("pretrain_end")

        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(self.x, self.edge_index, self.edge_weight)
            embeddings = F.normalize(h, dim=1).cpu().numpy()

        if self.n_clusters is None:
            if verbose:
                print("\nEstimating number of clusters...")
            self.n_clusters, _ = tune_n_clusters(embeddings)
            if verbose:
                print(f"Estimated k = {self.n_clusters}")

        self.ssc = SelfSupervisedClustering(self.n_clusters, self.hidden_dim).to(self.device)

        if verbose:
            print("\nInitializing clusters with K-means...")
        self.ssc.initialize_centers(h)

        if verbose:
            print("\nPhase 2: Self-supervised clustering...")
        profiler.snapshot("ssc_start")
        cb_list.on_phase_begin('ssc')
        refiner = ClusterRefiner(
            self.n_clusters, self.ssc_epochs, self.ssc_batch_size, device=self.device
        )
        self.labels_, self.ssc_losses = refiner.refine(
            self.encoder, self.ssc, self.x, self.edge_index, self.edge_weight, verbose
        )
        cb_list.on_phase_end('ssc', {'final_loss': self.ssc_losses[-1] if self.ssc_losses else 0})
        profiler.snapshot("ssc_end")

        # Store soft assignments for confidence scores
        with torch.no_grad():
            h = self.encoder(self.x, self.edge_index, self.edge_weight)
            self._soft_assignments, _ = self.ssc(h)
            self._soft_assignments = self._soft_assignments.cpu().numpy()

        if y is not None and verbose:
            evaluate_clustering(y, self.labels_, embeddings)

        cb_list.on_train_end({'labels': self.labels_})

        if memory_profiling:
            profiler.report()

        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, y, **kwargs)
        return self.labels_

    def predict(self, X: np.ndarray, preprocess: bool = True, n_pca: int = 50) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.encoder is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if preprocess:
            _, X_input = preprocess_data(X, n_pca_components=n_pca)
        else:
            X_input = X

        edge_index, edge_weight = self._build_graph(X_input)
        x = torch.tensor(X_input, dtype=torch.float32, device=self.device)

        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x, edge_index.to(self.device), edge_weight.to(self.device))
            return self.ssc.get_labels(h)

    def get_embeddings(self) -> np.ndarray:
        """Get learned embeddings."""
        if self.encoder is None:
            raise RuntimeError("Model not fitted.")

        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(self.x, self.edge_index, self.edge_weight)
            return F.normalize(h, dim=1).cpu().numpy()

    def get_confidence_scores(self) -> np.ndarray:
        """
        Get confidence scores for cluster assignments.

        Returns the maximum soft assignment probability for each cell,
        indicating how confident the model is in its cluster prediction.

        Returns
        -------
        np.ndarray
            Confidence scores in [0, 1] for each cell
        """
        if self._soft_assignments is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return np.max(self._soft_assignments, axis=1)

    def get_soft_assignments(self) -> np.ndarray:
        """
        Get soft cluster assignment probabilities.

        Returns the full probability distribution over clusters for each cell.

        Returns
        -------
        np.ndarray
            Soft assignment matrix of shape (n_cells, n_clusters)
        """
        if self._soft_assignments is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._soft_assignments.copy()

    def predict_with_confidence(
        self,
        X: np.ndarray,
        preprocess: bool = True,
        n_pca: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict cluster labels with confidence scores for new data.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        preprocess : bool
            Whether to preprocess the data
        n_pca : int
            Number of PCA components

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels
        confidence : np.ndarray
            Confidence scores for each prediction
        soft_assignments : np.ndarray
            Full soft assignment probabilities
        """
        if self.encoder is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if preprocess:
            _, X_input = preprocess_data(X, n_pca_components=n_pca)
        else:
            X_input = X

        edge_index, edge_weight = self._build_graph(X_input)
        x = torch.tensor(X_input, dtype=torch.float32, device=self.device)

        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x, edge_index.to(self.device), edge_weight.to(self.device))
            soft_assignments, labels = self.ssc(h)
            soft_assignments = soft_assignments.cpu().numpy()
            labels = labels.cpu().numpy()

        confidence = np.max(soft_assignments, axis=1)
        return labels, confidence, soft_assignments

    def save(self, path: str) -> None:
        """
        Save model to checkpoint.

        Parameters
        ----------
        path : str
            Path to save the checkpoint
        """
        torch.save({
            'encoder': self.encoder.state_dict(),
            'ssc': self.ssc.state_dict(),
            'config': {
                'n_clusters': self.n_clusters,
                'hidden_dim': self.hidden_dim,
                'proj_dim': self.proj_dim,
                'num_layers': self.num_layers,
                'encoder_type': self.encoder_type
            },
            'labels': self.labels_,
            'soft_assignments': self._soft_assignments
        }, path)

    def load(self, path: str, input_dim: Optional[int] = None) -> 'ScGCL':
        """
        Load model from checkpoint.

        Parameters
        ----------
        path : str
            Path to the saved checkpoint
        input_dim : int, optional
            Input feature dimension. Required if loading for inference on new data.

        Returns
        -------
        self
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt['config']

        self.n_clusters = cfg['n_clusters']
        self.hidden_dim = cfg['hidden_dim']
        self.proj_dim = cfg.get('proj_dim', self.proj_dim)
        self.num_layers = cfg.get('num_layers', self.num_layers)
        self.encoder_type = cfg.get('encoder_type', self.encoder_type)
        self.labels_ = ckpt.get('labels')
        self._soft_assignments = ckpt.get('soft_assignments')

        # Reconstruct encoder if input_dim is provided
        if input_dim is not None:
            self.encoder = ContrastiveEncoder(
                input_dim, self.hidden_dim, self.proj_dim,
                self.encoder_type, self.num_layers
            ).to(self.device)
            self.encoder.load_state_dict(ckpt['encoder'])

            self.ssc = SelfSupervisedClustering(
                self.n_clusters, self.hidden_dim
            ).to(self.device)
            self.ssc.load_state_dict(ckpt['ssc'])

        return self


def run_experiment(X: np.ndarray, y: np.ndarray, n_runs: int = 10, **kwargs) -> Dict:
    """Run multiple experiments and report statistics."""
    analyzer = ClusterStabilityAnalyzer(n_runs)

    for i in range(n_runs):
        print(f"\n{'='*50}\nRun {i+1}/{n_runs}\n{'='*50}")
        model = ScGCL(seed=42 + i, **kwargs)
        labels = model.fit_predict(X, y, verbose=False)
        analyzer.add_result(y, labels, model.get_embeddings())

    print("\n" + "="*50 + "\nFINAL RESULTS\n" + "="*50)
    analyzer.print_summary()
    return analyzer.summarize()
