"""scGCL: Main CPU implementation."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
from tqdm import tqdm

from .models.encoder import ContrastiveEncoder
from .utils.data import preprocess_data
from .utils.graph import build_adaptive_knn_graph, compute_snn_weights
from .utils.augmentation import ContrastiveAugmentation
from .utils.evaluation import evaluate_clustering, ClusterStabilityAnalyzer
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

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.encoder = None
        self.ssc = None
        self.x = None
        self.edge_index = None
        self.edge_weight = None
        self.pretrain_losses = []
        self.ssc_losses = []

    def _build_graph(self, X: np.ndarray):
        edge_index, edge_weight = build_adaptive_knn_graph(X, k_min=5, k_max=self.k_neighbors * 2)
        edge_weight = compute_snn_weights(edge_index, X.shape[0])
        return edge_index, edge_weight

    def _pretrain(self, x, edge_index, edge_weight, verbose=True):
        self.encoder.train()

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = CombinedContrastiveLoss(
            self.temperature, self.tau_plus, self.lambda_align, self.lambda_uniform
        )
        augmentor = ContrastiveAugmentation()

        losses = []
        warmup_epochs = min(10, self.pretrain_epochs // 5)
        iterator = tqdm(range(self.pretrain_epochs), desc="Pretraining") if verbose else range(self.pretrain_epochs)

        for epoch in iterator:
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

        return losses

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            preprocess: bool = True, n_pca: int = 50, verbose: bool = True) -> 'ScGCL':
        """Fit the clustering model."""
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
        self.pretrain_losses = self._pretrain(self.x, self.edge_index, self.edge_weight, verbose)

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
        refiner = ClusterRefiner(
            self.n_clusters, self.ssc_epochs, self.ssc_batch_size, device=self.device
        )
        self.labels_, self.ssc_losses = refiner.refine(
            self.encoder, self.ssc, self.x, self.edge_index, self.edge_weight, verbose
        )

        if y is not None and verbose:
            evaluate_clustering(y, self.labels_, embeddings)

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

    def save(self, path: str):
        """Save model."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'ssc': self.ssc.state_dict(),
            'config': {
                'n_clusters': self.n_clusters, 'hidden_dim': self.hidden_dim,
                'proj_dim': self.proj_dim, 'num_layers': self.num_layers,
                'encoder_type': self.encoder_type
            },
            'labels': self.labels_
        }, path)

    def load(self, path: str):
        """Load model."""
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt['config']
        self.n_clusters = cfg['n_clusters']
        self.hidden_dim = cfg['hidden_dim']
        self.labels_ = ckpt['labels']


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
