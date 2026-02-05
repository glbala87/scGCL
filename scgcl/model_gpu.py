"""scGCL GPU: GPU-accelerated version for large datasets."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, List, Dict
from tqdm import tqdm

from .models.encoder import ContrastiveEncoder
from .utils.data import preprocess_data
from .utils.graph import build_adaptive_knn_graph, compute_snn_weights
from .utils.augmentation import ContrastiveAugmentation
from .utils.evaluation import evaluate_clustering, ClusterStabilityAnalyzer
from .clustering.ssc import SelfSupervisedClustering, ClusterRefiner, tune_n_clusters


def tiled_contrastive_loss(z1, z2, temperature=0.5, tile_size=1024):
    """Memory-efficient contrastive loss using tiled computation."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    batch_size = z1.shape[0]
    device = z1.device

    total_loss = 0.0

    for i in range(0, batch_size, tile_size):
        i_end = min(i + tile_size, batch_size)
        z1_tile, z2_tile = z1[i:i_end], z2[i:i_end]
        tile_batch = z1_tile.shape[0]

        pos_sim = (z1_tile * z2_tile).sum(dim=1) / temperature

        neg_log_sum = []
        for j in range(0, batch_size, tile_size):
            j_end = min(j + tile_size, batch_size)

            sim_12 = torch.mm(z1_tile, z2[j:j_end].t()) / temperature
            sim_11 = torch.mm(z1_tile, z1[j:j_end].t()) / temperature

            if i == j:
                mask = torch.eye(tile_batch, device=device, dtype=torch.bool)
                sim_12.masked_fill_(mask, float('-inf'))
                sim_11.masked_fill_(mask, float('-inf'))

            neg_log_sum.append(torch.logsumexp(torch.cat([sim_12, sim_11], dim=1), dim=1))

        neg_term = torch.logsumexp(torch.stack(neg_log_sum, dim=1), dim=1)
        total_loss += (-pos_sim + neg_term).mean() * tile_batch

    return total_loss / batch_size


class TiledContrastiveLoss(nn.Module):
    """GPU-optimized contrastive loss with tiling."""

    def __init__(self, temperature=0.5, lambda_align=1.0, lambda_uniform=1.0, tile_size=1024):
        super().__init__()
        self.temperature = temperature
        self.lambda_align = lambda_align
        self.lambda_uniform = lambda_uniform
        self.tile_size = tile_size

    def forward(self, z1, z2, warmup_factor=1.0):
        l_con = tiled_contrastive_loss(z1, z2, self.temperature, self.tile_size)

        z1_n, z2_n = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        l_align = (z1_n - z2_n).norm(p=2, dim=1).pow(2).mean()

        # Simplified uniformity
        z = torch.cat([z1_n, z2_n], dim=0)
        sq_pdist = torch.cdist(z[:min(1024, len(z))], z[:min(1024, len(z))], p=2).pow(2)
        mask = torch.eye(sq_pdist.shape[0], device=z.device, dtype=torch.bool)
        sq_pdist.masked_fill_(mask, float('inf'))
        l_uniform = torch.exp(-2 * sq_pdist).mean().log()

        total = l_con + self.lambda_align * l_align + self.lambda_uniform * warmup_factor * l_uniform

        return {'total': total, 'contrastive': l_con, 'alignment': l_align, 'uniformity': l_uniform}


class ScGCLGPU:
    """
    GPU-accelerated scGCL for large datasets.

    Additional parameters over ScGCL:
    - use_amp: Use automatic mixed precision
    - tile_size: Tile size for memory-efficient computation
    - device_ids: GPU device IDs for DataParallel
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
        lambda_align: float = 1.0,
        lambda_uniform: float = 1.0,
        pretrain_epochs: int = 100,
        ssc_epochs: int = 500,
        batch_size: int = 256,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        tile_size: int = 2048,
        device_ids: Optional[List[int]] = None,
        seed: int = 42
    ):
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.lambda_align = lambda_align
        self.lambda_uniform = lambda_uniform
        self.pretrain_epochs = pretrain_epochs
        self.ssc_epochs = ssc_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_amp = use_amp and torch.cuda.is_available()
        self.tile_size = tile_size
        self.device_ids = device_ids
        self.seed = seed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.encoder = None
        self.ssc = None
        self.scaler = GradScaler() if self.use_amp else None
        self.pretrain_losses = []
        self.ssc_losses = []

    def _build_graph(self, X):
        edge_index, edge_weight = build_adaptive_knn_graph(X, k_min=5, k_max=self.k_neighbors * 2)
        edge_weight = compute_snn_weights(edge_index, X.shape[0])
        return edge_index, edge_weight

    def _pretrain(self, x, edge_index, edge_weight, verbose=True):
        self.encoder.train()

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = TiledContrastiveLoss(self.temperature, self.lambda_align, self.lambda_uniform, self.tile_size)
        augmentor = ContrastiveAugmentation()

        losses = []
        warmup_epochs = min(10, self.pretrain_epochs // 5)
        iterator = tqdm(range(self.pretrain_epochs), desc="Pretraining") if verbose else range(self.pretrain_epochs)

        for epoch in iterator:
            optimizer.zero_grad()
            view1, view2 = augmentor(x, edge_index, edge_weight)
            x1, ei1, ew1 = view1
            x2, ei2, ew2 = view2

            if self.use_amp:
                with autocast():
                    _, _, _, z1, z2 = self.encoder.contrastive_forward(x, ei1, ei2, ew1, ew2, x1, x2)
                    warmup = min(1.0, epoch / warmup_epochs)
                    loss_dict = criterion(z1, z2, warmup)
                    loss = loss_dict['total']

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
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

    def fit(self, X, y=None, preprocess=True, n_pca=50, verbose=True):
        """Fit the model."""
        if verbose:
            print(f"Device: {self.device}")
            if self.use_amp:
                print("Mixed precision enabled")

        if preprocess:
            if verbose:
                print("Preprocessing...")
            _, X_input = preprocess_data(X, n_pca_components=n_pca)
        else:
            X_input = X

        n_samples, n_features = X_input.shape
        if verbose:
            print(f"Data: {n_samples} cells x {n_features} features")
            print("Building graph...")

        edge_index, edge_weight = self._build_graph(X_input)

        self.x = torch.tensor(X_input, dtype=torch.float32, device=self.device)
        self.edge_index = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)

        self.encoder = ContrastiveEncoder(
            n_features, self.hidden_dim, self.proj_dim, self.encoder_type, self.num_layers
        ).to(self.device)

        if self.device_ids and len(self.device_ids) > 1:
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.device_ids)

        if verbose:
            print("\nPhase 1: Contrastive pretraining...")
        self.pretrain_losses = self._pretrain(self.x, self.edge_index, self.edge_weight, verbose)

        encoder_module = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder
        encoder_module.eval()

        with torch.no_grad():
            h = encoder_module(self.x, self.edge_index, self.edge_weight)
            embeddings = F.normalize(h, dim=1).cpu().numpy()

        if self.n_clusters is None:
            if verbose:
                print("\nEstimating clusters...")
            self.n_clusters, _ = tune_n_clusters(embeddings)
            if verbose:
                print(f"k = {self.n_clusters}")

        self.ssc = SelfSupervisedClustering(self.n_clusters, self.hidden_dim).to(self.device)
        self.ssc.initialize_centers(h)

        if verbose:
            print("\nPhase 2: SSC refinement...")
        refiner = ClusterRefiner(self.n_clusters, self.ssc_epochs, self.batch_size, device=str(self.device))
        self.labels_, self.ssc_losses = refiner.refine(
            encoder_module, self.ssc, self.x, self.edge_index, self.edge_weight, verbose
        )

        if y is not None and verbose:
            evaluate_clustering(y, self.labels_, embeddings)

        return self

    def fit_predict(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.labels_

    def get_embeddings(self):
        encoder = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder
        encoder.eval()
        with torch.no_grad():
            h = encoder(self.x, self.edge_index, self.edge_weight)
            return F.normalize(h, dim=1).cpu().numpy()

    def save(self, path):
        encoder = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder
        torch.save({
            'encoder': encoder.state_dict(),
            'ssc': self.ssc.state_dict(),
            'config': {'n_clusters': self.n_clusters, 'hidden_dim': self.hidden_dim},
            'labels': self.labels_
        }, path)


def run_gpu_experiment(X, y, n_runs=10, **kwargs) -> Dict:
    """Run multiple GPU experiments."""
    analyzer = ClusterStabilityAnalyzer(n_runs)

    for i in tqdm(range(n_runs), desc="Runs"):
        model = ScGCLGPU(seed=42 + i, **kwargs)
        labels = model.fit_predict(X, y, verbose=False)
        analyzer.add_result(y, labels, model.get_embeddings())

    print("\n" + "="*50 + "\nFINAL RESULTS\n" + "="*50)
    analyzer.print_summary()
    return analyzer.summarize()
