"""Contrastive loss functions for self-supervised learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) contrastive loss."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        batch_size = z1.shape[0]
        device = z1.device

        if normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device, dtype=torch.bool)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device, dtype=torch.bool)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device, dtype=torch.bool)

        pos_sim = sim[pos_mask].view(2 * batch_size, 1)
        loss = -pos_sim + torch.logsumexp(sim, dim=1, keepdim=True)

        return loss.mean()


class DebiasedContrastiveLoss(nn.Module):
    """Debiased contrastive loss accounting for false negatives."""

    def __init__(self, temperature: float = 0.5, tau_plus: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.tau_plus = tau_plus

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        batch_size = z1.shape[0]
        device = z1.device

        if normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        pos_sim = (z1 * z2).sum(dim=1) / self.temperature

        neg_sim_12 = torch.mm(z1, z2.t()) / self.temperature
        neg_sim_11 = torch.mm(z1, z1.t()) / self.temperature
        neg_sim_22 = torch.mm(z2, z2.t()) / self.temperature

        mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        neg_sim_11.masked_fill_(mask, float('-inf'))
        neg_sim_22.masked_fill_(mask, float('-inf'))
        neg_sim_12.masked_fill_(mask, float('-inf'))

        neg_sim = torch.cat([neg_sim_12, neg_sim_11, neg_sim_22], dim=1)
        N = neg_sim.shape[1]
        Ng = max(N * (1 - self.tau_plus) - self.tau_plus, 1)

        neg_exp = torch.exp(neg_sim)
        neg_term = neg_exp.sum(dim=1) / Ng

        loss = -pos_sim + torch.log(torch.exp(pos_sim) + neg_term)
        return loss.mean()


class AlignmentLoss(nn.Module):
    """Alignment loss - positive pairs should be close."""

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        return (z1 - z2).norm(p=self.alpha, dim=1).pow(self.alpha).mean()


class UniformityLoss(nn.Module):
    """Uniformity loss - embeddings should be spread on hypersphere."""

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, z: torch.Tensor, normalize: bool = True, tile_size: int = 1024) -> torch.Tensor:
        if normalize:
            z = F.normalize(z, dim=1)

        n = z.shape[0]

        if n <= tile_size:
            sq_pdist = torch.cdist(z, z, p=2).pow(2)
            mask = torch.eye(n, device=z.device, dtype=torch.bool)
            sq_pdist.masked_fill_(mask, float('inf'))
            return torch.exp(-self.t * sq_pdist).mean().log()
        else:
            total, count = 0.0, 0
            for i in range(0, n, tile_size):
                z_i = z[i:i+tile_size]
                for j in range(0, n, tile_size):
                    z_j = z[j:j+tile_size]
                    sq_pdist = torch.cdist(z_i, z_j, p=2).pow(2)
                    if i == j:
                        mask = torch.eye(z_i.shape[0], device=z.device, dtype=torch.bool)
                        sq_pdist.masked_fill_(mask, float('inf'))
                    total += torch.exp(-self.t * sq_pdist).sum()
                    count += sq_pdist.numel() - (z_i.shape[0] if i == j else 0)
            return (total / count).log()


class CombinedContrastiveLoss(nn.Module):
    """Combined contrastive loss with alignment and uniformity."""

    def __init__(
        self,
        temperature: float = 0.5,
        tau_plus: float = 0.1,
        lambda_align: float = 1.0,
        lambda_uniform: float = 1.0,
        use_debiased: bool = True
    ):
        super().__init__()

        self.contrastive_loss = DebiasedContrastiveLoss(temperature, tau_plus) if use_debiased else InfoNCELoss(temperature)
        self.alignment_loss = AlignmentLoss()
        self.uniformity_loss = UniformityLoss()
        self.lambda_align = lambda_align
        self.lambda_uniform = lambda_uniform

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, warmup_factor: float = 1.0) -> dict:
        l_con = self.contrastive_loss(z1, z2)
        l_align = self.alignment_loss(z1, z2)
        z_combined = torch.cat([z1, z2], dim=0)
        l_uniform = self.uniformity_loss(z_combined)

        total = l_con + self.lambda_align * l_align + self.lambda_uniform * warmup_factor * l_uniform

        return {'total': total, 'contrastive': l_con, 'alignment': l_align, 'uniformity': l_uniform}
