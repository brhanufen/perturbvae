"""Conditional VAE for perturbation-response prediction.

  * Encoder    q(z | x)              from an observed expression vector.
  * Prior      p(z | pert_emb)       from a perturbation embedding.
  * Decoder    x_hat(z, pert_emb)    reconstructs expression, conditioned on
                                     BOTH the latent and the perturbation.

Conditioning the decoder on the perturbation (not only the prior) lets it
directly modulate the output, and lets us read off a low-variance perturbation
effect: with the same latent draws, decode under the perturbation vs under the
control and take the difference (``predict_delta``). Using a perturbation
EMBEDDING (not a one-hot over training perturbations) is what makes prediction
for an unseen perturbation possible.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn


class _MLP(nn.Module):
    def __init__(self, sizes: List[int], act=nn.ReLU) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConditionalVAE(nn.Module):
    def __init__(self, n_genes: int, pert_dim: int, latent: int = 64, hidden: int = 512) -> None:
        super().__init__()
        self.latent = latent
        self.pert_dim = pert_dim
        self.n_genes = n_genes
        self.encoder = _MLP([n_genes, hidden, hidden, 2 * latent])
        self.prior = _MLP([pert_dim, hidden, 2 * latent])
        self.decoder = _MLP([latent + pert_dim, hidden, hidden, n_genes])

    def _split(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        mu, logvar = h[..., : self.latent], h[..., self.latent:]
        return mu, logvar.clamp(-8.0, 8.0)

    def forward(self, x: Tensor, pert_emb: Tensor):
        q_mu, q_lv = self._split(self.encoder(x))
        p_mu, p_lv = self._split(self.prior(pert_emb))
        z = q_mu + torch.randn_like(q_mu) * torch.exp(0.5 * q_lv)
        x_hat = self.decoder(torch.cat([z, pert_emb], dim=-1))
        return x_hat, (q_mu, q_lv), (p_mu, p_lv)

    @torch.no_grad()
    def predict_delta(self, pert_emb: Tensor, ctrl_emb: Tensor, n_samples: int = 30) -> Tensor:
        """Predicted per-gene expression change (delta) for one perturbation.

        Draw latents from the control prior, then decode under the perturbation
        and under the control with the SAME latents (paired) and average the
        difference. ``pert_emb`` and ``ctrl_emb`` are 1-D ``[pert_dim]``.
        """
        c_mu, c_lv = self._split(self.prior(ctrl_emb.unsqueeze(0)))
        std = torch.exp(0.5 * c_lv)
        z = c_mu + torch.randn(n_samples, self.latent, device=ctrl_emb.device) * std
        pe = pert_emb.unsqueeze(0).expand(n_samples, -1)
        ce = ctrl_emb.unsqueeze(0).expand(n_samples, -1)
        x_pert = self.decoder(torch.cat([z, pe], dim=-1)).mean(0)
        x_ctrl = self.decoder(torch.cat([z, ce], dim=-1)).mean(0)
        return x_pert - x_ctrl

    @torch.no_grad()
    def predict_response(self, pert_emb: Tensor, n_samples: int = 30) -> Tensor:
        """For a model trained on deltas (cell minus control mean): the predicted
        delta is just the decoded response. Sample z from the perturbation prior
        and decode under the perturbation. Returns ``[n_genes]``."""
        p_mu, p_lv = self._split(self.prior(pert_emb.unsqueeze(0)))
        std = torch.exp(0.5 * p_lv)
        z = p_mu + torch.randn(n_samples, self.latent, device=pert_emb.device) * std
        pe = pert_emb.unsqueeze(0).expand(n_samples, -1)
        return self.decoder(torch.cat([z, pe], dim=-1)).mean(0)


def vae_loss(
    x_hat: Tensor,
    x: Tensor,
    q: Tuple[Tensor, Tensor],
    p: Tuple[Tensor, Tensor],
    beta: float = 1.0,
    free_bits: float = 0.0,
):
    """Reconstruction (MSE) + KL(q(z|x) || p(z|pert)).  Returns (loss, recon, kl)."""
    q_mu, q_lv = q
    p_mu, p_lv = p
    recon = ((x_hat - x) ** 2).sum(-1).mean()
    kl = 0.5 * (p_lv - q_lv + (q_lv.exp() + (q_mu - p_mu) ** 2) / p_lv.exp() - 1.0).sum(-1)
    if free_bits > 0.0:
        kl = torch.clamp(kl, min=free_bits)
    total = recon + beta * kl.mean()
    return total, float(recon.item()), float(kl.mean().item())
