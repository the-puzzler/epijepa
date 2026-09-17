"""Epiplexity: how much of an embedding a ridge readout from a frozen random CNN can predict.

score(z, a) = 0.5 * log2 det(I + eta * W^T W),   W = a @ (z - mean(z))

where a is the ridge map from the reservoir's features on the same batch. A collapsed
embedding scores ~0; spreading predictable structure across more directions raises it.
Maximising this score, traded off against a view-invariance loss, is the whole method.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

RIDGE = 3.0  # readout regularisation
ETA = 30.0   # saturation inside the log-det


class ChannelNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class Reservoir(nn.Module):
    """Frozen random CNN. Stride 2 after the first stage, so `channels` sets the depth."""

    def __init__(self, width, seed, channels=(16, 32, 64, 64)):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            layers, inc = [], 3
            for i, c in enumerate(channels):
                layers += [nn.Conv2d(inc, c, 3, 1 if i == 0 else 2, 1), ChannelNorm(), nn.ELU()]
                inc = c
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(2), nn.Flatten(),
                                     nn.Linear(channels[-1] * 4, width))
        self.requires_grad_(False)

    def forward(self, x):
        return self.net(x)


def ridge_map(h, lam=RIDGE):
    """(H^T H + lam I)^-1 H^T on standardised features, via QR. Shape [features, batch]."""
    h = ((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])).double()
    aug = torch.cat([h, math.sqrt(lam) * torch.eye(h.shape[1], dtype=h.dtype, device=h.device)])
    q, r = torch.linalg.qr(aug, mode="reduced")
    return torch.linalg.solve_triangular(r, q[: h.shape[0]].T, upper=True)


def score(z, a, eta=ETA):
    z = z.double()
    w = a @ (z - z.mean(0))
    eye = torch.eye(w.shape[1], dtype=w.dtype, device=w.device)
    return 0.5 * torch.linalg.slogdet(eye + eta * w.T @ w)[1] / math.log(2)


def epiplexity(z, x, reservoir):
    """Score of embedding z (differentiable) against the frozen reservoir's view of images x."""
    with torch.no_grad():
        a = ridge_map(reservoir(x))
    return score(z, a)
