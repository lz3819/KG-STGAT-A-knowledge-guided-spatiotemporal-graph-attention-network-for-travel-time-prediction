# -*- coding: utf-8 -*-
"""
Spatio-temporal modules used by KG-STGAT speed predictor.

Fixed in this version:
- STEmbedding now supports step-wise temporal indices instead of repeating a
  single hour/minute token across the whole history+future window.
- Accepted inputs include (B,T,N), (B,T), (T,N), (T,), (B,N), (N,), scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention, SpatialAttention, TemporalAttention, FusionGate


class STEmbedding(nn.Module):
    """Spatio-Temporal Embedding (STE), returning (B, T, N, d_model)."""

    def __init__(self, num_nodes: int, d_model: int, *, minute_bins: int = 4, max_steps: int = 512):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.minute_bins = int(max(1, minute_bins))
        self.max_steps = int(max_steps)

        self.node_emb = nn.Embedding(self.num_nodes, self.d_model)
        self.hour_emb = nn.Embedding(24, self.d_model)
        self.minute_emb = nn.Embedding(self.minute_bins, self.d_model)
        self.pos_emb = nn.Embedding(self.max_steps, self.d_model)

    def _to_btn(self, x: torch.Tensor, B: int, T: int, N: int, device) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long, device=device)
        x = x.to(device=device, dtype=torch.long)

        if x.dim() == 0:
            return x.view(1, 1, 1).expand(B, T, N)
        if x.dim() == 1:
            if x.numel() == T:
                return x.view(1, T, 1).expand(B, T, N)
            if x.numel() == N:
                return x.view(1, 1, N).expand(B, T, N)
            if x.numel() == B:
                return x.view(B, 1, 1).expand(B, T, N)
            return torch.zeros((B, T, N), dtype=torch.long, device=device)
        if x.dim() == 2:
            if x.shape == (B, T):
                return x.unsqueeze(-1).expand(B, T, N)
            if x.shape == (T, N):
                return x.unsqueeze(0).expand(B, T, N)
            if x.shape == (B, N):
                return x.unsqueeze(1).expand(B, T, N)
            if x.shape[0] == 1 and x.shape[1] == T:
                return x.view(1, T, 1).expand(B, T, N)
            if x.shape[0] == 1 and x.shape[1] == N:
                return x.view(1, 1, N).expand(B, T, N)
            if x.shape[0] == B and x.shape[1] == 1:
                return x.view(B, 1, 1).expand(B, T, N)
            y = torch.zeros((B, T, N), dtype=torch.long, device=device)
            b = min(B, x.shape[0])
            t = min(T, x.shape[1])
            y[:b, :t, :] = x[:b, :t].unsqueeze(-1).expand(b, t, N)
            return y
        if x.dim() == 3:
            y = x[:B, :T, :N]
            if y.shape[0] < B:
                y = F.pad(y, (0, 0, 0, 0, 0, B - y.shape[0]), value=0)
            if y.shape[1] < T:
                y = F.pad(y, (0, 0, 0, T - y.shape[1], 0, 0), value=0)
            if y.shape[2] < N:
                y = F.pad(y, (0, N - y.shape[2], 0, 0, 0, 0), value=0)
            return y
        return x.view(B, T, N)

    def forward(self, batch_size: int, time_steps: int, hour: torch.Tensor, minute: torch.Tensor, device=None):
        if device is None:
            device = hour.device if torch.is_tensor(hour) else (minute.device if torch.is_tensor(minute) else 'cpu')

        B = int(batch_size)
        T = int(time_steps)
        N = self.num_nodes

        hour = self._to_btn(hour, B, T, N, device).clamp(0, 23)
        minute = self._to_btn(minute, B, T, N, device).clamp(0, self.minute_bins - 1)

        node_ids = torch.arange(N, device=device, dtype=torch.long)
        node_e = self.node_emb(node_ids).view(1, 1, N, self.d_model).expand(B, T, N, self.d_model)

        hour_e = self.hour_emb(hour)
        minute_e = self.minute_emb(minute)

        pos_ids = torch.arange(min(T, self.max_steps), device=device, dtype=torch.long)
        pos_e = self.pos_emb(pos_ids).view(1, -1, 1, self.d_model).expand(B, -1, N, self.d_model)
        if pos_e.shape[1] < T:
            pad_len = T - pos_e.shape[1]
            last = pos_e[:, -1:, :, :].expand(B, pad_len, N, self.d_model)
            pos_e = torch.cat([pos_e, last], dim=1)

        return node_e + hour_e + minute_e + pos_e


class _STUnit(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.spatial = SpatialAttention(d_model, num_heads, dropout)
        self.temporal = TemporalAttention(d_model, num_heads, dropout)
        self.fuse = FusionGate(d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x, ste)
        temporal, _ = self.temporal(x)
        fused = self.fuse(spatial, temporal)
        return self.norm(x + self.dropout(fused))


class STBlock(nn.Module):
    """Stacked spatio-temporal blocks."""

    def __init__(self, d_model: int, num_heads: int, num_blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([_STUnit(d_model, num_heads, dropout) for _ in range(int(max(1, num_blocks)))])

    def forward(self, x: torch.Tensor, ste: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h, ste)
        return h


class _BridgeUnit(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross_attn(q, kv, kv)
        ff = self.ff(out)
        return self.norm(out + self.dropout(ff))


class BridgeTrans(nn.Module):
    """Cross-attend from future STE to historical (states+STE), per node."""

    def __init__(self, d_model: int, num_heads: int, num_blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([_BridgeUnit(d_model, num_heads, dropout) for _ in range(int(max(1, num_blocks)))])

    def forward(self, historical_states: torch.Tensor, ste_historical: torch.Tensor, ste_future: torch.Tensor) -> torch.Tensor:
        B, P, N, D = historical_states.shape
        Q = ste_future.shape[1]

        kv = (historical_states + ste_historical).permute(0, 2, 1, 3).contiguous().view(B * N, P, D)
        q = ste_future.permute(0, 2, 1, 3).contiguous().view(B * N, Q, D)

        h = q
        for blk in self.blocks:
            h = blk(h, kv)

        return h.view(B, N, Q, D).permute(0, 2, 1, 3).contiguous()
