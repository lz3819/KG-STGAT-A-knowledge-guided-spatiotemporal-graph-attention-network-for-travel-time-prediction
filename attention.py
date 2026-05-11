# -*- coding: utf-8 -*-
"""
Attention Mechanisms for KG-STGAT
Including: Multi-head Attention, Spatial Attention, Temporal Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # ✅ safer broadcast
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B,1,q,k)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        output = self.layer_norm(output + residual)
        return output, attention_weights


class SpatialAttention(nn.Module):
    """Spatial Attention for modeling spatial correlations between road segments"""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super(SpatialAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.input_projection = nn.Linear(2 * d_model, d_model)

    def forward(self, x, ste):
        batch_size, time_steps, num_nodes, d_model = x.size()

        x_with_ste = torch.cat([x, ste], dim=-1)
        x_reshaped = x_with_ste.view(batch_size * time_steps, num_nodes, 2 * d_model)

        x_reshaped = self.input_projection(x_reshaped)

        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)

        residual = attn_output
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + residual)

        output = output.view(batch_size, time_steps, num_nodes, d_model)
        return output


class TemporalAttention(nn.Module):
    """Temporal Attention for modeling temporal correlations across time steps"""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super(TemporalAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, time_steps, num_nodes, d_model = x.size()

        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, time_steps, d_model)

        attn_output, attention_weights = self.attention(x_reshaped, x_reshaped, x_reshaped)

        residual = attn_output
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + residual)

        output = output.view(batch_size, num_nodes, time_steps, d_model).permute(0, 2, 1, 3).contiguous()
        return output, attention_weights


class FusionGate(nn.Module):
    """Fusion Gate Network for combining spatial and temporal features (learnable)"""

    def __init__(self, d_model):
        super(FusionGate, self).__init__()
        self.gate_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, spatial_features, temporal_features):
        gate = torch.sigmoid(self.gate_proj(torch.cat([spatial_features, temporal_features], dim=-1)))
        fused = gate * spatial_features + (1 - gate) * temporal_features
        return fused


class HolisticAttention(nn.Module):
    """Holistic Attention for connecting road network states with travel features"""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super(HolisticAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, travel_features, road_states):
        """
        Args:
            travel_features: (B, L, d_model)
            road_states: (B, T, L, d_model)
        Returns:
            output: (B, L, d_model)  ✅ always 3D
            attention_weights
        """
        batch_size, traj_len, d_model = travel_features.size()
        _, total_time, _, _ = road_states.size()

        road_states_reshaped = road_states.permute(0, 2, 1, 3).contiguous().view(batch_size * traj_len, total_time, d_model)
        travel_features_reshaped = travel_features.view(batch_size * traj_len, 1, d_model)

        attn_output, attention_weights = self.attention(
            travel_features_reshaped,
            road_states_reshaped,
            road_states_reshaped
        )

        residual = attn_output
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + residual)

        # ✅ Always (B, L, d_model)
        output = output.view(batch_size, traj_len, d_model)
        return output, attention_weights
