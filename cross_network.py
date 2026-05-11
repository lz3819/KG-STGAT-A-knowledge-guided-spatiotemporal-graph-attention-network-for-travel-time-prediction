import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetwork(nn.Module):
    """
    Cross Network for KG-STGAT.

    Key fixes:
    - Keep all 7 common features (vehicle + time) without summing time embeddings together.
    - Trajectory feature construction uses [distance_emb, segment_emb, tiled_common(7*emb)] -> latent_dim.
    """

    def __init__(self, feature_dim: int, latent_dim: int, trajectory_length: int, num_common_features: int = 7):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.trajectory_length = trajectory_length
        self.num_common_features = num_common_features

        self.embedding = nn.Embedding(feature_dim, latent_dim)
        self.linear = nn.Embedding(feature_dim, 1)

        self.process_trajectory = nn.Sequential(
            nn.Linear((2 + self.num_common_features) * latent_dim, latent_dim),
            nn.ReLU()
        )

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, feature_indices: torch.Tensor):
        batch_size = feature_indices.size(0)
        emb = self.embedding(feature_indices)  # (B, field_cnt, K)

        common_features = emb[:, :self.num_common_features, :]  # (B, 7, K)
        common_flat = common_features.reshape(batch_size, 1, -1)  # (B, 1, 7K)

        traj_distances = emb[:, self.num_common_features:self.num_common_features + self.trajectory_length, :]
        traj_routes = emb[:, self.num_common_features + self.trajectory_length:, :]

        traj_combined = torch.cat([traj_distances, traj_routes], dim=-1)  # (B, L, 2K)
        common_tiled = common_flat.expand(-1, self.trajectory_length, -1)  # (B, L, 7K)
        traj_with_common = torch.cat([traj_combined, common_tiled], dim=-1)
        trajectory_features = self.process_trajectory(traj_with_common)

        linear_term = self.linear(feature_indices).sum(dim=1)  # (B, 1)

        sum_emb = emb.sum(dim=1)
        sum_emb_sq = sum_emb * sum_emb
        emb_sq_sum = (emb * emb).sum(dim=1)
        interaction = 0.5 * (sum_emb_sq - emb_sq_sum).sum(dim=1, keepdim=True)
        fm_output = linear_term + interaction

        return fm_output, trajectory_features, linear_term, interaction


class SemanticTransformer(nn.Module):
    """Semantic Transformer using 1D CNN to extract global semantics"""

    def __init__(self, d_model, trajectory_length, kernel_size=3, stride=2):
        super(SemanticTransformer, self).__init__()
        self.trajectory_length = trajectory_length

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.activation = nn.ReLU()

    def forward(self, trajectory_features):
        x = trajectory_features.transpose(1, 2)
        x = self.activation(self.conv1d(x))
        global_semantic = torch.mean(x, dim=2)
        return global_semantic


class ITTEHead(nn.Module):
    """
    Individual Travel Time Estimation Head.

    Fixed in this version:
    - Route-level context is injected back into segment prediction via a softmax
      allocation, instead of maintaining an unconstrained separate total head.
    - The model output can therefore be made exactly route-consistent downstream.
    """

    def __init__(self, d_model, trajectory_length):
        super(ITTEHead, self).__init__()
        self.trajectory_length = trajectory_length

        self.segment_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.total_linear = nn.Linear(d_model, 1)
        self.segment_allocator = nn.Linear(d_model, 1)

    def forward(self, segment_features, global_features, linear_term, interaction_term):
        segment_base = self.segment_predictor(segment_features).squeeze(-1)  # (B, L)
        route_bias = self.total_linear(global_features) + (linear_term + interaction_term)  # (B, 1)
        alloc = torch.softmax(self.segment_allocator(segment_features).squeeze(-1), dim=1)  # (B, L)
        segment_times = segment_base + alloc * route_bias
        total_time_proxy = segment_times.sum(dim=1, keepdim=True)
        return segment_times, total_time_proxy
