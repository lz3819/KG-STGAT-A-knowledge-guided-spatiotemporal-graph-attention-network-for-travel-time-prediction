# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_network import CrossNetwork, SemanticTransformer, ITTEHead
from st_modules import STBlock, BridgeTrans, STEmbedding


def _as_tensor(x, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.tensor(float(x), device=like.device, dtype=like.dtype)


def _lognormal_multiplier(shape, sigma: float, like: torch.Tensor) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return torch.ones(shape, device=like.device, dtype=like.dtype)
    eps = torch.randn(shape, device=like.device, dtype=like.dtype)
    sig = _as_tensor(sigma, like)
    return torch.exp(sig * eps - 0.5 * (sig ** 2))


def _inv_softplus(x: float) -> float:
    x = float(x)
    if x <= 0:
        return -20.0
    return float(math.log(math.expm1(x) + 1e-12))


def _get_time_sequence(batch: Dict[str, torch.Tensor], primary_key: str, fallback_key: str, B: int, T: int, N: int, device):
    if primary_key in batch:
        x = batch[primary_key]
    elif fallback_key in batch:
        x = batch[fallback_key]
    else:
        return torch.zeros((B, T), dtype=torch.long, device=device)

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.long, device=device)
    x = x.to(device=device, dtype=torch.long)

    if x.dim() == 0:
        return x.view(1, 1).expand(B, T)
    if x.dim() == 1:
        if x.numel() == T:
            return x.view(1, T).expand(B, T)
        if x.numel() == B:
            return x.view(B, 1).expand(B, T)
        return torch.zeros((B, T), dtype=torch.long, device=device)
    if x.dim() == 2:
        if x.shape == (B, T):
            return x
        if x.shape == (T, N):
            return x[:, 0].view(1, T).expand(B, T)
        if x.shape == (B, N):
            return x[:, :1].expand(B, T)
        if x.shape[0] == 1 and x.shape[1] == T:
            return x.expand(B, T)
        y = torch.zeros((B, T), dtype=torch.long, device=device)
        b = min(B, x.shape[0])
        t = min(T, x.shape[1])
        y[:b, :t] = x[:b, :t]
        return y
    if x.dim() == 3:
        if x.shape[0] == B and x.shape[1] == T:
            return x[:, :, 0]
    return torch.zeros((B, T), dtype=torch.long, device=device)


class KGSTGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.N = int(config.site_num)
        self.P = int(config.input_length)
        self.Q = int(config.output_length)
        self.L = int(config.trajectory_length)
        self.K = int(config.emb_size)
        self.num_heads = int(getattr(config, 'num_heads', 4))
        self.num_blocks = int(getattr(config, 'num_blocks', 2))
        self.normalize = bool(getattr(config, 'normalize', True))

        unit_min = float(getattr(config, 'flow_unit_minutes', 15))
        minute_bins = int(max(1, round(60.0 / max(unit_min, 1e-6))))
        minute_bins = int(max(1, min(minute_bins, 60)))

        self.ste_layer = STEmbedding(self.N, self.K, minute_bins=minute_bins)
        self.flow_proj = nn.Sequential(
            nn.Linear(1, self.K),
            nn.ReLU(),
            nn.Linear(self.K, self.K),
        )
        self.st_block = STBlock(self.K, self.num_heads, self.num_blocks, float(getattr(config, 'dropout', 0.0)))
        self.bridge_trans = BridgeTrans(self.K, self.num_heads, self.num_blocks, float(getattr(config, 'dropout', 0.0)))
        self.flow_predictor = nn.Linear(self.K, 1)

        self.cross_network = CrossNetwork(
            feature_dim=int(config.feature_tra),
            latent_dim=self.K,
            trajectory_length=self.L,
            num_common_features=7,
        )
        self.semantic_transformer = SemanticTransformer(self.K, self.L)
        self.itte_head = ITTEHead(self.K, self.L)

        self.use_fusion = bool(getattr(config, 'use_fusion', True))
        self.use_impedance_aux = bool(getattr(config, 'use_impedance_aux', True))
        self.impedance_aux_log1p = bool(getattr(config, 'impedance_aux_log1p', True))
        self.bpr_trainable = bool(getattr(config, 'bpr_trainable', True))
        self.bpr_trainable_physics = bool(getattr(config, 'bpr_trainable_physics', False))
        self.bpr_stochastic = bool(getattr(config, 'bpr_stochastic', True))
        self.bpr_mc_samples = max(int(getattr(config, 'bpr_mc_samples', 1)), 1)
        self.bpr_alpha_noise = float(getattr(config, 'bpr_alpha_noise', 0.05))
        self.bpr_beta_noise = float(getattr(config, 'bpr_beta_noise', 0.02))
        self.bpr_capacity_noise = float(getattr(config, 'bpr_capacity_noise', 0.05))
        self.bpr_speed_noise = float(getattr(config, 'bpr_speed_noise', 0.01))

        if self.use_impedance_aux:
            self.imp_mlp = nn.Sequential(
                nn.Linear(3, self.K),
                nn.ReLU(),
                nn.Linear(self.K, self.K),
            )
            self.imp_ln = nn.LayerNorm(self.K)
        else:
            self.imp_mlp = None
            self.imp_ln = None

        if self.use_fusion and self.bpr_trainable_physics:
            self.bpr_cap_scale_raw = nn.Embedding(self.N, 1)
            self.bpr_t0_scale_raw = nn.Embedding(self.N, 1)
            init_raw = torch.tensor(_inv_softplus(1.0), dtype=torch.float32)
            nn.init.constant_(self.bpr_cap_scale_raw.weight, init_raw)
            nn.init.constant_(self.bpr_t0_scale_raw.weight, init_raw)
        else:
            self.bpr_cap_scale_raw = None
            self.bpr_t0_scale_raw = None

        alpha0 = float(getattr(self.cfg, 'bpr_init_alpha', 0.15))
        beta0 = float(getattr(self.cfg, 'bpr_init_beta', 4.0))
        if self.bpr_trainable:
            self.bpr_alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha0), dtype=torch.float32))
            self.bpr_beta_raw = nn.Parameter(torch.tensor(_inv_softplus(max(beta0 - 1.0, 1e-3)), dtype=torch.float32))
        else:
            self.register_buffer('bpr_alpha_fixed', torch.tensor(alpha0, dtype=torch.float32))
            self.register_buffer('bpr_beta_fixed', torch.tensor(beta0, dtype=torch.float32))

        if self.use_fusion:
            self.seg_gate_phys_encoder = nn.Sequential(
                nn.Linear(4, self.K),
                nn.ReLU(),
                nn.Linear(self.K, self.K),
            )
            self.total_gate_phys_encoder = nn.Sequential(
                nn.Linear(4, self.K),
                nn.ReLU(),
                nn.Linear(self.K, self.K),
            )
            self.gate_seg = nn.Sequential(
                nn.Linear(2 * self.K + 1, self.K),
                nn.ReLU(),
                nn.Linear(self.K, 1),
                nn.Sigmoid(),
            )
            self.gate_total = nn.Sequential(
                nn.Linear(2 * self.K + 1, self.K),
                nn.ReLU(),
                nn.Linear(self.K, 1),
                nn.Sigmoid(),
            )
        else:
            self.seg_gate_phys_encoder = None
            self.total_gate_phys_encoder = None
            self.gate_seg = None
            self.gate_total = None

        if self.normalize:
            self.register_buffer('flow_mean_buf', torch.as_tensor(getattr(config, 'flow_mean'), dtype=torch.float32).view(1, self.N))
            self.register_buffer('flow_std_buf', torch.as_tensor(getattr(config, 'flow_std'), dtype=torch.float32).view(1, self.N))
            self.register_buffer('tt_seg_mean_buf', torch.as_tensor(getattr(config, 'tt_seg_mean'), dtype=torch.float32).view(1, self.L))
            self.register_buffer('tt_seg_std_buf', torch.as_tensor(getattr(config, 'tt_seg_std'), dtype=torch.float32).view(1, self.L))
            self.register_buffer('tt_total_mean_buf', torch.as_tensor(getattr(config, 'tt_total_mean'), dtype=torch.float32).view(1, 1))
            self.register_buffer('tt_total_std_buf', torch.as_tensor(getattr(config, 'tt_total_std'), dtype=torch.float32).view(1, 1))
        else:
            self.register_buffer('flow_mean_buf', torch.zeros(1, self.N))
            self.register_buffer('flow_std_buf', torch.ones(1, self.N))
            self.register_buffer('tt_seg_mean_buf', torch.zeros(1, self.L))
            self.register_buffer('tt_seg_std_buf', torch.ones(1, self.L))
            self.register_buffer('tt_total_mean_buf', torch.zeros(1, 1))
            self.register_buffer('tt_total_std_buf', torch.ones(1, 1))

    def _get_bpr_alpha_beta(self):
        if self.bpr_trainable:
            alpha = F.softplus(self.bpr_alpha_raw)
            beta = F.softplus(self.bpr_beta_raw) + 1.0
            return alpha, beta
        return self.bpr_alpha_fixed, self.bpr_beta_fixed

    def _sample_bpr_params(self, like: torch.Tensor, B: int) -> dict:
        shape = (B, 1, 1)
        return {
            'm_alpha': _lognormal_multiplier(shape, self.bpr_alpha_noise, like),
            'm_beta': _lognormal_multiplier(shape, self.bpr_beta_noise, like),
            'm_cap': _lognormal_multiplier(shape, self.bpr_capacity_noise, like),
            'm_spd': _lognormal_multiplier(shape, self.bpr_speed_noise, like),
        }

    def _seg_norm_to_min(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return x * (self.tt_seg_std_buf + 1e-6) + self.tt_seg_mean_buf

    def _seg_min_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self.tt_seg_mean_buf) / (self.tt_seg_std_buf + 1e-6)

    def _total_norm_to_min(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return x * (self.tt_total_std_buf + 1e-6) + self.tt_total_mean_buf

    def _total_min_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self.tt_total_mean_buf) / (self.tt_total_std_buf + 1e-6)

    def _flow_norm_to_real(self, x: torch.Tensor, seg: torch.Tensor | None = None) -> torch.Tensor:
        if not self.normalize:
            return x
        if seg is None:
            return x * (self.flow_std_buf.unsqueeze(1) + 1e-6) + self.flow_mean_buf.unsqueeze(1)
        mean_seg = torch.gather(self.flow_mean_buf.expand(seg.size(0), -1), 1, seg)
        std_seg = torch.gather(self.flow_std_buf.expand(seg.size(0), -1), 1, seg)
        return x * (std_seg.unsqueeze(1) + 1e-6) + mean_seg.unsqueeze(1)

    def _compute_bpr_components(
        self,
        flow_vph: torch.Tensor,
        distance_m: torch.Tensor,
        lanes,
        capacity_per_lane_vph,
        alpha,
        beta,
        speed_limit_kmh,
        cap_scale=None,
        t0_scale=None,
    ):
        like = flow_vph
        device = like.device
        dtype = like.dtype

        lanes_t = _as_tensor(lanes, like).clamp(min=1e-6)
        cap_lane_t = _as_tensor(capacity_per_lane_vph, like).clamp(min=1e-6)
        speed_kmh_t = _as_tensor(speed_limit_kmh, like).clamp(min=1e-6)
        alpha_t = _as_tensor(alpha, like)
        beta_t = _as_tensor(beta, like).clamp(min=1e-6)

        speed_mps = speed_kmh_t * 1000.0 / 3600.0
        t0_min = (distance_m.to(device=device, dtype=dtype) / speed_mps) / 60.0
        if t0_scale is not None:
            ts = t0_scale.to(device=device, dtype=dtype)
            if ts.dim() == 2 and t0_min.dim() == 3:
                ts = ts.unsqueeze(1)
            t0_min = t0_min * ts

        base_cap = lanes_t * cap_lane_t
        cap = base_cap
        if cap_scale is not None:
            cs = cap_scale.to(device=device, dtype=dtype)
            if cs.dim() == 2 and flow_vph.dim() == 3:
                cs = cs.unsqueeze(1)
            cap = cap * cs

        vc = flow_vph / cap.clamp(min=1e-6)
        tt = t0_min * (1.0 + alpha_t * torch.pow(vc.clamp(min=0.0), beta_t))
        delay_ratio = tt / t0_min.clamp(min=1e-6)
        excess_delay = (tt - t0_min).clamp(min=0.0)
        return {
            'tt': tt,
            't0': t0_min,
            'cap': cap,
            'vc': vc,
            'delay_ratio': delay_ratio,
            'excess_delay': excess_delay,
        }

    def _select_route_profile(self, phys_q: dict, unit_min: float):
        tt_q = phys_q['tt']
        B, Q, L = tt_q.shape
        device = tt_q.device
        dtype = tt_q.dtype
        cum_min = torch.zeros(B, 1, device=device, dtype=dtype)
        idx_list = []
        out = {k: [] for k in ['tt', 't0', 'vc', 'delay_ratio', 'excess_delay', 'flow_vph']}

        for l in range(L):
            arr_idx = torch.floor(cum_min / max(unit_min, 1e-6)).long().clamp(0, Q - 1)
            idx_list.append(arr_idx.squeeze(1))
            gather_idx = arr_idx.unsqueeze(-1)
            out['tt'].append(torch.gather(phys_q['tt'][:, :, l:l + 1], 1, gather_idx).squeeze(1))
            out['t0'].append(torch.gather(phys_q['t0'][:, :, l:l + 1], 1, gather_idx).squeeze(1))
            out['vc'].append(torch.gather(phys_q['vc'][:, :, l:l + 1], 1, gather_idx).squeeze(1))
            out['delay_ratio'].append(torch.gather(phys_q['delay_ratio'][:, :, l:l + 1], 1, gather_idx).squeeze(1))
            out['excess_delay'].append(torch.gather(phys_q['excess_delay'][:, :, l:l + 1], 1, gather_idx).squeeze(1))
            out['flow_vph'].append(torch.gather(phys_q['flow_vph'][:, :, l:l + 1], 1, gather_idx).squeeze(1))
            cum_min = cum_min + out['tt'][-1]

        out = {k: torch.cat(v, dim=1) for k, v in out.items()}
        out['arrival_bin'] = torch.stack(idx_list, dim=1)
        out['total_min'] = out['tt'].sum(dim=1, keepdim=True)
        return out

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_flow = batch['flow']
        B = x_flow.size(0)
        device = x_flow.device
        total_steps = self.P + self.Q

        hour_seq = _get_time_sequence(batch, 'flow_hour_seq', 'flow_hour', B, total_steps, self.N, device).clamp(0, 23)
        minute_seq = _get_time_sequence(batch, 'flow_minute_seq', 'flow_minute', B, total_steps, self.N, device)
        ste = self.ste_layer(B, total_steps, hour_seq, minute_seq, device=device)
        ste_hist = ste[:, :self.P, :, :]
        ste_fut = ste[:, self.P:, :, :]

        flow_feat = self.flow_proj(x_flow.unsqueeze(-1))
        hist_states = self.st_block(flow_feat, ste_hist)
        fut_states = self.bridge_trans(hist_states, ste_hist, ste_fut)
        pred_flow = self.flow_predictor(fut_states).squeeze(-1)

        feat = batch['tra_features'].long()
        _, traj_feat, linear_term, inter_term = self.cross_network(feat)
        global_feat = self.semantic_transformer(traj_feat)
        pred_seg_raw, _ = self.itte_head(traj_feat, global_feat, linear_term, inter_term)
        pred_total = self._total_min_to_norm(self._seg_norm_to_min(pred_seg_raw).sum(dim=1, keepdim=True))
        pred_seg = pred_seg_raw

        need_phys = self.use_fusion or self.use_impedance_aux
        phys = None
        if need_phys:
            seg = batch['segments']
            if seg.dim() == 1:
                seg = seg.unsqueeze(0).expand(B, -1)
            seg = seg.long()

            dist_m = batch['distances']
            if dist_m.dim() == 1:
                dist_m = dist_m.unsqueeze(0).expand(B, -1)
            dist_m = dist_m.float()

            pf_route = torch.gather(pred_flow, dim=2, index=seg.unsqueeze(1).expand(-1, self.Q, -1))
            seg_flow_real_q = self._flow_norm_to_real(pf_route, seg=seg)

            unit_min = float(getattr(self.cfg, 'flow_unit_minutes', 15))
            unit_min = max(unit_min, 1e-6)
            seg_flow_vph_q = (seg_flow_real_q.clamp(min=0.0)) * (60.0 / unit_min)

            alpha, beta = self._get_bpr_alpha_beta()
            alpha = alpha.to(device=device, dtype=torch.float32)
            beta = beta.to(device=device, dtype=torch.float32)

            cap_scale = None
            t0_scale = None
            if self.bpr_trainable_physics and self.bpr_cap_scale_raw is not None:
                cap_scale = F.softplus(self.bpr_cap_scale_raw(seg)).squeeze(-1).clamp(0.1, 10.0)
                t0_scale = F.softplus(self.bpr_t0_scale_raw(seg)).squeeze(-1).clamp(0.1, 10.0)

            dist_q = dist_m.unsqueeze(1).expand(-1, self.Q, -1).to(device=device, dtype=seg_flow_vph_q.dtype)
            do_stoch = self.training and self.bpr_stochastic
            mc = self.bpr_mc_samples if do_stoch else 1
            comp_samples = []
            for _ in range(mc):
                if do_stoch:
                    ms = self._sample_bpr_params(seg_flow_vph_q, B)
                    alpha_s = alpha * ms['m_alpha']
                    beta_s = beta * ms['m_beta']
                    cap_lane_s = float(getattr(self.cfg, 'capacity_per_lane_vph', 2000.0)) * ms['m_cap']
                    speed_s = float(getattr(self.cfg, 'speed_limit_kmh', 120.0)) * ms['m_spd']
                else:
                    alpha_s = alpha
                    beta_s = beta
                    cap_lane_s = float(getattr(self.cfg, 'capacity_per_lane_vph', 2000.0))
                    speed_s = float(getattr(self.cfg, 'speed_limit_kmh', 120.0))
                comp = self._compute_bpr_components(
                    flow_vph=seg_flow_vph_q,
                    distance_m=dist_q,
                    lanes=float(getattr(self.cfg, 'lanes', 2)),
                    capacity_per_lane_vph=cap_lane_s,
                    alpha=alpha_s,
                    beta=beta_s,
                    speed_limit_kmh=speed_s,
                    cap_scale=cap_scale,
                    t0_scale=t0_scale,
                )
                comp['flow_vph'] = seg_flow_vph_q
                comp_samples.append(comp)

            phys_q = {}
            for k in comp_samples[0].keys():
                phys_q[k] = torch.stack([c[k] for c in comp_samples], dim=0).mean(dim=0)

            route_phys = self._select_route_profile(phys_q, unit_min=unit_min)
            phys_seg_min = route_phys['tt']
            phys_total_min = route_phys['total_min']
            phys_seg = self._seg_min_to_norm(phys_seg_min)
            phys_total = self._total_min_to_norm(phys_total_min)

            phys = {
                'phys_seg_min_q': phys_q['tt'],
                'phys_seg_min': phys_seg_min,
                'phys_total_min': phys_total_min,
                'phys_seg': phys_seg,
                'phys_total': phys_total,
                'phys_vc': route_phys['vc'],
                'phys_delay_ratio': route_phys['delay_ratio'],
                'phys_excess_delay': route_phys['excess_delay'],
                'phys_t0_min': route_phys['t0'],
                'phys_flow_vph': route_phys['flow_vph'],
                'arrival_bin': route_phys['arrival_bin'],
                'bpr_alpha': alpha,
                'bpr_beta': beta,
                'bpr_cap_scale': cap_scale,
                'bpr_t0_scale': t0_scale,
            }

        if self.use_impedance_aux and (phys is not None):
            excess_delay = phys['phys_excess_delay']
            if self.impedance_aux_log1p:
                excess_delay = torch.log1p(excess_delay.clamp(min=0.0))
            imp_feat = torch.stack([
                phys['phys_vc'].clamp(min=0.0),
                phys['phys_delay_ratio'].clamp(min=0.0),
                excess_delay,
            ], dim=-1)
            imp_emb = self.imp_mlp(imp_feat)
            traj_feat = self.imp_ln(traj_feat + imp_emb)
            global_feat = self.semantic_transformer(traj_feat)
            pred_seg_raw, _ = self.itte_head(traj_feat, global_feat, linear_term, inter_term)
            pred_total = self._total_min_to_norm(self._seg_norm_to_min(pred_seg_raw).sum(dim=1, keepdim=True))
            pred_seg = pred_seg_raw

        out = {
            'pred_flow': pred_flow,
            'pred_seg': pred_seg,
            'pred_total': pred_total,
            'final_seg': pred_seg,
            'final_total': pred_total,
        }

        if self.use_fusion and (phys is not None):
            seg_phys_in = torch.stack([
                phys['phys_seg'],
                phys['phys_vc'],
                phys['phys_delay_ratio'],
                torch.log1p(phys['phys_excess_delay'].clamp(min=0.0)),
            ], dim=-1)
            total_phys_in = torch.cat([
                phys['phys_total'],
                phys['phys_vc'].mean(dim=1, keepdim=True),
                phys['phys_delay_ratio'].mean(dim=1, keepdim=True),
                phys['phys_excess_delay'].sum(dim=1, keepdim=True),
            ], dim=-1)

            seg_phys_emb = self.seg_gate_phys_encoder(seg_phys_in)
            total_phys_emb = self.total_gate_phys_encoder(total_phys_in)

            seg_gate_in = torch.cat([
                traj_feat,
                seg_phys_emb,
                (pred_seg - phys['phys_seg']).abs().unsqueeze(-1),
            ], dim=-1)
            total_gate_in = torch.cat([
                global_feat,
                total_phys_emb,
                (pred_total - phys['phys_total']).abs(),
            ], dim=-1)

            g_seg = self.gate_seg(seg_gate_in).squeeze(-1)
            g_total = self.gate_total(total_gate_in)

            fused_seg_norm = g_seg * pred_seg + (1.0 - g_seg) * phys['phys_seg']
            fused_total_norm_raw = g_total * pred_total + (1.0 - g_total) * phys['phys_total']

            fused_seg_min = self._seg_norm_to_min(fused_seg_norm)
            target_total_min = self._total_norm_to_min(fused_total_norm_raw)
            current_total_min = fused_seg_min.sum(dim=1, keepdim=True).clamp(min=1e-6)
            scale = (target_total_min / current_total_min).clamp(min=0.2, max=5.0)
            final_seg_min = fused_seg_min * scale
            final_total_min = final_seg_min.sum(dim=1, keepdim=True)

            final_seg = self._seg_min_to_norm(final_seg_min)
            final_total = self._total_min_to_norm(final_total_min)

            out.update({
                'phys_seg': phys['phys_seg'],
                'phys_total': phys['phys_total'],
                'phys_seg_min_q': phys['phys_seg_min_q'],
                'phys_seg_min': phys['phys_seg_min'],
                'phys_total_min': phys['phys_total_min'],
                'phys_vc': phys['phys_vc'],
                'phys_delay_ratio': phys['phys_delay_ratio'],
                'phys_excess_delay': phys['phys_excess_delay'],
                'final_seg': final_seg,
                'final_total': final_total,
                'gate_total': g_total,
                'gate_seg': g_seg,
                'arrival_bin': phys['arrival_bin'],
            })
        return out


def create_model(config):
    return KGSTGAT(config)
