# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

try:
    from visualize_results import run_visualization_test
except Exception:
    run_visualization_test = None


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _denorm_flow(x_norm: np.ndarray, cfg) -> np.ndarray:
    if not getattr(cfg, 'normalize', True):
        return x_norm
    mean = np.asarray(getattr(cfg, 'flow_mean'))
    std = np.asarray(getattr(cfg, 'flow_std'))
    return x_norm * (std + 1e-6) + mean


def _denorm_total_time(x_norm: np.ndarray, cfg) -> np.ndarray:
    if not getattr(cfg, 'normalize', True):
        return x_norm
    mean = float(np.asarray(getattr(cfg, 'tt_total_mean')).reshape(-1)[0])
    std = float(np.asarray(getattr(cfg, 'tt_total_std')).reshape(-1)[0])
    return x_norm * (std + 1e-6) + mean


def _denorm_seg_time(x_norm: np.ndarray, cfg) -> np.ndarray:
    if not getattr(cfg, 'normalize', True):
        return x_norm
    mean = np.asarray(getattr(cfg, 'tt_seg_mean')).reshape(1, -1)
    std = np.asarray(getattr(cfg, 'tt_seg_std')).reshape(1, -1)
    return x_norm * (std + 1e-6) + mean


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, *, mape_min_target: float | None = None, mape_eps: float = 1e-6) -> Dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    if mape_min_target is not None:
        mask = np.abs(y_true) >= float(mape_min_target)
        mape = float('nan') if mask.sum() == 0 else float(np.mean(np.abs(err[mask] / (y_true[mask] + mape_eps))) * 100.0)
    else:
        mape = float(np.mean(np.abs(err / (y_true + mape_eps))) * 100.0)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def _get_loss_fn(cfg):
    lt = str(getattr(cfg, 'loss_type', 'mae')).lower()
    if lt in ['mae', 'l1']:
        return nn.L1Loss(reduction='none')
    if lt in ['mse', 'l2']:
        return nn.MSELoss(reduction='none')
    if lt in ['huber', 'smoothl1']:
        delta = float(getattr(cfg, 'huber_delta', 1.0))
        return nn.SmoothL1Loss(beta=delta, reduction='none')
    return nn.L1Loss(reduction='none')


def _fix_total_shape(t: torch.Tensor) -> torch.Tensor:
    if t is None:
        return t
    while t.dim() > 2 and t.size(-1) == 1:
        t = t.squeeze(-1)
    if t.dim() == 1:
        t = t.view(-1, 1)
    elif t.dim() == 2:
        if t.size(1) != 1:
            t = t[:, :1]
    else:
        t = t.view(t.size(0), -1)
        t = t[:, :1]
    return t


def _masked_loss(loss_fn, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = loss_fn(pred, target)
    if mask is None:
        return loss.mean()
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(loss)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def _masked_mean_over_time(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    numer = (x * mask).sum(axis=1)
    denom = np.clip(mask.sum(axis=1), 1e-6, None)
    return numer / denom


@torch.no_grad()
def evaluate(model, loader, cfg) -> Tuple[float, Dict]:
    model.eval()
    loss_fn = _get_loss_fn(cfg)

    lam_f = float(getattr(cfg, 'lambda_flow', 0.2))
    lam_t = float(getattr(cfg, 'lambda_total', 0.4))
    lam_s = float(getattr(cfg, 'lambda_segment', 0.4))
    flow_mape_min_target = float(getattr(cfg, 'flow_mape_min_target', 1e-6))

    losses = []
    all_total_true, all_total_pred = [], []
    all_seg_true, all_seg_pred = [], []
    all_flow_true, all_flow_pred = [], []

    for batch in loader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(cfg.device)

        out = model(batch)
        y_flow = batch['y_flow']
        y_flow_mask = batch.get('y_flow_mask')
        y_total = _fix_total_shape(batch['y_total_time'])
        y_seg = batch['y_seg_time']

        pred_flow = out['pred_flow']
        pred_total = _fix_total_shape(out['final_total'])
        pred_seg = out['final_seg']

        l_flow = _masked_loss(loss_fn, pred_flow, y_flow, y_flow_mask)
        l_total = _masked_loss(loss_fn, pred_total, y_total)
        l_seg = _masked_loss(loss_fn, pred_seg, y_seg)
        loss = lam_f * l_flow + lam_t * l_total + lam_s * l_seg
        losses.append(float(loss.detach().cpu().item()))

        total_true_min = _to_np(_fix_total_shape(batch['y_total_time_min']))
        seg_true_min = _to_np(batch['y_seg_time_min'])
        total_pred_min = _denorm_total_time(_to_np(pred_total), cfg).reshape(-1, 1)
        seg_pred_min = _denorm_seg_time(_to_np(pred_seg), cfg)
        all_total_true.append(total_true_min)
        all_total_pred.append(total_pred_min)
        all_seg_true.append(seg_true_min)
        all_seg_pred.append(seg_pred_min)

        seg_cols = _to_np(batch['segments']).astype(int)
        if seg_cols.ndim == 1:
            seg_cols = np.tile(seg_cols.reshape(1, -1), (pred_flow.shape[0], 1))

        flow_true_full = _denorm_flow(_to_np(y_flow), cfg)
        flow_pred_full = _denorm_flow(_to_np(pred_flow), cfg)
        flow_mask_full = _to_np(y_flow_mask) if y_flow_mask is not None else np.ones_like(flow_true_full, dtype=np.float32)

        flow_true = _masked_mean_over_time(flow_true_full, flow_mask_full)
        flow_pred = _masked_mean_over_time(flow_pred_full, flow_mask_full)

        gathered_true = np.take_along_axis(flow_true, seg_cols, axis=1)
        gathered_pred = np.take_along_axis(flow_pred, seg_cols, axis=1)
        all_flow_true.append(gathered_true)
        all_flow_pred.append(gathered_pred)

    val_loss = float(np.mean(losses)) if losses else 0.0
    total_true = np.concatenate(all_total_true, axis=0)
    total_pred = np.concatenate(all_total_pred, axis=0)
    seg_true = np.concatenate(all_seg_true, axis=0)
    seg_pred = np.concatenate(all_seg_pred, axis=0)
    flow_true = np.concatenate(all_flow_true, axis=0)
    flow_pred = np.concatenate(all_flow_pred, axis=0)

    metrics = {
        'total': _metrics(total_true, total_pred),
        'seg_pooled': _metrics(seg_true.reshape(-1, 1), seg_pred.reshape(-1, 1)),
        'flow_route': _metrics(flow_true.reshape(-1, 1), flow_pred.reshape(-1, 1), mape_min_target=flow_mape_min_target),
        'per_segment': {},
    }

    route_seg_ids = list(getattr(cfg, 'route_segment_ids', list(range(seg_true.shape[1]))))
    for j in range(seg_true.shape[1]):
        metrics['per_segment'][str(route_seg_ids[j])] = _metrics(seg_true[:, j:j + 1], seg_pred[:, j:j + 1])

    return val_loss, metrics


def train_model(model, train_loader, val_loader, cfg):
    model.to(cfg.device)
    opt = Adam(model.parameters(), lr=float(cfg.learning_rate))
    loss_fn = _get_loss_fn(cfg)

    lam_f = float(getattr(cfg, 'lambda_flow', 0.2))
    lam_t = float(getattr(cfg, 'lambda_total', 0.4))
    lam_s = float(getattr(cfg, 'lambda_segment', 0.4))

    best_val = float('inf')
    patience = 10
    bad = 0

    save_dir = cfg.save_path
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(1, int(cfg.epoch) + 1):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), ncols=120, desc=f'Epoch {epoch}')
        losses = []
        lf_list, lt_list, ls_list = [], [], []

        for batch in pbar:
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(cfg.device)

            out = model(batch)
            y_flow = batch['y_flow']
            y_flow_mask = batch.get('y_flow_mask')
            y_total = _fix_total_shape(batch['y_total_time'])
            y_seg = batch['y_seg_time']

            pred_flow = out['pred_flow']
            pred_total = _fix_total_shape(out['final_total'])
            pred_seg = out['final_seg']

            l_flow = _masked_loss(loss_fn, pred_flow, y_flow, y_flow_mask)
            l_total = _masked_loss(loss_fn, pred_total, y_total)
            l_seg = _masked_loss(loss_fn, pred_seg, y_seg)
            loss = lam_f * l_flow + lam_t * l_total + lam_s * l_seg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            lf_list.append(float(l_flow.detach().cpu().item()))
            lt_list.append(float(l_total.detach().cpu().item()))
            ls_list.append(float(l_seg.detach().cpu().item()))
            pbar.set_postfix({'loss': f'{np.mean(losses):.3f}', 'flow': f'{np.mean(lf_list):.3f}', 'total': f'{np.mean(lt_list):.3f}', 'seg': f'{np.mean(ls_list):.3f}'})

        val_loss, val_metrics = evaluate(model, val_loader, cfg)
        flow_mape_min_target = float(getattr(cfg, 'flow_mape_min_target', 1e-6))

        print(f"\nEpoch {epoch}/{int(cfg.epoch)}")
        print(f"  Train Loss: {np.mean(losses):.6f} (Flow: {np.mean(lf_list):.6f}, Total: {np.mean(lt_list):.6f}, Seg: {np.mean(ls_list):.6f})")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Total Travel Time (minutes) - MAE: {val_metrics['total']['MAE']:.2f}, RMSE: {val_metrics['total']['RMSE']:.2f}, MAPE: {val_metrics['total']['MAPE']:.2f}%")
        print(f"  Segment Travel Time (minutes, pooled) - MAE: {val_metrics['seg_pooled']['MAE']:.2f}, RMSE: {val_metrics['seg_pooled']['RMSE']:.2f}, MAPE: {val_metrics['seg_pooled']['MAPE']:.2f}%")
        print(f"  Flow (route segments only, horizon-mean, MAPE abs(y_true)≥{flow_mape_min_target:g}) - MAE: {val_metrics['flow_route']['MAE']:.2f}, RMSE: {val_metrics['flow_route']['RMSE']:.2f}, MAPE: {val_metrics['flow_route']['MAPE']:.2f}%")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.6f})")
        else:
            bad += 1
            print(f"  No improvement for {bad} epochs")
            if bad >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs\n")
                break
    return model


@torch.no_grad()
def test_model(model, test_loader, cfg):
    model.eval()
    do_viz = bool(getattr(cfg, 'visualize', True)) and (run_visualization_test is not None)
    max_points = int(getattr(cfg, 'viz_max_points', 50000))
    save_npz = bool(getattr(cfg, 'viz_save_npz', True))
    out_dir = getattr(cfg, 'viz_dir', None)

    if do_viz:
        res = run_visualization_test(model, test_loader, cfg, out_dir=out_dir, max_points=max_points, save_npz=save_npz)
        metrics = res['metrics']
        out_dir = res['out_dir']
    else:
        _loss, metrics = evaluate(model, test_loader, cfg)
        out_dir = None

    flow_mape_min_target = float(getattr(cfg, 'flow_mape_min_target', 1e-6))
    print('\nTest Results:')
    print('-' * 70)
    print('\nTotal Travel Time (minutes):')
    print(f"  MAE:  {metrics['total']['MAE']:.2f} min")
    print(f"  RMSE: {metrics['total']['RMSE']:.2f} min")
    print(f"  MAPE: {metrics['total']['MAPE']:.2f}%")

    print('\nSegment Travel Time (minutes):')
    print('  Overall (all segments pooled):')
    print(f"    MAE:  {metrics['seg_pooled']['MAE']:.2f} min")
    print(f"    RMSE: {metrics['seg_pooled']['RMSE']:.2f} min")
    print(f"    MAPE: {metrics['seg_pooled']['MAPE']:.2f}%")
    print('\n  Per-segment:')
    for seg_id, m in metrics['per_segment'].items():
        print(f"    Segment {seg_id}: MAE={m['MAE']:.2f} min, RMSE={m['RMSE']:.2f} min, MAPE={m['MAPE']:.2f}%")

    print('\nFlow Prediction (route segments only, horizon-mean):')
    print(f"  MAE:  {metrics['flow_route']['MAE']:.2f}")
    print(f"  RMSE: {metrics['flow_route']['RMSE']:.2f}")
    print(f"  MAPE (abs(y_true)≥{flow_mape_min_target:g}): {metrics['flow_route']['MAPE']:.2f}%")
    print('-' * 70)
    if out_dir is not None:
        print(f"\n✓ Saved prediction CSVs and scatter plots to:\n  {out_dir}\n")
