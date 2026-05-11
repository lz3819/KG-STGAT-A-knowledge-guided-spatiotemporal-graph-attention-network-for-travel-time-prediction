# -*- coding: utf-8 -*-
"""
KG-STGAT Data Loader (REAL)

Key behaviors (matches your intention):
- start_time is kept as-is for the travel-time (ITTE) module input features:
  week/day/hour/minute/second are extracted from the original start_time.
- For flow/BPR alignment, start_time is mapped to the latest flow timestamp <= start_time,
  i.e. 12:09 uses 12:00 flow when the flow interval is 15 minutes.

Fixes included in this version:
1) ✅ Flow window slicing no longer "shifts" windows near boundaries (prevents leakage / label mismatch).
   We now use strict windows with padding:
     x_flow uses [t0-P, t0) (pad at beginning if needed),
     y_flow uses [t0, t0+Q) (pad at end if needed).
2) ✅ Flow interval is inferred from flow timestamps and written back to config.flow_unit_minutes,
   so BPR uses the correct aggregation interval.
3) ✅ Route segment mapping: required route segments are guaranteed to be kept in flow columns;
   if not found, we raise an error instead of unsafe fallback.
4) ✅ Trajectory file path resolution is more robust than simple replace("_1.csv", ...).
5) ✅ trajectory_time_unit='auto' is inferred more safely using distance+speed plausibility (train split only).

Batch keys (compatible with trainer.py / model.py):
  flow, y_flow, y_seg_time, y_total_time, y_seg_time_min, y_total_time_min,
  segments, distances, tra_features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ----------------------------- CSV utils -----------------------------
def _read_csv_auto(file_path: str, encoding: str = "auto") -> pd.DataFrame:
    if encoding != "auto":
        return pd.read_csv(file_path, encoding=encoding)
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def _resolve_traj_file(template: str, route_id: int) -> str:
    """
    Resolve trajectory file path robustly.

    Supported examples:
      - ../Data/trajectory_{route_id}.csv
      - ../Data/trajectory_1.csv  -> ../Data/trajectory_3.csv
      - ../Data/trajectory.csv    -> ../Data/trajectory_3.csv
    """
    s = str(template)

    # 1) python-format placeholder
    if "{route_id}" in s:
        return s.format(route_id=route_id)

    p = Path(s)
    stem = p.stem
    suffix = p.suffix or ".csv"

    # 2) replace trailing _<number>
    m = re.search(r"_(\d+)$", stem)
    if m is not None:
        new_stem = stem[: m.start()] + f"_{route_id}"
    else:
        # 3) append _route_id
        new_stem = f"{stem}_{route_id}"

    return str(p.with_name(new_stem + suffix))


def _load_flow_series(
    file_path: str,
    site_num: int,
    *,
    encoding: str = "auto",
    required_road_ids: list[int] | None = None,
):
    """
    Flow CSV required columns: road_index, hour, minute, flow, and (date or day).

    Returns:
      flow_series: (T,N) float32
      road_ids: list[int] columns kept (length N)
      dt_index: DatetimeIndex length T
      interval_minutes: inferred interval (int)
    """
    df = _read_csv_auto(file_path, encoding=encoding)
    df.columns = [str(c).strip() for c in df.columns]

    for c in ["road_index", "hour", "minute", "flow"]:
        if c not in df.columns:
            raise ValueError(f"Flow CSV missing column: {c}. Found: {list(df.columns)}")

    df = df.copy()
    df["road_index"] = pd.to_numeric(df["road_index"], errors="coerce").fillna(0).astype(int)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    df["minute"] = pd.to_numeric(df["minute"], errors="coerce").fillna(0).astype(int)
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(0.0).astype(np.float32)

    # Build date
    if "date" in df.columns:
        dt_date = pd.to_datetime(df["date"], errors="coerce")
        if dt_date.notna().sum() == 0:
            raise ValueError("Flow CSV column 'date' exists but cannot be parsed to datetime.")
        df["_date"] = dt_date.dt.normalize()
    else:
        if "day" not in df.columns:
            raise ValueError("Flow CSV missing both 'date' and 'day'. Need at least one to build timeline.")
        base = pd.Timestamp("1970-01-01")
        df["_date"] = base + pd.to_timedelta(pd.to_numeric(df["day"], errors="coerce").fillna(0).astype(int), unit="D")

    df["_dt"] = df["_date"] + pd.to_timedelta(df["hour"], unit="h") + pd.to_timedelta(df["minute"], unit="m")
    df = df.sort_values(["_dt", "road_index"]).reset_index(drop=True)

    # Infer interval safely (seconds -> minutes)
    dt_unique = pd.Index(df["_dt"].unique()).sort_values()
    interval_minutes = 15
    if len(dt_unique) >= 2:
        dt_vals = dt_unique.to_numpy(dtype="datetime64[ns]")
        diffs_sec = np.diff(dt_vals).astype("timedelta64[s]").astype(np.int64)
        diffs_min = diffs_sec / 60.0
        diffs_min = diffs_min[diffs_min > 0]
        if diffs_min.size > 0:
            interval_minutes = int(np.median(diffs_min))
            if interval_minutes <= 0:
                interval_minutes = 15

    pv = df.pivot_table(index="_dt", columns="road_index", values="flow", aggfunc="mean")
    pv = pv.ffill().bfill().fillna(0.0)

    road_ids_all = sorted([int(x) for x in pv.columns.tolist()])

    if site_num <= 0:
        raise ValueError(f"config.site_num must be > 0, got {site_num}")

    if len(road_ids_all) < site_num:
        raise ValueError(f"Flow CSV has only {len(road_ids_all)} roads, but config.site_num={site_num}")

    required_set = set(int(x) for x in (required_road_ids or []))
    if required_set:
        missing_req = required_set.difference(set(road_ids_all))
        if missing_req:
            raise ValueError(
                "Some required route segment road_index are missing from the flow CSV: "
                f"{sorted(list(missing_req))}. Please check your flow data."
            )
        if len(required_set) > site_num:
            raise ValueError(
                f"Route requires {len(required_set)} segments, but site_num={site_num} is smaller."
            )

    # Select road_ids while guaranteeing required ids are included
    if site_num >= len(road_ids_all):
        road_ids = road_ids_all
    else:
        others = [rid for rid in road_ids_all if rid not in required_set][: (site_num - len(required_set))]
        road_ids = sorted(list(required_set) + others)

    pv = pv[road_ids]
    flow_series = pv.values.astype(np.float32)
    dt_index = pd.DatetimeIndex(pv.index)
    return flow_series, road_ids, dt_index, interval_minutes


# ----------------------------- time utils -----------------------------
def _traj_time_sort_indices(traj_df: pd.DataFrame) -> np.ndarray:
    if "start_time" not in traj_df.columns:
        raise ValueError("Trajectory CSV missing start_time")

    s = traj_df["start_time"]
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() > 0:
        dt_filled = dt.fillna(pd.Timestamp("1970-01-01 00:00:00"))
        return np.argsort(dt_filled.values.astype("datetime64[ns]")).astype(np.int64)

    idx = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    return np.argsort(idx).astype(np.int64)


def _time_split_indices(traj_df: pd.DataFrame, cfg):
    n = len(traj_df)
    order = _traj_time_sort_indices(traj_df)

    tr = float(cfg.training_set_rate)
    vr = float(cfg.validate_set_rate)
    ter = float(getattr(cfg, 'test_set_rate', max(0.0, 1.0 - tr - vr)))
    total = max(tr + vr + ter, 1e-8)
    tr, vr, ter = tr / total, vr / total, ter / total

    n_train = max(int(round(n * tr)), 1)
    n_val = max(int(round(n * vr)), 1)
    n_test = max(n - n_train - n_val, 1)

    while n_train + n_val + n_test > n:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        else:
            n_test -= 1
    while n_train + n_val + n_test < n:
        n_test += 1

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:n_train + n_val + n_test]

    rng = np.random.RandomState(int(cfg.seed))
    rng.shuffle(train_idx)

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def _departure_time_fields(traj_df: pd.DataFrame):
    """
    week(0-6), day(1-31), hour(0-23), minute(0-59), second(0-59)
    If start_time unparsable -> zeros.
    """
    st = pd.to_datetime(traj_df["start_time"], errors="coerce")
    n = len(traj_df)
    if st.notna().sum() == 0:
        return (
            np.zeros(n, dtype=np.int64),
            np.ones(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
        )

    week = st.dt.dayofweek.fillna(0).astype(int).values
    day = st.dt.day.fillna(1).astype(int).values
    hour = st.dt.hour.fillna(0).astype(int).values
    minute = st.dt.minute.fillna(0).astype(int).values
    second = st.dt.second.fillna(0).astype(int).values

    week = np.clip(week, 0, 6)
    day = np.clip(day, 1, 31)
    hour = np.clip(hour, 0, 23)
    minute = np.clip(minute, 0, 59)
    second = np.clip(second, 0, 59)

    return week.astype(np.int64), day.astype(np.int64), hour.astype(np.int64), minute.astype(np.int64), second.astype(np.int64)


def _compute_start_indices(traj_df: pd.DataFrame, flow_dt_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Map each trajectory start_time to the latest flow time bin <= start_time.

    This implements the common "use the most recent observed flow before departure"
    alignment logic (e.g., 12:09 -> 12:00 for 15-min bins).
    """
    st = pd.to_datetime(traj_df["start_time"], errors="coerce")
    if st.notna().sum() == 0:
        idx = pd.to_numeric(traj_df["start_time"], errors="coerce").fillna(0).astype(int).values
        idx = np.clip(idx, 0, len(flow_dt_index) - 1)
        return idx.astype(np.int64)

    # fill NaT with earliest flow time
    st_filled = st.fillna(flow_dt_index.min())
    # searchsorted gives insertion point; we need the last index <= time => right-1
    idx = flow_dt_index.searchsorted(st_filled.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    idx = np.clip(idx, 0, len(flow_dt_index) - 1)
    return idx.astype(np.int64)


def _convert_time_to_minutes(arr: np.ndarray, unit: str) -> np.ndarray:
    unit = str(unit).lower().strip()
    if unit == "minute":
        return arr
    if unit == "hour":
        return arr * 60.0
    if unit == "second":
        return arr / 60.0
    # fallback
    return arr


def _infer_time_unit_auto(
    y_seg_raw: np.ndarray,
    *,
    dist_m: np.ndarray | None,
    speed_limit_kmh: float = 120.0,
):
    """
    Infer travel_time unit from (y_seg_raw, dist_m) using speed plausibility.

    Returns:
      unit: one of {"minute","hour","second"}
      report: dict with diagnostic stats
    """
    report = {}
    y = y_seg_raw.astype(np.float64)

    # If no distance info, fall back to conservative heuristics (prefer "minute")
    if dist_m is None:
        flat = y[np.isfinite(y) & (y > 0)]
        if flat.size == 0:
            report["reason"] = "no_positive_values"
            return "minute", report
        q99 = float(np.quantile(flat, 0.99))
        q50 = float(np.quantile(flat, 0.50))
        report.update({"q50": q50, "q99": q99, "reason": "no_distance_fallback"})
        # very large -> seconds
        if q99 > 300.0:
            return "second", report
        # only infer hour when values are not extremely small
        if q99 < 0.3 and q50 > 0.02:
            return "hour", report
        return "minute", report

    dist = dist_m.astype(np.float64)

    # valid mask
    mask_base = np.isfinite(y) & np.isfinite(dist) & (y > 0) & (dist > 0)
    if mask_base.sum() < 50:
        # not enough valid pairs -> fallback
        report["reason"] = "insufficient_valid_pairs"
        return "minute", report

    # plausible speed range (km/h)
    max_speed = max(160.0, float(speed_limit_kmh) * 2.0)
    min_speed = 1.0
    target_speed = float(speed_limit_kmh) * 0.75
    target_speed = float(np.clip(target_speed, 30.0, 120.0))

    best_unit = "minute"
    best_score = float("inf")

    for unit in ["minute", "hour", "second"]:
        t_min = _convert_time_to_minutes(y, unit)
        mask = mask_base & np.isfinite(t_min) & (t_min > 0)

        if mask.sum() < 50:
            continue

        speed_kmh = (dist[mask] / 1000.0) / (t_min[mask] / 60.0)  # km/h
        speed_kmh = speed_kmh[np.isfinite(speed_kmh) & (speed_kmh > 0)]
        if speed_kmh.size < 50:
            continue

        med = float(np.median(speed_kmh))
        q05 = float(np.quantile(speed_kmh, 0.05))
        q95 = float(np.quantile(speed_kmh, 0.95))
        frac_out = float(np.mean((speed_kmh < min_speed) | (speed_kmh > max_speed)))

        # score: out-of-range dominates, then closeness to target
        center = abs(np.log(med + 1e-6) - np.log(target_speed + 1e-6))
        score = frac_out * 100.0 + center * 10.0

        report[f"{unit}_median_speed"] = med
        report[f"{unit}_q05_speed"] = q05
        report[f"{unit}_q95_speed"] = q95
        report[f"{unit}_frac_outside"] = frac_out
        report[f"{unit}_score"] = score

        if score < best_score:
            best_score = score
            best_unit = unit

    report["chosen_unit"] = best_unit
    report["chosen_score"] = best_score
    report["reason"] = "distance_speed_plausibility"
    return best_unit, report


# ----------------------------- distance binning -----------------------------
@dataclass
class DistanceBinner:
    edges: np.ndarray  # shape (bins-1,)

    def bin(self, x: np.ndarray) -> np.ndarray:
        # x: (...,) float
        # returns: (...,) int in [0,bins-1]
        return np.digitize(x, self.edges, right=False).astype(np.int64)


def _build_distance_binner(dist_values: np.ndarray, bins: int = 50) -> DistanceBinner:
    """
    Build quantile edges on train distances (meters). Robust to long tails.
    """
    x = dist_values.reshape(-1).astype(np.float32)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < 10:
        edges = np.linspace(0.0, 1.0, bins - 1).astype(np.float32)
        return DistanceBinner(edges=edges)

    # use log1p to stabilize
    xl = np.log1p(x)
    qs = np.linspace(0, 1, bins + 1)[1:-1]  # exclude 0,1
    edges = np.quantile(xl, qs).astype(np.float32)
    edges = np.unique(edges)
    if edges.size < (bins - 1):
        mn, mx = float(xl.min()), float(xl.max())
        edges = np.linspace(mn, mx, bins - 1).astype(np.float32)
    return DistanceBinner(edges=edges)




def _build_flow_time_features(flow_dt_index: pd.DatetimeIndex, interval_minutes: int, t0: int, P: int, Q: int):
    """
    Build step-wise hour/minute-bin features for the whole [history + future] window.
    For indices outside the observed flow range, we extrapolate timestamps using the
    inferred flow interval so temporal embeddings remain meaningful near boundaries.
    """
    total_steps = int(P + Q)
    offsets = np.arange(-P, Q, dtype=np.int64)
    base_delta = pd.to_timedelta(int(interval_minutes), unit='m')
    out_hour = np.zeros((total_steps,), dtype=np.int64)
    out_minute = np.zeros((total_steps,), dtype=np.int64)
    minute_bins = int(max(1, round(60.0 / max(float(interval_minutes), 1e-6))))
    minute_bins = int(max(1, min(minute_bins, 60)))

    for i, off in enumerate(offsets):
        idx = int(t0 + off)
        if 0 <= idx < len(flow_dt_index):
            dt = flow_dt_index[idx]
        elif idx < 0:
            dt = flow_dt_index[0] + idx * base_delta
        else:
            dt = flow_dt_index[-1] + (idx - (len(flow_dt_index) - 1)) * base_delta
        out_hour[i] = int(dt.hour)
        out_minute[i] = int((dt.minute // max(1, interval_minutes)) % minute_bins)
    return out_hour, out_minute

# ----------------------------- Dataset -----------------------------
class KGSTGATDataset(Dataset):
    def __init__(
        self,
        flow_series: np.ndarray,
        flow_dt_index: pd.DatetimeIndex,
        interval_minutes: int,
        traj_df: pd.DataFrame,
        segments_cols: list[int],
        config,
        split_indices: np.ndarray,
        *,
        flow_mean=None, flow_std=None,
        tt_total_mean=None, tt_total_std=None,
        tt_seg_mean=None, tt_seg_std=None,
        vehicle_id_vocab=None,
        vehicle_type_vocab=None,
        distance_binner: DistanceBinner | None = None,
    ):
        self.flow_series = flow_series
        self.flow_dt_index = flow_dt_index
        self.interval_minutes = int(interval_minutes)
        self.T, self.N = flow_series.shape

        self.traj_df = traj_df.reset_index(drop=True)
        self.cfg = config
        self.P = int(config.input_length)
        self.Q = int(config.output_length)
        self.L = int(config.trajectory_length)

        self.indices = np.array(split_indices, dtype=np.int64)

        self.segments_cols = np.array(segments_cols[:self.L], dtype=np.int64)

        # ---------- distances (meters) ----------
        dist_cols = [f"distance_{i}" for i in range(1, self.L + 1)]
        if all(c in self.traj_df.columns for c in dist_cols):
            dist = self.traj_df[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
            self.dist_m = dist
        else:
            self.dist_m = np.zeros((len(self.traj_df), self.L), dtype=np.float32)

        self.distance_binner = distance_binner
        if self.distance_binner is None:
            self.distance_binner = DistanceBinner(edges=np.array([np.inf], dtype=np.float32))
        dist_bins = self.distance_binner.bin(np.log1p(self.dist_m))
        self.dist_bin = dist_bins.astype(np.int64)  # (num_traj,L)

        # ---------- labels ----------
        seg_cols = [f"travel_time_{i}" for i in range(1, self.L + 1)]
        for c in seg_cols:
            if c not in self.traj_df.columns:
                raise ValueError(f"Trajectory CSV missing column: {c}")

        y_seg_raw = self.traj_df[seg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

        # use effective unit resolved in create_dataloaders
        unit_eff = getattr(config, "trajectory_time_unit_effective", getattr(config, "trajectory_time_unit", "auto"))
        if str(unit_eff).lower() == "auto":
            # fallback (should not happen if create_dataloaders ran)
            unit_eff = "minute"
        self.y_seg_time_min = _convert_time_to_minutes(y_seg_raw, unit_eff).astype(np.float32)  # (num_traj,L)
        self.y_total_time_min = self.y_seg_time_min.sum(axis=1)                                 # (num_traj,)

        # ---------- start index (for flow alignment only) ----------
        self.start_idx = _compute_start_indices(self.traj_df, self.flow_dt_index)

        # ---------- categorical common features ----------
        self.vehicle_id_vocab = vehicle_id_vocab or {"<UNK>": 0}
        self.vehicle_type_vocab = vehicle_type_vocab or {"<UNK>": 0}

        if getattr(config, "use_vehicle_id", False) and "vehicle_id" in self.traj_df.columns:
            vids = self.traj_df["vehicle_id"].astype(str).values
            self.vid_idx = np.array([self.vehicle_id_vocab.get(v, 0) for v in vids], dtype=np.int64)
        else:
            self.vid_idx = np.zeros(len(self.traj_df), dtype=np.int64)

        if "vehicle_type" in self.traj_df.columns:
            vts = self.traj_df["vehicle_type"].astype(str).values
            self.vtype_idx = np.array([self.vehicle_type_vocab.get(v, 0) for v in vts], dtype=np.int64)
        else:
            self.vtype_idx = np.zeros(len(self.traj_df), dtype=np.int64)

        week, day, hour, minute, second = _departure_time_fields(self.traj_df)
        self.dep_week = week
        self.dep_day = day
        self.dep_hour = hour
        self.dep_minute = minute
        self.dep_second = second

        # ---------- scalers ----------
        self.flow_mean = flow_mean
        self.flow_std = flow_std
        self.tt_total_mean = tt_total_mean
        self.tt_total_std = tt_total_std
        self.tt_seg_mean = tt_seg_mean
        self.tt_seg_std = tt_seg_std

        # ---------- offsets for shared feature index space ----------
        self.feature_offsets = getattr(config, "feature_offsets", None)
        if not isinstance(self.feature_offsets, dict):
            raise ValueError("config.feature_offsets must be a dict (set in create_dataloaders).")

    @staticmethod
    def _norm(x, mean, std):
        if mean is None or std is None:
            return x
        return (x - mean) / (std + 1e-6)

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, i):
        idx = int(self.indices[i])
        t0 = int(self.start_idx[idx])

        # ---------------- flow windows (strict + padding) ----------------
        # history: [t0-P, t0)
        hs = max(t0 - self.P, 0)
        he = t0
        x_flow = self.flow_series[hs:he]
        if x_flow.shape[0] < self.P:
            pad_len = self.P - x_flow.shape[0]
            pad = np.zeros((pad_len, self.N), dtype=np.float32)
            x_flow = np.concatenate([pad, x_flow], axis=0)
        elif x_flow.shape[0] > self.P:
            x_flow = x_flow[-self.P:]

        # future: [t0, t0+Q)
        fs = t0
        fe = min(t0 + self.Q, self.T)
        y_flow = self.flow_series[fs:fe]
        valid_q = y_flow.shape[0]
        y_flow_mask = np.zeros((self.Q, self.N), dtype=np.float32)
        if valid_q > 0:
            y_flow_mask[:valid_q, :] = 1.0

        if y_flow.shape[0] < self.Q:
            pad_len = self.Q - y_flow.shape[0]
            pad = np.zeros((pad_len, self.N), dtype=np.float32)
            y_flow = np.concatenate([y_flow, pad], axis=0)
        elif y_flow.shape[0] > self.Q:
            y_flow = y_flow[:self.Q]
            y_flow_mask[:, :] = 1.0

        if self.cfg.normalize:
            x_flow = self._norm(x_flow, self.flow_mean, self.flow_std)
            y_flow = self._norm(y_flow, self.flow_mean, self.flow_std)

        flow_hour_seq, flow_minute_seq = _build_flow_time_features(
            self.flow_dt_index,
            self.interval_minutes,
            t0=t0,
            P=self.P,
            Q=self.Q,
        )

        # labels in minutes (for reporting)
        y_seg_min = self.y_seg_time_min[idx].copy()
        y_total_min = float(self.y_total_time_min[idx])

        # normalized labels for training
        y_seg = y_seg_min.copy()
        y_total = y_total_min
        if self.cfg.normalize:
            y_seg = self._norm(y_seg, self.tt_seg_mean, self.tt_seg_std)
            y_total = self._norm(np.array([y_total], dtype=np.float32), self.tt_total_mean, self.tt_total_std)[0]

        dist_m = self.dist_m[idx].copy().astype(np.float32)
        dist_bin = self.dist_bin[idx].copy().astype(np.int64)

        off = self.feature_offsets
        feat = np.zeros(7 + 2 * self.L, dtype=np.int64)
        feat[0] = off['vehicle_id'] + int(self.vid_idx[idx])
        feat[1] = off['vehicle_type'] + int(self.vtype_idx[idx])
        feat[2] = off['week'] + int(self.dep_week[idx])
        feat[3] = off['day'] + (int(self.dep_day[idx]) - 1)
        feat[4] = off['hour'] + int(self.dep_hour[idx])
        feat[5] = off['minute'] + int(self.dep_minute[idx])
        feat[6] = off['second'] + int(self.dep_second[idx])

        for j in range(self.L):
            feat[7 + j] = off['distance_bin'] + int(dist_bin[j])
        for j in range(self.L):
            feat[7 + self.L + j] = off['segment_id'] + int(self.segments_cols[j])

        batch = {
            'flow': torch.tensor(x_flow, dtype=torch.float32),
            'tra_features': torch.tensor(feat, dtype=torch.long),
            'segments': torch.tensor(self.segments_cols, dtype=torch.long),
            'distances': torch.tensor(dist_m, dtype=torch.float32),
            'y_flow': torch.tensor(y_flow, dtype=torch.float32),
            'y_flow_mask': torch.tensor(y_flow_mask, dtype=torch.float32),
            'flow_hour_seq': torch.tensor(flow_hour_seq, dtype=torch.long),
            'flow_minute_seq': torch.tensor(flow_minute_seq, dtype=torch.long),
            'flow_hour': torch.tensor(int(flow_hour_seq[self.P]), dtype=torch.long),
            'flow_minute': torch.tensor(int(flow_minute_seq[self.P]), dtype=torch.long),
            'y_seg_time': torch.tensor(y_seg, dtype=torch.float32),
            'y_total_time': torch.tensor([float(y_total)], dtype=torch.float32),
            'y_seg_time_min': torch.tensor(y_seg_min, dtype=torch.float32),
            'y_total_time_min': torch.tensor([float(y_total_min)], dtype=torch.float32),
        }
        return batch


# ----------------------------- Factory -----------------------------
def create_dataloaders(config, route_id: int):
    # fixed route segments (road_index ids in the flow CSV)
    if route_id == 1:
        route_seg_ids = [32, 33, 35, 36, 37]
    elif route_id == 2:
        route_seg_ids = [38, 39, 40, 43, 37]
    elif route_id == 3:
        route_seg_ids = [71, 70, 69]
    else:
        route_seg_ids = [32, 34, 47, 50, 51]

    config.route_segment_ids = list(route_seg_ids)

    # Load flow (guarantee route segments kept)
    flow_file = getattr(config, "file_flow", None) or getattr(config, "file_train_s")
    flow_series, road_ids, flow_dt_index, interval_minutes = _load_flow_series(
        flow_file,
        int(config.site_num),
        encoding=getattr(config, "flow_csv_encoding", "auto"),
        required_road_ids=route_seg_ids,
    )

    # write back interval for BPR conversion
    config.interval_minutes = int(interval_minutes)
    config.flow_unit_minutes = int(interval_minutes)
    config.flow_road_ids = list(road_ids)

    # Resolve trajectory file
    traj_file = _resolve_traj_file(getattr(config, "file_train_t"), route_id)
    traj_df = _read_csv_auto(traj_file, encoding=getattr(config, "traj_csv_encoding", "auto"))

    # map route segment ids -> flow column index
    rid_to_col = {int(rid): i for i, rid in enumerate(road_ids)}
    L = int(config.trajectory_length)
    missing = [int(sid) for sid in route_seg_ids[:L] if int(sid) not in rid_to_col]
    if missing:
        raise ValueError(
            "Route segment road_index not found in selected flow columns. "
            f"Missing: {missing}.\n"
            "Possible reasons:\n"
            "  - config.site_num is too small and truncated away these road_index\n"
            "  - flow CSV does not contain these road_index\n"
            "Fix:\n"
            "  - increase --site_num, or check flow CSV road_index values."
        )

    segments_cols = [int(rid_to_col[int(sid)]) for sid in route_seg_ids[:L]]
    config.route_segment_cols = list(segments_cols)

    print(f"\nLoading data for Route {route_id} ...")
    print(f"Route segments (road_index ids): {route_seg_ids}")
    print(f"Route segments (flow column idx): {segments_cols}")
    print(f"Flow time range: {flow_dt_index.min()} -> {flow_dt_index.max()} (interval≈{interval_minutes}min)")
    print(f"Flow data shape: {flow_series.shape}  (T,N)")
    print(f"Trajectory file: {traj_file}")
    print(f"Trajectory columns: {list(traj_df.columns)}")
    print(f"Trajectory data shape: {traj_df.shape}\n")

    # Warn if trajectory start_time is outside flow range (flow features will be clipped/padded)
    st = pd.to_datetime(traj_df.get("start_time", pd.Series([], dtype=str)), errors="coerce")
    if st.notna().sum() > 0:
        low = int((st < flow_dt_index.min()).sum())
        high = int((st > flow_dt_index.max()).sum())
        if (low + high) > 0:
            print(f"⚠ Warning: {low + high} trajectories have start_time outside flow timeline "
                  f"(earlier={low}, later={high}). Their flow windows will be clipped/padded.\n")

    # time-based split
    train_idx, val_idx, test_idx = _time_split_indices(traj_df, config)

    print("Data split (time-based):")
    print(f"  Train: {len(train_idx)} trajectories (earliest, then shuffled)")
    print(f"  Val:   {len(val_idx)} trajectories (middle)")
    print(f"  Test:  {len(test_idx)} trajectories (latest)")

    # ----------------- build vocabs on TRAIN only -----------------
    vehicle_id_vocab = {"<UNK>": 0}
    if getattr(config, "use_vehicle_id", False) and "vehicle_id" in traj_df.columns:
        for v in traj_df.loc[train_idx, "vehicle_id"].astype(str).tolist():
            if v not in vehicle_id_vocab:
                vehicle_id_vocab[v] = len(vehicle_id_vocab)

    vehicle_type_vocab = {"<UNK>": 0}
    if "vehicle_type" in traj_df.columns:
        for v in traj_df.loc[train_idx, "vehicle_type"].astype(str).tolist():
            if v not in vehicle_type_vocab:
                vehicle_type_vocab[v] = len(vehicle_type_vocab)

    # ----------------- distance bins on TRAIN only -----------------
    dist_cols = [f"distance_{i}" for i in range(1, L + 1)]
    if all(c in traj_df.columns for c in dist_cols):
        dist_train = traj_df.loc[train_idx, dist_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    else:
        dist_train = np.zeros((len(train_idx), L), dtype=np.float32)

    distance_bins = int(getattr(config, "distance_bins", 50))
    distance_binner = _build_distance_binner(dist_train, bins=distance_bins)

    # ----------------- infer trajectory time unit (TRAIN only) -----------------
    if str(getattr(config, "trajectory_time_unit", "auto")).lower() == "auto":
        seg_cols = [f"travel_time_{i}" for i in range(1, L + 1)]
        if not all(c in traj_df.columns for c in seg_cols):
            raise ValueError("Trajectory CSV missing travel_time_* columns required for training.")
        y_seg_train_raw = traj_df.loc[train_idx, seg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

        # Use distance if available
        dist_for_infer = dist_train if dist_train is not None else None
        unit_eff, report = _infer_time_unit_auto(
            y_seg_train_raw,
            dist_m=dist_for_infer,
            speed_limit_kmh=float(getattr(config, "speed_limit_kmh", 120.0)),
        )
        config.trajectory_time_unit_effective = unit_eff
    else:
        config.trajectory_time_unit_effective = str(getattr(config, "trajectory_time_unit")).lower()

    # ----------------- build shared index space offsets -----------------
    offsets = {}
    cursor = 0

    offsets["vehicle_id"] = cursor
    cursor += max(len(vehicle_id_vocab), 1)

    offsets["vehicle_type"] = cursor
    cursor += max(len(vehicle_type_vocab), 1)

    offsets["week"] = cursor
    cursor += 7

    offsets["day"] = cursor
    cursor += 31  # 0..30

    offsets["hour"] = cursor
    cursor += 24

    offsets["minute"] = cursor
    cursor += 60

    offsets["second"] = cursor
    cursor += 60

    offsets["distance_bin"] = cursor
    cursor += distance_bins

    offsets["segment_id"] = cursor
    cursor += int(config.site_num)  # 0..N-1 (flow column ids)

    config.feature_offsets = offsets
    config.feature_tra = cursor          # shared embedding table size
    config.field_cnt = 7 + 2 * L         # number of fields
    config.vehicle_vocab_size = len(vehicle_id_vocab)
    config.vehicle_type_vocab_size = len(vehicle_type_vocab)
    config.distance_bins = distance_bins

    # ----------------- compute scalers on TRAIN only -----------------
    tmp_ds = KGSTGATDataset(
        flow_series, flow_dt_index, interval_minutes,
        traj_df, segments_cols, config, train_idx,
        vehicle_id_vocab=vehicle_id_vocab,
        vehicle_type_vocab=vehicle_type_vocab,
        distance_binner=distance_binner,
    )

    T = flow_series.shape[0]
    used = np.zeros(T, dtype=bool)
    t0s = tmp_ds.start_idx[train_idx]
    P, Q = int(config.input_length), int(config.output_length)

    for off in range(-P, Q):
        tt = t0s + off
        tt = tt[(tt >= 0) & (tt < T)]
        used[tt] = True
    if used.sum() == 0:
        used[:] = True

    flow_train = flow_series[used]
    flow_mean = flow_train.mean(axis=0, keepdims=True).astype(np.float32)
    flow_std = flow_train.std(axis=0, keepdims=True).astype(np.float32)
    flow_std[flow_std < 1e-6] = 1.0

    y_seg_train_min = tmp_ds.y_seg_time_min[train_idx]                    # (n_train,L)
    y_total_train_min = tmp_ds.y_total_time_min[train_idx].reshape(-1, 1) # (n_train,1)

    tt_total_mean = y_total_train_min.mean(axis=0, keepdims=True).astype(np.float32)
    tt_total_std = y_total_train_min.std(axis=0, keepdims=True).astype(np.float32)
    tt_total_std[tt_total_std < 1e-6] = 1.0

    tt_seg_mean = y_seg_train_min.mean(axis=0).astype(np.float32)         # (L,)
    tt_seg_std = y_seg_train_min.std(axis=0).astype(np.float32)
    tt_seg_std[tt_seg_std < 1e-6] = 1.0

    config.flow_mean = flow_mean
    config.flow_std = flow_std
    config.tt_total_mean = tt_total_mean
    config.tt_total_std = tt_total_std
    config.tt_seg_mean = tt_seg_mean
    config.tt_seg_std = tt_seg_std

    # ----------------- datasets & loaders -----------------
    train_ds = KGSTGATDataset(
        flow_series, flow_dt_index, interval_minutes,
        traj_df, segments_cols, config, train_idx,
        flow_mean=flow_mean, flow_std=flow_std,
        tt_total_mean=tt_total_mean, tt_total_std=tt_total_std,
        tt_seg_mean=tt_seg_mean, tt_seg_std=tt_seg_std,
        vehicle_id_vocab=vehicle_id_vocab,
        vehicle_type_vocab=vehicle_type_vocab,
        distance_binner=distance_binner,
    )
    val_ds = KGSTGATDataset(
        flow_series, flow_dt_index, interval_minutes,
        traj_df, segments_cols, config, val_idx,
        flow_mean=flow_mean, flow_std=flow_std,
        tt_total_mean=tt_total_mean, tt_total_std=tt_total_std,
        tt_seg_mean=tt_seg_mean, tt_seg_std=tt_seg_std,
        vehicle_id_vocab=vehicle_id_vocab,
        vehicle_type_vocab=vehicle_type_vocab,
        distance_binner=distance_binner,
    )
    test_ds = KGSTGATDataset(
        flow_series, flow_dt_index, interval_minutes,
        traj_df, segments_cols, config, test_idx,
        flow_mean=flow_mean, flow_std=flow_std,
        tt_total_mean=tt_total_mean, tt_total_std=tt_total_std,
        tt_seg_mean=tt_seg_mean, tt_seg_std=tt_seg_std,
        vehicle_id_vocab=vehicle_id_vocab,
        vehicle_type_vocab=vehicle_type_vocab,
        distance_binner=distance_binner,
    )

    train_loader = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(config.batch_size), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(config.batch_size), shuffle=False, drop_last=False)

    print(f"TRAIN dataset size: {len(train_ds)}")
    print(f"VAL dataset size: {len(val_ds)}")
    print(f"TEST dataset size: {len(test_ds)}")

    return train_loader, val_loader, test_loader