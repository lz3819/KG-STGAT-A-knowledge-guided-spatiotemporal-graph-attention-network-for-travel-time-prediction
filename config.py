# -*- coding: utf-8 -*-
"""
Configuration for KG-STGAT (REAL pipeline)

Notes:
- Adds: trajectory_time_unit (with safer auto-infer implemented in dataloader),
        flow_csv_encoding, traj_csv_encoding
- Adds: flow_mape_min_target (avoid divide-by-zero MAPE for flow)
- Adds: bpr_trainable (whether alpha/beta are trainable in fusion)
"""
import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description='KG-STGAT Configuration')

    # ---------------- Data paths ----------------
    parser.add_argument('--file_train_s', type=str,
                        default='../Data/flow_15_aligned_true.csv',
                        help='(legacy) flow data file path')
    parser.add_argument('--file_flow', type=str,
                        default='../Data/flow_15_aligned_true.csv',
                        help='Flow data file path')
    parser.add_argument('--file_train_t', type=str,
                        default='../Data/trajectory_1.csv',
                        help=('Trajectory data file path. '
                              'Supports patterns like trajectory_{route_id}.csv or trajectory_1.csv.'))
    parser.add_argument('--file_adj', type=str,
                        default='../Data/adjacent.csv',
                        help='Adjacency matrix file path (not used in REAL pipeline)')

    # ---------------- CSV encoding ----------------
    parser.add_argument('--flow_csv_encoding', type=str, default='auto',
                        choices=['auto', 'utf-8', 'utf-8-sig', 'gbk', 'gb18030'],
                        help='Encoding for flow csv (auto will try several encodings)')
    parser.add_argument('--traj_csv_encoding', type=str, default='auto',
                        choices=['auto', 'utf-8', 'utf-8-sig', 'gbk', 'gb18030'],
                        help='Encoding for trajectory csv (auto will try several encodings)')

    # ---------------- Route selection ----------------
    parser.add_argument('--route_id', type=int, default=1,
                        choices=[1, 2, 3, 4],
                        help='Route ID')

    # ---------------- Model parameters ----------------
    parser.add_argument('--site_num', type=int, default=108, help='Number of road segments (flow nodes)')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # ---------------- Sequence lengths ----------------
    parser.add_argument('--input_length', type=int, default=12, help='Input sequence length (P)')
    parser.add_argument('--output_length', type=int, default=6, help='Output sequence length (Q)')

    # ---------------- Training parameters ----------------
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='Epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # ---------------- Data split ----------------
    parser.add_argument('--training_set_rate', type=float, default=0.7, help='Training ratio')
    parser.add_argument('--validate_set_rate', type=float, default=0.15, help='Validation ratio')
    parser.add_argument('--test_set_rate', type=float, default=0.15, help='Test ratio')

    # ---------------- Loss weights ----------------
    parser.add_argument('--lambda_flow', type=float, default=0.3, help='Weight for flow loss')
    parser.add_argument('--lambda_total', type=float, default=0.4, help='Weight for total travel time loss')
    parser.add_argument('--lambda_segment', type=float, default=0.3, help='Weight for segment travel time loss')

    # ---------------- Loss type ----------------
    parser.add_argument('--loss_type', type=str, default='mae',
                        choices=['mae', 'huber', 'mse'],
                        help='Training loss type (paper uses MAE)')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber delta for SmoothL1Loss')

    # ---------------- Vehicle ID ----------------
    parser.add_argument('--use_vehicle_id', dest='use_vehicle_id', action='store_true', help='Use vehicle_id feature')
    parser.add_argument('--no_vehicle_id', dest='use_vehicle_id', action='store_false', help='Disable vehicle_id feature')
    parser.set_defaults(use_vehicle_id=True)

    # ---------------- Normalization ----------------
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Enable normalization')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false', help='Disable normalization')
    parser.set_defaults(normalize=True)

    # ---------------- Trajectory travel time unit ----------------
    # IMPORTANT: auto-infer is implemented using distances (if available) to avoid common mis-infer.
    parser.add_argument('--trajectory_time_unit', type=str, default='auto',
                        choices=['auto', 'minute', 'hour', 'second'],
                        help='Unit of travel_time_* columns in trajectory csv. auto will infer (recommended).')
    parser.add_argument('--assume_time_in_minutes', action='store_true',
                        help='Legacy flag (kept for compatibility). If set, forces minute.')

    # ---------------- Physics / BPR ----------------
    parser.add_argument('--speed_limit_kmh', type=float, default=120.0, help='Speed limit in km/h')
    parser.add_argument('--lanes', type=int, default=2, help='Number of lanes per direction')
    parser.add_argument('--capacity_per_lane_vph', type=float, default=2000.0,
                        help='Capacity per lane (veh/hour), typical for expressways in China')
    parser.add_argument('--bpr_init_alpha', type=float, default=0.15, help='Initial BPR alpha')
    parser.add_argument('--bpr_init_beta', type=float, default=4.0, help='Initial BPR beta')
    # NOTE: flow_unit_minutes will be overwritten by dataloader based on flow timestamps if possible.
    parser.add_argument('--flow_unit_minutes', type=int, default=15,
                        help='Flow aggregation interval in minutes (e.g., 15 means veh/15min). '
                             'Will be overridden by inferred interval from flow data.')

    # ---------------- Impedance auxiliary input (derived from predicted flow + BPR) ----------------
    parser.add_argument('--use_impedance_aux', dest='use_impedance_aux', action='store_true',
                        help='Use BPR-derived impedance/time as auxiliary input to ITTE')
    parser.add_argument('--no_impedance_aux', dest='use_impedance_aux', action='store_false',
                        help='Disable impedance auxiliary input')
    parser.set_defaults(use_impedance_aux=True)
    parser.add_argument('--impedance_aux_log1p', dest='impedance_aux_log1p', action='store_true',
                        help='Apply log1p to raw minutes when normalize=False (stabilize scale)')
    parser.add_argument('--no_impedance_aux_log1p', dest='impedance_aux_log1p', action='store_false')
    parser.set_defaults(impedance_aux_log1p=True)

    # ---------------- Stochastic BPR (randomized parameters to model volatility) ----------------
    parser.add_argument('--bpr_stochastic', dest='bpr_stochastic', action='store_true',
                        help='Enable stochastic BPR parameters (mean-1 lognormal noise)')
    parser.add_argument('--bpr_deterministic', dest='bpr_stochastic', action='store_false',
                        help='Disable stochastic BPR (use fixed/trainable parameters only)')
    parser.set_defaults(bpr_stochastic=True)
    parser.add_argument('--bpr_mc_samples', type=int, default=1,
                        help='Monte Carlo samples for stochastic BPR (>=1)')
    parser.add_argument('--bpr_alpha_noise', type=float, default=0.05,
                        help='Lognormal sigma for alpha (relative volatility)')
    parser.add_argument('--bpr_beta_noise', type=float, default=0.02,
                        help='Lognormal sigma for beta (relative volatility)')
    parser.add_argument('--bpr_capacity_noise', type=float, default=0.05,
                        help='Lognormal sigma for capacity (relative volatility)')
    parser.add_argument('--bpr_speed_noise', type=float, default=0.01,
                        help='Lognormal sigma for speed limit (relative volatility)')

    # Whether alpha/beta are trainable in the fusion module
    parser.add_argument('--bpr_trainable', dest='bpr_trainable', action='store_true',
                        help='Make BPR alpha/beta trainable (recommended when fusion enabled)')
    parser.add_argument('--no_bpr_trainable', dest='bpr_trainable', action='store_false',
                        help='Freeze BPR alpha/beta (use fixed values)')
    parser.set_defaults(bpr_trainable=True)


    # Optional: learnable calibration for capacity/free-flow time (per node) in BPR
    parser.add_argument('--bpr_trainable_physics', dest='bpr_trainable_physics', action='store_true',
                        help='Make BPR capacity/free-flow-time calibration trainable (per road segment)')
    parser.add_argument('--no_bpr_trainable_physics', dest='bpr_trainable_physics', action='store_false',
                        help='Disable trainable physics calibration')
    parser.set_defaults(bpr_trainable_physics=True)
    # ---------------- Fusion ----------------
    parser.add_argument('--use_fusion', dest='use_fusion', action='store_true', help='Fuse individual & physics time')
    parser.add_argument('--no_fusion', dest='use_fusion', action='store_false', help='Disable fusion')
    parser.set_defaults(use_fusion=True)

    # ---------------- Metrics ----------------
    parser.add_argument('--flow_mape_min_target', type=float, default=1e-6,
                        help='Flow MAPE ignores targets with abs(y_true) < this threshold (avoid division by zero).')

    # ---------------- Paths ----------------
    parser.add_argument('--save_path', type=str, default='weights/KG-STGAT/', help='Save path base')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'])
    parser.add_argument('--checkpoint', type=str, default=None)

    # ---------------- Device ----------------
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--print_model', action='store_true')

    config = parser.parse_args()

    # Resolve legacy path
    if not getattr(config, "file_flow", None):
        config.file_flow = config.file_train_s

    # Legacy force minutes flag
    if config.assume_time_in_minutes:
        config.trajectory_time_unit = 'minute'

    # Route trajectory length
    route_lengths = {1: 5, 2: 5, 3: 3, 4: 5}
    config.trajectory_length = route_lengths[config.route_id]

    # Placeholders (will be filled by dataloader)
    config.feature_tra = 0
    config.field_cnt = 0
    config.feature_offsets = None
    config.k = config.emb_size

    config.model_name = f'KG-STGAT_Route{config.route_id}'
    config.is_training = (config.mode in ['train', 'both'])

    config.save_path = os.path.join(config.save_path, f'Route{config.route_id}')
    os.makedirs(config.save_path, exist_ok=True)
    return config


def print_config(config):
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    for arg in vars(config):
        print(f"{arg:30s}: {getattr(config, arg)}")
    print("=" * 70 + "\n")
