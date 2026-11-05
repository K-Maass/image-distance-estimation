from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from utils import load_dataset, load_test_dataset, save_results


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
CACHE_DIR = ARTIFACT_DIR / "experiment_driver_cache"
PHASE_OUTPUTS = {
    1: ARTIFACT_DIR / "feature_sweep_summary.csv",
    2: ARTIFACT_DIR / "feature_importance_summary.csv",
    3: ARTIFACT_DIR / "preprocessing_ablations.csv",
    4: ARTIFACT_DIR / "model_ablations.csv",
    5: ARTIFACT_DIR / "low_bin_diagnostics.csv",
    6: ARTIFACT_DIR / "experiment_driver_phase6_single_model_summary.csv",
}
BIN_RANGES = ((0.25, 1.00), (1.00, 2.00), (2.00, 2.78))
RANDOM_STATE = 42
TASK_WARP_Y_START = 0.08
TASK_WARP_Y_POWER_SOFT = 2.05
TASK_WARP_X_POWER_SOFT = 1.35
TASK_WARP_Y_POWER_HARD = 2.65
TASK_WARP_X_POWER_HARD = 1.75
FEATURE_USEFUL_THRESHOLD = -0.0015
FEATURE_HARMFUL_THRESHOLD = 0.0015
PCA_BLOCK_DIM = 96


@dataclass(frozen=True)
class PreprocConfig:
    load_rgb: bool
    downsample: int
    normalize: bool
    use_warp: bool
    view_family: str = "combined"

    def short_tag(self) -> str:
        return (
            f"{'rgb' if self.load_rgb else 'gray'}"
            f"_ds{self.downsample}"
            f"_norm{int(self.normalize)}"
            f"_warp{int(self.use_warp)}"
            f"_{self.view_family}"
        )


@dataclass(frozen=True)
class ModelSpec:
    family: str
    params: tuple[tuple[str, Any], ...]

    @property
    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def short_tag(self) -> str:
        if not self.params:
            return self.family
        parts = [f"{key}-{value}" for key, value in self.params]
        return f"{self.family}__{'__'.join(parts)}"


@dataclass
class CVResult:
    model_family: str
    model_tag: str
    preproc_tag: str
    groups: list[str]
    overall_mae: float
    per_bin: dict[str, float]
    oof_pred: np.ndarray
    fold_rows: list[dict[str, Any]]
    importance_rows: list[dict[str, Any]]


GROUP_SOURCE: dict[str, str] = {
    "gray_global_stats": "existing",
    "gray_region_stats": "existing",
    "gradient_stats": "existing",
    "projection_profiles": "existing",
    "orientation_histograms": "existing",
    "symmetry": "existing",
    "radial_rings": "existing",
    "threshold_mass": "existing",
    "spectrum_summaries": "existing",
    "pooled_gray_grids": "existing",
    "rgb_summary": "existing",
    "multiview_deltas": "existing",
    "pca_coarse_blocks": "existing",
    "warped_view_groups": "existing",
    "detrended_gradmag_groups": "existing",
    "percentile_spread_ladders": "new",
    "rowcol_entropy": "new",
    "center_border_contrast": "new",
    "quadrant_imbalance": "new",
    "horizon_band_transition": "new",
    "dark_run_continuity": "new",
    "corridor_width_proxy": "new",
    "dominant_orientation_concentration": "new",
    "freq_energy_ratio": "new",
    "patch_variance_hist": "new",
}
GROUP_FAMILY: dict[str, str] = {
    "gray_global_stats": "multiview",
    "gray_region_stats": "multiview",
    "gradient_stats": "shape",
    "projection_profiles": "multiview",
    "orientation_histograms": "shape",
    "symmetry": "shape",
    "radial_rings": "shape",
    "threshold_mass": "shape",
    "spectrum_summaries": "shape",
    "pooled_gray_grids": "multiview",
    "rgb_summary": "multiview",
    "multiview_deltas": "multiview",
    "pca_coarse_blocks": "multiview",
    "warped_view_groups": "multiview",
    "detrended_gradmag_groups": "shape",
    "percentile_spread_ladders": "multiview",
    "rowcol_entropy": "multiview",
    "center_border_contrast": "multiview",
    "quadrant_imbalance": "multiview",
    "horizon_band_transition": "multiview",
    "dark_run_continuity": "shape",
    "corridor_width_proxy": "shape",
    "dominant_orientation_concentration": "shape",
    "freq_energy_ratio": "shape",
    "patch_variance_hist": "multiview",
}
CANDIDATE_GROUPS = list(GROUP_SOURCE.keys())
ANCHOR_GROUP = "_reference_anchor"


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)


def stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_cv_bins(y: np.ndarray, n_bins: int = 8) -> np.ndarray:
    quantiles = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, quantiles).astype(int)


def get_cv_splits(y: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    bins = make_cv_bins(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(np.zeros(len(y)), bins))


def bin_mask(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (y >= lo) & (y < hi if hi < BIN_RANGES[-1][1] else y <= hi)


def per_bin_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for lo, hi in BIN_RANGES:
        label = f"[{lo:.2f},{hi:.2f})"
        mask = bin_mask(y_true, lo, hi)
        out[label] = float(mean_absolute_error(y_true[mask], y_pred[mask])) if np.any(mask) else float("nan")
    return out


def unit_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr.astype(np.float32)


def normalize_gray_image(gray: np.ndarray, apply_normalize: bool) -> np.ndarray:
    gray = unit_scale(gray)
    if not apply_normalize:
        return np.clip(gray, 0.0, 1.0).astype(np.float32)
    mn = float(gray.min())
    mx = float(gray.max())
    if mx > mn:
        gray = (gray - mn) / (mx - mn)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)
    return gray.astype(np.float32)


def normalize_rgb_image(rgb: np.ndarray, apply_normalize: bool) -> np.ndarray:
    rgb = unit_scale(rgb)
    if not apply_normalize:
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for ch in range(rgb.shape[2]):
        channel = rgb[:, :, ch]
        mn = float(channel.min())
        mx = float(channel.max())
        if mx > mn:
            out[:, :, ch] = (channel - mn) / (mx - mn)
        else:
            out[:, :, ch] = 0.0
    return out.astype(np.float32)


def normalize_signed(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(arr.mean())
    arr = arr / (float(arr.std()) + 1e-6)
    arr = np.clip(arr, -3.0, 3.0)
    return ((arr + 3.0) / 6.0).astype(np.float32)


def to_gray(img: np.ndarray, apply_normalize: bool) -> np.ndarray:
    if img.ndim == 3:
        return normalize_gray_image(img.mean(axis=2), apply_normalize)
    return normalize_gray_image(img, apply_normalize)


def center_of_mass(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    yy, xx = np.indices((h, w)).astype(np.float32)
    mass = np.maximum(arr, 0.0) + 1e-6
    total = float(mass.sum())
    y_com = float((mass * yy).sum() / total) / max(1.0, h - 1)
    x_com = float((mass * xx).sum() / total) / max(1.0, w - 1)
    return np.array([y_com, x_com], dtype=np.float32)


def resample_1d(signal: np.ndarray, n_out: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(signal), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def pool_grid(arr: np.ndarray, rows: int, cols: int, reducer: str = "mean") -> np.ndarray:
    h, w = arr.shape
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)
    out: list[float] = []
    for ridx in range(rows):
        for cidx in range(cols):
            patch = arr[ys[ridx] : ys[ridx + 1], xs[cidx] : xs[cidx + 1]]
            if patch.size == 0:
                out.append(0.0)
            elif reducer == "std":
                out.append(float(patch.std()))
            elif reducer == "max":
                out.append(float(patch.max()))
            else:
                out.append(float(patch.mean()))
    return np.asarray(out, dtype=np.float32)


def coarse_flatten(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return pool_grid(arr, rows, cols, reducer="mean")


def pool_rgb_grid(rgb: np.ndarray, rows: int, cols: int) -> np.ndarray:
    parts = []
    for ch in range(rgb.shape[2]):
        parts.append(pool_grid(rgb[:, :, ch], rows, cols, reducer="mean"))
        parts.append(pool_grid(rgb[:, :, ch], rows, cols, reducer="std"))
    return np.concatenate(parts).astype(np.float32)


def gradient_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = np.gradient(gray, axis=1).astype(np.float32)
    gy = np.gradient(gray, axis=0).astype(np.float32)
    mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    return gx, gy, mag


def laplacian_np(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros_like(gray, dtype=np.float32)
    lap = np.zeros_like(gray, dtype=np.float32)
    center = gray[1:-1, 1:-1]
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    )
    return lap.astype(np.float32)


def task_warp_nn(gray: np.ndarray, y_power: float, x_power: float, y_start: float = TASK_WARP_Y_START) -> np.ndarray:
    h, w = gray.shape
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y_map = y_start + (1.0 - y_start) * np.power(ys, y_power)
    x_map = 0.5 + 0.5 * np.sign(xs) * np.power(np.abs(xs), x_power)
    row_idx = np.clip(np.round(y_map * (h - 1)).astype(int), 0, h - 1)
    col_idx = np.clip(np.round(x_map * (w - 1)).astype(int), 0, w - 1)
    return gray[np.ix_(row_idx, col_idx)].astype(np.float32)


def task_warp_rgb_nn(rgb: np.ndarray, y_power: float, x_power: float, y_start: float = TASK_WARP_Y_START) -> np.ndarray:
    channels = [task_warp_nn(rgb[:, :, ch], y_power, x_power, y_start=y_start) for ch in range(rgb.shape[2])]
    return np.stack(channels, axis=2).astype(np.float32)


def orientation_histogram(gx: np.ndarray, gy: np.ndarray, rows: int, cols: int, n_bins: int = 8) -> np.ndarray:
    h, w = gx.shape
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)
    angles = (np.arctan2(gy, gx) + np.pi) % np.pi
    mag = np.sqrt(gx ** 2 + gy ** 2)
    feats: list[float] = []
    for ridx in range(rows):
        for cidx in range(cols):
            a = angles[ys[ridx] : ys[ridx + 1], xs[cidx] : xs[cidx + 1]].reshape(-1)
            m = mag[ys[ridx] : ys[ridx + 1], xs[cidx] : xs[cidx + 1]].reshape(-1)
            if a.size == 0:
                feats.extend([0.0] * n_bins)
                continue
            hist, _ = np.histogram(a, bins=n_bins, range=(0.0, np.pi), weights=m)
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-8
            feats.extend(hist.tolist())
    return np.asarray(feats, dtype=np.float32)


def symmetry_features(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    lr = np.abs(gray - np.fliplr(gray))
    tb = np.abs(gray - np.flipud(gray))
    mid = gray.shape[1] // 2
    left = gray[:, :mid]
    right = gray[:, -mid:] if mid > 0 else gray
    if left.size and right.size:
        overlap = np.abs(left - np.fliplr(right))
        corr = float(np.mean(left * np.fliplr(right)))
        mean_gap = float(overlap.mean())
        std_gap = float(overlap.std())
    else:
        corr = 0.0
        mean_gap = 0.0
        std_gap = 0.0
    return np.array(
        [float(lr.mean()), float(lr.std()), float(tb.mean()), float(tb.std()), corr, mean_gap, std_gap],
        dtype=np.float32,
    )


def radial_ring_features(gray: np.ndarray, n_rings: int = 6) -> np.ndarray:
    h, w = gray.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius /= max(float(radius.max()), 1e-6)
    feats: list[float] = []
    edges = np.linspace(0.0, 1.0, n_rings + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radius >= lo) & (radius < hi if hi < 1.0 else radius <= hi)
        vals = gray[mask]
        if vals.size == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([float(vals.mean()), float(vals.std()), float(np.percentile(vals, 80))])
    return np.asarray(feats, dtype=np.float32)


def threshold_mass_features(gray: np.ndarray, thresholds: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8)) -> np.ndarray:
    h, w = gray.shape
    lower = gray[int(0.55 * h) :, :]
    center = gray[int(0.45 * h) :, int(0.20 * w) : int(0.80 * w)]
    feats: list[float] = []
    for thr in thresholds:
        mask = gray <= thr
        lower_mask = lower <= thr if lower.size else np.array([], dtype=bool)
        center_mask = center <= thr if center.size else np.array([], dtype=bool)
        mask_f = mask.astype(np.float32)
        left = mask_f[:, : w // 2]
        right = mask_f[:, w // 2 :]
        feats.extend(
            [
                float(mask.mean()),
                float(lower_mask.mean()) if lower_mask.size else 0.0,
                float(center_mask.mean()) if center_mask.size else 0.0,
                *center_of_mass(mask_f),
                float(left.mean() - right.mean()) if left.size and right.size else 0.0,
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def projection_features(gray: np.ndarray, bins: int = 20) -> np.ndarray:
    gx, gy, mag = gradient_maps(gray)
    return np.concatenate(
        [
            resample_1d(gray.mean(axis=1), bins),
            resample_1d(gray.mean(axis=0), bins),
            resample_1d(mag.mean(axis=1), bins),
            resample_1d(mag.mean(axis=0), bins),
            resample_1d(np.abs(gx).mean(axis=1), bins),
            resample_1d(np.abs(gy).mean(axis=0), bins),
        ]
    ).astype(np.float32)


def spectrum_features(gray: np.ndarray, size: int = 8) -> np.ndarray:
    spec = np.abs(np.fft.rfft2(gray.astype(np.float32)))
    spec = np.log1p(spec).astype(np.float32)
    spec /= float(spec.sum()) + 1e-6
    crop = spec[:size, :size]
    if crop.shape[1] < size:
        padded = np.zeros((size, size), dtype=np.float32)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded
    return crop.reshape(-1).astype(np.float32)


def rgb_summary_features(rgb: np.ndarray, bins: int = 12) -> np.ndarray:
    h, w, _ = rgb.shape
    center = rgb[int(0.45 * h) :, int(0.20 * w) : int(0.80 * w), :]
    lower = rgb[int(0.55 * h) :, :, :]
    means = rgb.mean(axis=(0, 1))
    stds = rgb.std(axis=(0, 1))
    p10 = np.percentile(rgb, 10, axis=(0, 1))
    p90 = np.percentile(rgb, 90, axis=(0, 1))
    center_means = center.mean(axis=(0, 1)) if center.size else np.zeros(3, dtype=np.float32)
    lower_means = lower.mean(axis=(0, 1)) if lower.size else np.zeros(3, dtype=np.float32)
    proj = []
    for ch in range(3):
        proj.append(resample_1d(rgb[:, :, ch].mean(axis=0), bins))
        proj.append(resample_1d(rgb[:, :, ch].mean(axis=1), bins))
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    rb = rgb[:, :, 0] - rgb[:, :, 2]
    gb = rgb[:, :, 1] - rgb[:, :, 2]
    return np.concatenate(
        [
            means.astype(np.float32),
            stds.astype(np.float32),
            p10.astype(np.float32),
            p90.astype(np.float32),
            center_means.astype(np.float32),
            lower_means.astype(np.float32),
            np.array(
                [
                    float(means[0] - means[1]),
                    float(means[0] - means[2]),
                    float(means[1] - means[2]),
                    float(center_means[0] - lower_means[0]),
                    float(center_means[1] - lower_means[1]),
                    float(center_means[2] - lower_means[2]),
                    float(rg.mean()),
                    float(rb.mean()),
                    float(gb.mean()),
                    float(rg.std()),
                    float(rb.std()),
                    float(gb.std()),
                ],
                dtype=np.float32,
            ),
            pool_rgb_grid(rgb, 3, 3),
            np.concatenate(proj).astype(np.float32),
        ]
    ).astype(np.float32)


def percentile_spread_ladders(gray: np.ndarray) -> np.ndarray:
    percentiles = [5, 10, 20, 35, 50, 65, 80, 90, 95]
    vals = np.percentile(gray, percentiles).astype(np.float32)
    spreads = np.diff(vals).astype(np.float32)
    return np.concatenate([vals, spreads]).astype(np.float32)


def rowcol_entropy_summaries(gray: np.ndarray) -> np.ndarray:
    row = gray.mean(axis=1)
    col = gray.mean(axis=0)
    row_p = np.clip(row / (row.sum() + 1e-8), 1e-8, 1.0)
    col_p = np.clip(col / (col.sum() + 1e-8), 1e-8, 1.0)
    row_ent = -np.sum(row_p * np.log(row_p)) / np.log(len(row_p))
    col_ent = -np.sum(col_p * np.log(col_p)) / np.log(len(col_p))
    return np.array(
        [
            float(row_ent),
            float(col_ent),
            float(row.max() - row.min()),
            float(col.max() - col.min()),
            float(np.std(np.diff(row))),
            float(np.std(np.diff(col))),
        ],
        dtype=np.float32,
    )


def center_border_contrast_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    center = gray[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)]
    mask = np.ones_like(gray, dtype=bool)
    mask[int(0.25 * h) : int(0.75 * h), int(0.25 * w) : int(0.75 * w)] = False
    border = gray[mask]
    center_mean = float(center.mean()) if center.size else 0.0
    center_std = float(center.std()) if center.size else 0.0
    border_mean = float(border.mean()) if border.size else 0.0
    border_std = float(border.std()) if border.size else 0.0
    return np.array(
        [
            center_mean,
            center_std,
            border_mean,
            border_std,
            center_mean - border_mean,
            center_std - border_std,
        ],
        dtype=np.float32,
    )


def quadrant_imbalance_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    quads = [
        gray[: h // 2, : w // 2],
        gray[: h // 2, w // 2 :],
        gray[h // 2 :, : w // 2],
        gray[h // 2 :, w // 2 :],
    ]
    means = np.array([float(q.mean()) if q.size else 0.0 for q in quads], dtype=np.float32)
    stds = np.array([float(q.std()) if q.size else 0.0 for q in quads], dtype=np.float32)
    return np.concatenate([means, stds, np.abs(means[:, None] - means[None, :]).reshape(-1)]).astype(np.float32)


def horizon_band_transition_features(gray: np.ndarray) -> np.ndarray:
    h = gray.shape[0]
    bands = [gray[int(lo * h) : int(hi * h)] for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))]
    means = np.array([float(b.mean()) if b.size else 0.0 for b in bands], dtype=np.float32)
    stds = np.array([float(b.std()) if b.size else 0.0 for b in bands], dtype=np.float32)
    diffs = np.diff(means).astype(np.float32)
    return np.concatenate([means, stds, diffs]).astype(np.float32)


def dark_run_continuity_features(gray: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    mask = gray <= threshold
    row_runs = []
    for row in mask:
        padded = np.concatenate([[0], row.astype(np.int8), [0]])
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        runs = ends - starts
        row_runs.append(float(runs.max()) if runs.size else 0.0)
    col_runs = []
    for col in mask.T:
        padded = np.concatenate([[0], col.astype(np.int8), [0]])
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        runs = ends - starts
        col_runs.append(float(runs.max()) if runs.size else 0.0)
    row_runs = np.asarray(row_runs, dtype=np.float32)
    col_runs = np.asarray(col_runs, dtype=np.float32)
    return np.array(
        [
            float(row_runs.mean()),
            float(row_runs.std()),
            float(np.percentile(row_runs, 90)),
            float(col_runs.mean()),
            float(col_runs.std()),
            float(np.percentile(col_runs, 90)),
        ],
        dtype=np.float32,
    )


def corridor_width_proxy_features(gray: np.ndarray, bright_threshold: float = 0.55) -> np.ndarray:
    h, w = gray.shape
    lower = gray[int(0.45 * h) :, :]
    widths: list[float] = []
    center_col = w // 2
    for row in lower:
        bright = row >= bright_threshold
        left = center_col
        right = center_col
        while left > 0 and bright[left - 1]:
            left -= 1
        while right < w - 1 and bright[right + 1]:
            right += 1
        widths.append(float(right - left + 1) / max(1, w))
    width_arr = np.asarray(widths, dtype=np.float32)
    return np.array(
        [
            float(width_arr.mean()),
            float(width_arr.std()),
            float(np.percentile(width_arr, 25)),
            float(np.percentile(width_arr, 75)),
            float(np.mean(width_arr > 0.35)),
        ],
        dtype=np.float32,
    )


def dominant_orientation_concentration_features(gray: np.ndarray) -> np.ndarray:
    gx, gy, _ = gradient_maps(gray)
    full_hist = orientation_histogram(gx, gy, rows=1, cols=1, n_bins=8)
    entropy = -np.sum(np.clip(full_hist, 1e-8, 1.0) * np.log(np.clip(full_hist, 1e-8, 1.0))) / np.log(len(full_hist))
    top2 = np.sort(full_hist)[-2:]
    verticalish = float(full_hist[0] + full_hist[-1])
    horizontalish = float(full_hist[len(full_hist) // 2])
    return np.array(
        [
            float(full_hist.max()),
            float(top2.sum()),
            float(entropy),
            verticalish,
            horizontalish,
            float(verticalish - horizontalish),
        ],
        dtype=np.float32,
    )


def freq_energy_ratio_features(gray: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))) ** 2
    h, w = spec.shape
    yy, xx = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius /= max(float(radius.max()), 1e-6)
    low = float(spec[radius <= 0.2].sum())
    mid = float(spec[(radius > 0.2) & (radius <= 0.5)].sum())
    high = float(spec[radius > 0.5].sum())
    total = low + mid + high + 1e-6
    return np.array(
        [
            low / total,
            mid / total,
            high / total,
            low / (high + 1e-6),
            (low + mid) / (high + 1e-6),
        ],
        dtype=np.float32,
    )


def patch_variance_hist_features(gray: np.ndarray) -> np.ndarray:
    vars_ = pool_grid(gray, 6, 6, reducer="std") ** 2
    hist, _ = np.histogram(vars_, bins=6, range=(0.0, max(float(vars_.max()), 1e-6)))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    return np.concatenate(
        [
            np.array([float(vars_.mean()), float(vars_.std()), float(np.percentile(vars_, 90))], dtype=np.float32),
            hist,
        ]
    ).astype(np.float32)


def reference_anchor_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    center = gray[int(0.45 * h) :, int(0.20 * w) : int(0.80 * w)]
    lower = gray[int(0.55 * h) :, :]
    gx, gy, mag = gradient_maps(gray)
    return np.concatenate(
        [
            np.array(
                [
                    float(gray.mean()),
                    float(gray.std()),
                    float(center.mean()) if center.size else 0.0,
                    float(center.std()) if center.size else 0.0,
                    float(lower.mean()) if lower.size else 0.0,
                    float(lower.std()) if lower.size else 0.0,
                    float(np.abs(gx).mean()),
                    float(np.abs(gy).mean()),
                    float(mag.mean()),
                    float(mag.std()),
                ],
                dtype=np.float32,
            ),
            pool_grid(gray, 4, 4, reducer="mean"),
        ]
    ).astype(np.float32)


def build_views(gray: np.ndarray, rgb: np.ndarray | None, use_warp: bool) -> dict[str, np.ndarray]:
    h, w = gray.shape
    center_slice = (slice(int(0.45 * h), h), slice(int(0.18 * w), int(0.82 * w)))
    lower_slice = (slice(int(0.55 * h), h), slice(0, w))
    gx, gy, mag = gradient_maps(gray)
    detrended = normalize_signed(gray - gray.mean(axis=0, keepdims=True) - gray.mean(axis=1, keepdims=True) + gray.mean())
    grad_mag = normalize_signed(mag)
    views: dict[str, np.ndarray] = {
        "gray_full": gray,
        "gray_center": gray[center_slice],
        "gray_lower": gray[lower_slice],
        "gray_detrended": detrended,
        "gray_grad_mag": grad_mag,
        "gray_edge_x": normalize_signed(np.abs(gx)),
        "gray_edge_y": normalize_signed(np.abs(gy)),
        "gx": gx,
        "gy": gy,
        "mag": mag,
    }
    if use_warp:
        views["gray_warp_soft"] = task_warp_nn(gray, TASK_WARP_Y_POWER_SOFT, TASK_WARP_X_POWER_SOFT)
        views["gray_warp_hard"] = task_warp_nn(gray, TASK_WARP_Y_POWER_HARD, TASK_WARP_X_POWER_HARD)
    if rgb is not None:
        views["rgb_full"] = rgb
        views["rgb_center"] = rgb[center_slice]
        views["rgb_lower"] = rgb[lower_slice]
        if use_warp:
            views["rgb_warp_soft"] = task_warp_rgb_nn(rgb, TASK_WARP_Y_POWER_SOFT, TASK_WARP_X_POWER_SOFT)
    return views


def feature_vector_for_group(group: str, gray: np.ndarray, rgb: np.ndarray | None, use_warp: bool) -> np.ndarray:
    views = build_views(gray, rgb, use_warp=use_warp)
    full = views["gray_full"]
    center = views["gray_center"]
    lower = views["gray_lower"]
    gx = views["gx"]
    gy = views["gy"]
    mag = views["mag"]
    grad_mag = views["gray_grad_mag"]
    detrended = views["gray_detrended"]

    if group == ANCHOR_GROUP:
        return reference_anchor_features(full)
    if group == "gray_global_stats":
        lap = laplacian_np(full)
        return np.array(
            [
                float(full.mean()),
                float(full.std()),
                float(np.percentile(full, 10)),
                float(np.percentile(full, 50)),
                float(np.percentile(full, 90)),
                float(lap.var()),
                *center_of_mass(full),
            ],
            dtype=np.float32,
        )
    if group == "gray_region_stats":
        bottom = full[int(0.70 * full.shape[0]) :, :]
        return np.array(
            [
                float(center.mean()) if center.size else 0.0,
                float(center.std()) if center.size else 0.0,
                float(lower.mean()) if lower.size else 0.0,
                float(lower.std()) if lower.size else 0.0,
                float(bottom.mean()) if bottom.size else 0.0,
                float(bottom.std()) if bottom.size else 0.0,
                float(center.mean() - lower.mean()) if center.size and lower.size else 0.0,
                float(center.std() - lower.std()) if center.size and lower.size else 0.0,
            ],
            dtype=np.float32,
        )
    if group == "gradient_stats":
        return np.array(
            [
                float(mag.mean()),
                float(mag.std()),
                float(np.percentile(mag, 90)),
                float(np.abs(gx).mean()),
                float(np.abs(gy).mean()),
                *center_of_mass(mag),
            ],
            dtype=np.float32,
        )
    if group == "projection_profiles":
        return projection_features(full, bins=24)
    if group == "orientation_histograms":
        return orientation_histogram(gx, gy, rows=3, cols=3, n_bins=8)
    if group == "symmetry":
        return symmetry_features(full)
    if group == "radial_rings":
        return np.concatenate([radial_ring_features(full, n_rings=6), radial_ring_features(lower, n_rings=4)]).astype(np.float32)
    if group == "threshold_mass":
        return np.concatenate([threshold_mass_features(full), threshold_mass_features(grad_mag, thresholds=(0.40, 0.55, 0.70, 0.82))]).astype(np.float32)
    if group == "spectrum_summaries":
        return np.concatenate([spectrum_features(full, size=8), spectrum_features(detrended, size=8)]).astype(np.float32)
    if group == "pooled_gray_grids":
        return np.concatenate(
            [
                pool_grid(full, 5, 5, reducer="mean"),
                pool_grid(full, 5, 5, reducer="std"),
                pool_grid(mag, 5, 5, reducer="mean"),
                pool_grid(mag, 5, 5, reducer="std"),
            ]
        ).astype(np.float32)
    if group == "rgb_summary":
        if rgb is None:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(
            [
                rgb_summary_features(views["rgb_full"], bins=12),
                rgb_summary_features(views["rgb_center"], bins=8),
                rgb_summary_features(views["rgb_lower"], bins=8),
            ]
        ).astype(np.float32)
    if group == "multiview_deltas":
        feats = [
            float(center.mean() - lower.mean()) if center.size and lower.size else 0.0,
            float(center.std() - lower.std()) if center.size and lower.size else 0.0,
            float(center_of_mass(center)[0]) if center.size else 0.0,
            float(center_of_mass(lower)[0]) if lower.size else 0.0,
        ]
        return np.asarray(feats, dtype=np.float32)
    if group == "pca_coarse_blocks":
        blocks = [
            coarse_flatten(full, 18, 18),
            coarse_flatten(center, 16, 16),
            coarse_flatten(lower, 16, 16),
            coarse_flatten(grad_mag, 18, 18),
        ]
        return np.concatenate(blocks).astype(np.float32)
    if group == "warped_view_groups":
        if not use_warp:
            return np.zeros(0, dtype=np.float32)
        warp_soft = views["gray_warp_soft"]
        warp_hard = views["gray_warp_hard"]
        return np.concatenate(
            [
                np.array(
                    [
                        float(warp_soft.mean()),
                        float(warp_soft.std()),
                        float(warp_hard.mean()),
                        float(warp_hard.std()),
                        float(warp_soft.mean() - full.mean()),
                        float(warp_hard.mean() - full.mean()),
                    ],
                    dtype=np.float32,
                ),
                projection_features(warp_soft, bins=16),
                projection_features(warp_hard, bins=16),
            ]
        ).astype(np.float32)
    if group == "detrended_gradmag_groups":
        edge_x = views["gray_edge_x"]
        edge_y = views["gray_edge_y"]
        return np.concatenate(
            [
                np.array(
                    [
                        float(detrended.mean()),
                        float(detrended.std()),
                        float(grad_mag.mean()),
                        float(grad_mag.std()),
                        float(edge_x.mean()),
                        float(edge_y.mean()),
                        float(np.mean(edge_x[:, : edge_x.shape[1] // 2]) - np.mean(edge_x[:, edge_x.shape[1] // 2 :])),
                    ],
                    dtype=np.float32,
                ),
                spectrum_features(grad_mag, size=8),
                spectrum_features(detrended, size=8),
            ]
        ).astype(np.float32)
    if group == "percentile_spread_ladders":
        return percentile_spread_ladders(full)
    if group == "rowcol_entropy":
        return rowcol_entropy_summaries(full)
    if group == "center_border_contrast":
        return center_border_contrast_features(full)
    if group == "quadrant_imbalance":
        return quadrant_imbalance_features(full)
    if group == "horizon_band_transition":
        return horizon_band_transition_features(full)
    if group == "dark_run_continuity":
        return dark_run_continuity_features(full)
    if group == "corridor_width_proxy":
        return corridor_width_proxy_features(full)
    if group == "dominant_orientation_concentration":
        return dominant_orientation_concentration_features(full)
    if group == "freq_energy_ratio":
        return freq_energy_ratio_features(full)
    if group == "patch_variance_hist":
        return patch_variance_hist_features(full)
    raise KeyError(f"Unknown feature group: {group}")


class DataRepository:
    def __init__(self) -> None:
        ensure_dirs()
        self._y: np.ndarray | None = None

    def _image_bank_path(self, split: str, load_rgb: bool, downsample: int) -> Path:
        return CACHE_DIR / f"image_bank_{split}_{'rgb' if load_rgb else 'gray'}_ds{downsample}.npy"

    def _gray_bank_path(self, split: str, load_rgb: bool, downsample: int, normalize: bool) -> Path:
        return CACHE_DIR / f"gray_bank_{split}_{'rgb' if load_rgb else 'gray'}_ds{downsample}_norm{int(normalize)}.npy"

    def _rgb_bank_path(self, split: str, downsample: int, normalize: bool) -> Path:
        return CACHE_DIR / f"rgb_bank_{split}_rgb_ds{downsample}_norm{int(normalize)}.npy"

    def load_y(self) -> np.ndarray:
        if self._y is None:
            config = {"data_dir": ROOT / "data", "load_rgb": True, "downsample_factor": 3}
            _, y = load_dataset(config)
            self._y = np.asarray(y, dtype=np.float32)
        return self._y.copy()

    def load_image_bank(self, split: str, load_rgb: bool, downsample: int) -> np.ndarray:
        path = self._image_bank_path(split, load_rgb, downsample)
        if path.exists():
            print(f"[cache-hit] {path.name}")
            return np.load(path)
        config = {"data_dir": ROOT / "data", "load_rgb": load_rgb, "downsample_factor": downsample}
        if split == "train":
            flats, y = load_dataset(config)
            self._y = np.asarray(y, dtype=np.float32)
        else:
            flats = np.asarray(load_test_dataset(config), dtype=np.float32)
        flats = np.asarray(flats, dtype=np.float32)
        h = 300 // downsample
        w = 300 // downsample
        if load_rgb:
            bank = flats.reshape(len(flats), h, w, 3).astype(np.float32)
        else:
            bank = flats.reshape(len(flats), h, w).astype(np.float32)
        np.save(path, bank)
        print(f"[cache-save] {path.name}")
        return bank

    def load_gray_bank(self, split: str, load_rgb: bool, downsample: int, normalize: bool) -> np.ndarray:
        path = self._gray_bank_path(split, load_rgb, downsample, normalize)
        if path.exists():
            print(f"[cache-hit] {path.name}")
            return np.load(path)
        raw = self.load_image_bank(split, load_rgb, downsample)
        if load_rgb:
            gray_bank = np.stack([to_gray(img, normalize) for img in raw]).astype(np.float32)
        else:
            gray_bank = np.stack([normalize_gray_image(img, normalize) for img in raw]).astype(np.float32)
        np.save(path, gray_bank)
        print(f"[cache-save] {path.name}")
        return gray_bank

    def load_rgb_bank(self, split: str, downsample: int, normalize: bool) -> np.ndarray:
        path = self._rgb_bank_path(split, downsample, normalize)
        if path.exists():
            print(f"[cache-hit] {path.name}")
            return np.load(path)
        raw = self.load_image_bank(split, True, downsample)
        rgb_bank = np.stack([normalize_rgb_image(img, normalize) for img in raw]).astype(np.float32)
        np.save(path, rgb_bank)
        print(f"[cache-save] {path.name}")
        return rgb_bank


class FeatureRepository:
    def __init__(self, data_repo: DataRepository) -> None:
        self.data_repo = data_repo

    def _group_path(self, split: str, preproc: PreprocConfig, group: str) -> Path:
        return CACHE_DIR / f"feat_{split}_{group}_{preproc.short_tag()}.npy"

    def get_group_matrix(self, split: str, preproc: PreprocConfig, group: str) -> np.ndarray:
        path = self._group_path(split, preproc, group)
        if path.exists():
            print(f"[cache-hit] {path.name}")
            return np.load(path)
        gray_bank = self.data_repo.load_gray_bank(split, preproc.load_rgb, preproc.downsample, preproc.normalize)
        rgb_bank = self.data_repo.load_rgb_bank(split, preproc.downsample, preproc.normalize) if preproc.load_rgb else None
        rows: list[np.ndarray] = []
        for idx in range(len(gray_bank)):
            gray = gray_bank[idx]
            rgb = rgb_bank[idx] if rgb_bank is not None else None
            rows.append(feature_vector_for_group(group, gray, rgb, use_warp=preproc.use_warp))
        if not rows:
            matrix = np.zeros((0, 0), dtype=np.float32)
        elif rows[0].size == 0:
            matrix = np.zeros((len(rows), 0), dtype=np.float32)
        else:
            matrix = np.vstack(rows).astype(np.float32)
        np.save(path, matrix)
        print(f"[cache-save] {path.name}")
        return matrix

    def supported_groups(self, preproc: PreprocConfig, groups: list[str]) -> list[str]:
        supported: list[str] = []
        for group in groups:
            if group == "rgb_summary" and not preproc.load_rgb:
                continue
            if group == "warped_view_groups" and not preproc.use_warp:
                continue
            if preproc.view_family != "combined" and group in GROUP_FAMILY and GROUP_FAMILY[group] != preproc.view_family:
                continue
            supported.append(group)
        return supported

    def assemble_fold_features(
        self,
        split: str,
        preproc: PreprocConfig,
        groups: list[str],
        train_idx: np.ndarray,
        query_idx: np.ndarray,
        fit_train_artifacts: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[tuple[str, slice]], dict[str, Any]]:
        artifacts = {} if fit_train_artifacts is None else dict(fit_train_artifacts)
        matrices: list[np.ndarray] = []
        slices: list[tuple[str, slice]] = []
        cursor = 0
        for group in self.supported_groups(preproc, groups):
            raw = self.get_group_matrix(split, preproc, group)
            if raw.shape[1] == 0:
                continue
            if group == "pca_coarse_blocks":
                if fit_train_artifacts is None:
                    scaler = StandardScaler()
                    train_scaled = scaler.fit_transform(raw[train_idx])
                    n_comp = min(PCA_BLOCK_DIM, train_scaled.shape[0], train_scaled.shape[1])
                    pca = PCA(n_components=n_comp, svd_solver="full")
                    train_transformed = pca.fit_transform(train_scaled).astype(np.float32)
                    artifacts[group] = {"scaler": scaler, "pca": pca}
                    block = train_transformed
                else:
                    scaler = artifacts[group]["scaler"]
                    pca = artifacts[group]["pca"]
                    block = pca.transform(scaler.transform(raw[query_idx])).astype(np.float32)
            else:
                block = raw[train_idx] if fit_train_artifacts is None else raw[query_idx]
            if fit_train_artifacts is None and group != "pca_coarse_blocks":
                block = raw[train_idx]
            elif fit_train_artifacts is not None and group != "pca_coarse_blocks":
                block = raw[query_idx]
            matrices.append(np.asarray(block, dtype=np.float32))
            next_cursor = cursor + block.shape[1]
            slices.append((group, slice(cursor, next_cursor)))
            cursor = next_cursor
        if not matrices:
            return np.zeros((len(query_idx), 0), dtype=np.float32), slices, artifacts
        return np.hstack(matrices).astype(np.float32), slices, artifacts


class NeighborDescriptorRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pca_dim: int = 180, neighbor_count: int = 40, alpha: float = 1.0):
        self.pca_dim = pca_dim
        self.neighbor_count = neighbor_count
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.y_ = np.asarray(y, dtype=np.float32)
        self.scaler_ = RobustScaler()
        Xs = self.scaler_.fit_transform(np.asarray(X, dtype=np.float32))
        n_comp = min(self.pca_dim, Xs.shape[0], Xs.shape[1])
        self.pca_ = PCA(n_components=n_comp, svd_solver="full")
        self.Xp_ = self.pca_.fit_transform(Xs).astype(np.float32)
        k_desc = min(self.neighbor_count + 1, len(self.Xp_))
        self.nn_ = NearestNeighbors(n_neighbors=k_desc, metric="euclidean")
        self.nn_.fit(self.Xp_)
        d, i = self.nn_.kneighbors(self.Xp_)
        d = d[:, 1:]
        i = i[:, 1:]
        X_desc = build_neighbor_descriptors(i, d, self.y_)
        self.model_ = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])
        self.model_.fit(X_desc, self.y_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32))
        Xp = self.pca_.transform(Xs).astype(np.float32)
        d, i = self.nn_.kneighbors(Xp, n_neighbors=min(self.neighbor_count, len(self.Xp_)))
        X_desc = build_neighbor_descriptors(i, d, self.y_)
        return np.asarray(self.model_.predict(X_desc), dtype=np.float32)


class LocalRidgeNeighborsRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pca_dim: int = 180, neighbor_count: int = 48, alpha: float = 1.0):
        self.pca_dim = pca_dim
        self.neighbor_count = neighbor_count
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.y_ = np.asarray(y, dtype=np.float32)
        self.scaler_ = RobustScaler()
        Xs = self.scaler_.fit_transform(np.asarray(X, dtype=np.float32))
        n_comp = min(self.pca_dim, Xs.shape[0], Xs.shape[1])
        self.pca_ = PCA(n_components=n_comp, svd_solver="full")
        self.Xp_ = self.pca_.fit_transform(Xs).astype(np.float32)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.neighbor_count, len(self.Xp_)), metric="euclidean")
        self.nn_.fit(self.Xp_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float32))
        Xp = self.pca_.transform(Xs).astype(np.float32)
        _, idx = self.nn_.kneighbors(Xp)
        preds = np.zeros(len(Xp), dtype=np.float32)
        for row_idx in range(len(Xp)):
            Xi = self.Xp_[idx[row_idx]]
            yi = self.y_[idx[row_idx]]
            model = Ridge(alpha=self.alpha)
            model.fit(Xi, yi)
            preds[row_idx] = model.predict(Xp[row_idx : row_idx + 1])[0]
        return preds.astype(np.float32)


def build_neighbor_descriptors(indices: np.ndarray, distances: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    eps = 1e-6
    inv_d = 1.0 / (distances + eps)
    weights = inv_d / np.clip(inv_d.sum(axis=1, keepdims=True), eps, None)
    y_neighbors = y_ref[indices]
    weighted_mean = (weights * y_neighbors).sum(axis=1)
    weighted_var = (weights * (y_neighbors - weighted_mean[:, None]) ** 2).sum(axis=1)
    weighted_std = np.sqrt(np.maximum(weighted_var, 0.0))
    desc = [
        weighted_mean,
        weighted_std,
        y_neighbors[:, 0],
        np.median(y_neighbors[:, : min(5, y_neighbors.shape[1])], axis=1),
        np.median(y_neighbors, axis=1),
        y_neighbors.min(axis=1),
        y_neighbors.max(axis=1),
        distances.mean(axis=1),
        distances.std(axis=1),
        distances[:, 0],
    ]
    return np.column_stack(desc).astype(np.float32)


def make_model(model_spec: ModelSpec, quick: bool) -> BaseEstimator:
    params = model_spec.params_dict
    family = model_spec.family
    if family == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=400 if quick else 900,
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 63)),
            min_samples_leaf=int(params.get("min_samples_leaf", 8)),
            l2_regularization=0.3,
            random_state=RANDOM_STATE,
        )
    if family == "hgb_log":
        base = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=400 if quick else 900,
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 63)),
            min_samples_leaf=int(params.get("min_samples_leaf", 8)),
            l2_regularization=0.3,
            random_state=RANDOM_STATE,
        )
        return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
    if family == "et":
        return ExtraTreesRegressor(
            n_estimators=300 if quick else 900,
            max_depth=None,
            min_samples_leaf=int(params.get("min_samples_leaf", 4)),
            max_features=float(params.get("max_features", 0.45)),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if family == "neighbor_ridge":
        return NeighborDescriptorRegressor(
            pca_dim=int(params.get("pca_dim", 180)),
            neighbor_count=int(params.get("neighbor_count", 40)),
            alpha=float(params.get("alpha", 1.0)),
        )
    if family == "local_ridge":
        return LocalRidgeNeighborsRegressor(
            pca_dim=int(params.get("pca_dim", 180)),
            neighbor_count=int(params.get("neighbor_count", 48)),
            alpha=float(params.get("alpha", 1.0)),
        )
    if family == "nys_ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "nys",
                    Nystroem(
                        kernel="rbf",
                        gamma=float(params.get("gamma", 0.02)),
                        n_components=int(params.get("n_components", 350 if quick else 700)),
                        random_state=RANDOM_STATE,
                    ),
                ),
                ("ridge", Ridge(alpha=float(params.get("alpha", 1.0)))),
            ]
        )
    raise KeyError(f"Unsupported model family: {family}")


def clip_predictions(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(pred, dtype=np.float32), float(np.min(y)), float(np.max(y))).astype(np.float32)


def group_permutation_importance(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    group_slices: list[tuple[str, slice]],
    baseline_mae: float,
    rng: np.random.Generator,
    repeats: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if X_val.shape[1] == 0:
        return rows
    feature_importances = getattr(model, "feature_importances_", None)
    for group, slc in group_slices:
        deltas: list[float] = []
        for _ in range(repeats):
            shuffled = X_val.copy()
            order = rng.permutation(len(shuffled))
            shuffled[:, slc] = shuffled[order][:, slc]
            pred = model.predict(shuffled)
            mae = mean_absolute_error(y_val, pred)
            deltas.append(float(mae - baseline_mae))
        row = {
            "group": group,
            "perm_delta_mae": float(np.mean(deltas)),
        }
        if feature_importances is not None:
            row["group_impurity_importance"] = float(np.sum(feature_importances[slc]))
        rows.append(row)
    return rows


class ExperimentDriver:
    def __init__(self, quick: bool = False, max_groups: int | None = None) -> None:
        self.quick = quick
        self.max_groups = max_groups
        self.data_repo = DataRepository()
        self.feature_repo = FeatureRepository(self.data_repo)
        self.y = self.data_repo.load_y()
        self.screen_splits = 2 if quick else 3
        self.full_splits = 3 if quick else 5

    def _cv_cache_path(
        self,
        model_spec: ModelSpec,
        preproc: PreprocConfig,
        groups: list[str],
        n_splits: int,
        compute_importance: bool,
    ) -> Path:
        key = stable_hash(
            {
                "model_family": model_spec.family,
                "model_params": model_spec.params,
                "preproc": preproc.short_tag(),
                "groups": groups,
                "n_splits": n_splits,
                "compute_importance": compute_importance,
                "quick": self.quick,
            }
        )
        return CACHE_DIR / f"cv_result_{key}.npz"

    def _load_cv_cache(self, path: Path) -> CVResult:
        blob = np.load(path, allow_pickle=False)
        return CVResult(
            model_family=str(blob["model_family"][0]),
            model_tag=str(blob["model_tag"][0]),
            preproc_tag=str(blob["preproc_tag"][0]),
            groups=json.loads(str(blob["groups_json"][0])),
            overall_mae=float(blob["overall_mae"][0]),
            per_bin=json.loads(str(blob["per_bin_json"][0])),
            oof_pred=blob["oof_pred"].astype(np.float32),
            fold_rows=json.loads(str(blob["fold_rows_json"][0])),
            importance_rows=json.loads(str(blob["importance_rows_json"][0])),
        )

    def _save_cv_cache(self, path: Path, result: CVResult) -> None:
        np.savez_compressed(
            path,
            model_family=np.array([result.model_family]),
            model_tag=np.array([result.model_tag]),
            preproc_tag=np.array([result.preproc_tag]),
            groups_json=np.array([json.dumps(result.groups)]),
            overall_mae=np.array([result.overall_mae], dtype=np.float32),
            per_bin_json=np.array([json.dumps(result.per_bin)]),
            oof_pred=result.oof_pred.astype(np.float32),
            fold_rows_json=np.array([json.dumps(result.fold_rows)]),
            importance_rows_json=np.array([json.dumps(result.importance_rows)]),
        )

    def require_previous_phase(self, phase: int) -> None:
        if phase <= 1:
            return
        for prev in range(1, phase):
            if not PHASE_OUTPUTS[prev].exists():
                raise FileNotFoundError(
                    f"Phase {phase} requires {PHASE_OUTPUTS[prev].name}. Run phase {prev} first."
                )

    def evaluate_model_cv(
        self,
        model_spec: ModelSpec,
        preproc: PreprocConfig,
        groups: list[str],
        n_splits: int,
        compute_importance: bool = False,
    ) -> CVResult:
        groups = self.feature_repo.supported_groups(preproc, groups)
        cache_path = self._cv_cache_path(model_spec, preproc, groups, n_splits, compute_importance)
        if cache_path.exists():
            print(f"[cache-hit] {cache_path.name}")
            return self._load_cv_cache(cache_path)
        splits = get_cv_splits(self.y, n_splits=n_splits)
        oof = np.zeros(len(self.y), dtype=np.float32)
        fold_rows: list[dict[str, Any]] = []
        importance_rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(RANDOM_STATE)
        for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
            Xtr, _, artifacts = self.feature_repo.assemble_fold_features("train", preproc, groups, tr_idx, tr_idx)
            Xva, group_slices, _ = self.feature_repo.assemble_fold_features(
                "train", preproc, groups, tr_idx, va_idx, fit_train_artifacts=artifacts
            )
            model = make_model(model_spec, quick=self.quick)
            model.fit(Xtr, self.y[tr_idx])
            pred = clip_predictions(model.predict(Xva), self.y)
            oof[va_idx] = pred
            fold_mae = float(mean_absolute_error(self.y[va_idx], pred))
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "mae": fold_mae,
                    **{f"bin_{label}": score for label, score in per_bin_mae(self.y[va_idx], pred).items()},
                }
            )
            if compute_importance:
                for row in group_permutation_importance(
                    model,
                    Xva,
                    self.y[va_idx],
                    group_slices,
                    baseline_mae=fold_mae,
                    rng=rng,
                    repeats=1 if self.quick else 2,
                ):
                    importance_rows.append(
                        {
                            "model_family": model_spec.family,
                            "model_tag": model_spec.short_tag(),
                            "fold": fold_idx,
                            **row,
                        }
                    )
        overall = float(mean_absolute_error(self.y, oof))
        result = CVResult(
            model_family=model_spec.family,
            model_tag=model_spec.short_tag(),
            preproc_tag=preproc.short_tag(),
            groups=groups,
            overall_mae=overall,
            per_bin=per_bin_mae(self.y, oof),
            oof_pred=oof,
            fold_rows=fold_rows,
            importance_rows=importance_rows,
        )
        self._save_cv_cache(cache_path, result)
        print(f"[cache-save] {cache_path.name}")
        return result

    def _phase1_groups(self) -> list[str]:
        groups = CANDIDATE_GROUPS.copy()
        if self.max_groups is not None:
            groups = groups[: self.max_groups]
        return groups

    def run_phase1(self) -> pd.DataFrame:
        preproc = PreprocConfig(load_rgb=True, downsample=4, normalize=True, use_warp=True, view_family="combined")
        anchor_spec = ModelSpec("hgb", tuple())
        anchor_result = self.evaluate_model_cv(anchor_spec, preproc, [ANCHOR_GROUP], n_splits=self.screen_splits)
        rows: list[dict[str, Any]] = []
        print(f"Phase 1 anchor CV MAE: {anchor_result.overall_mae:.5f}")
        for group in self._phase1_groups():
            result = self.evaluate_model_cv(anchor_spec, preproc, [ANCHOR_GROUP, group], n_splits=self.screen_splits)
            delta = float(result.overall_mae - anchor_result.overall_mae)
            if delta <= FEATURE_USEFUL_THRESHOLD:
                status = "useful"
            elif delta >= FEATURE_HARMFUL_THRESHOLD:
                status = "harmful"
            else:
                status = "neutral"
            rows.append(
                {
                    "group": group,
                    "source": GROUP_SOURCE[group],
                    "family": GROUP_FAMILY[group],
                    "reference_preproc": preproc.short_tag(),
                    "reference_model": anchor_spec.family,
                    "anchor_cv_mae": anchor_result.overall_mae,
                    "cv_mae": result.overall_mae,
                    "mean_delta_mae": delta,
                    "classification": status,
                    **{f"bin_{label}": score for label, score in result.per_bin.items()},
                }
            )
            print(f"Phase 1 | {group:<34} delta={delta:+.5f} -> {status}")
        df = pd.DataFrame(rows).sort_values(["mean_delta_mae", "group"]).reset_index(drop=True)
        df.to_csv(PHASE_OUTPUTS[1], index=False)
        print(f"Saved {PHASE_OUTPUTS[1]}")
        return df

    def _phase2_shortlist(self, df: pd.DataFrame) -> list[str]:
        useful = df.loc[df["classification"] == "useful", "group"].tolist()
        neutral = df.loc[df["classification"] == "neutral", ["group", "mean_delta_mae"]].sort_values("mean_delta_mae")
        shortlist = useful.copy()
        for group in neutral["group"].tolist():
            if len(shortlist) >= 10:
                break
            shortlist.append(group)
        return shortlist or df.nsmallest(8, "mean_delta_mae")["group"].tolist()

    def run_phase2(self) -> pd.DataFrame:
        phase1 = pd.read_csv(PHASE_OUTPUTS[1])
        shortlisted = self._phase2_shortlist(phase1)
        preproc = PreprocConfig(load_rgb=True, downsample=3, normalize=True, use_warp=True, view_family="combined")
        model_specs = [
            ModelSpec("hgb", tuple()),
            ModelSpec("et", tuple()),
            ModelSpec("neighbor_ridge", tuple()),
        ]
        rows: list[dict[str, Any]] = []
        importance_rows: list[dict[str, Any]] = []
        for spec in model_specs:
            result = self.evaluate_model_cv(spec, preproc, shortlisted, n_splits=self.full_splits, compute_importance=True)
            rows.append(
                {
                    "section": "model_compare",
                    "model_family": spec.family,
                    "model_tag": result.model_tag,
                    "preproc": preproc.short_tag(),
                    "groups": "|".join(shortlisted),
                    "cv_mae": result.overall_mae,
                    **{f"bin_{label}": score for label, score in result.per_bin.items()},
                }
            )
            importance_rows.extend(result.importance_rows)
            print(f"Phase 2 | {spec.family:<16} CV MAE={result.overall_mae:.5f}")
        imp_df = pd.DataFrame(importance_rows)
        agg = (
            imp_df.groupby(["group", "model_family"], as_index=False)
            .agg({"perm_delta_mae": "mean", "group_impurity_importance": "mean"})
            .fillna(0.0)
        )
        perm_summary = agg.groupby("group", as_index=False)["perm_delta_mae"].mean().rename(columns={"perm_delta_mae": "mean_perm_delta_mae"})
        merged = perm_summary.merge(phase1[["group", "classification", "mean_delta_mae"]], on="group", how="left")
        merged["retained"] = (
            (merged["classification"] == "useful") | (merged["mean_perm_delta_mae"] >= 0.0008)
        ).astype(int)
        if int(merged["retained"].sum()) < 6:
            top_groups = merged.sort_values("mean_perm_delta_mae", ascending=False).head(6)["group"]
            merged.loc[merged["group"].isin(top_groups), "retained"] = 1
        merged["section"] = "group_summary"
        out = pd.concat([pd.DataFrame(rows), agg.assign(section="group_importance"), merged], ignore_index=True, sort=False)
        out.to_csv(PHASE_OUTPUTS[2], index=False)
        print(f"Saved {PHASE_OUTPUTS[2]}")
        return out

    def retained_groups(self) -> list[str]:
        phase2 = pd.read_csv(PHASE_OUTPUTS[2])
        groups = phase2.loc[(phase2["section"] == "group_summary") & (phase2["retained"] == 1), "group"].tolist()
        return groups

    def best_phase2_model(self) -> str:
        phase2 = pd.read_csv(PHASE_OUTPUTS[2])
        models = phase2.loc[phase2["section"] == "model_compare"].sort_values("cv_mae")
        return str(models.iloc[0]["model_family"])

    def run_phase3(self) -> pd.DataFrame:
        groups = self.retained_groups()
        model_family = self.best_phase2_model()
        model_spec = ModelSpec(model_family, tuple())
        candidates = [
            ("rgb", PreprocConfig(True, 3, True, True, "combined")),
            ("gray", PreprocConfig(False, 3, True, True, "combined")),
            ("ds3", PreprocConfig(True, 3, True, True, "combined")),
            ("ds4", PreprocConfig(True, 4, True, True, "combined")),
            ("ds5", PreprocConfig(True, 5, True, True, "combined")),
            ("norm_on", PreprocConfig(True, 3, True, True, "combined")),
            ("norm_off", PreprocConfig(True, 3, False, True, "combined")),
            ("warp_on", PreprocConfig(True, 3, True, True, "combined")),
            ("warp_off", PreprocConfig(True, 3, True, False, "combined")),
            ("multiview_only", PreprocConfig(True, 3, True, True, "multiview")),
            ("shape_only", PreprocConfig(True, 3, True, True, "shape")),
            ("combined", PreprocConfig(True, 3, True, True, "combined")),
        ]
        rows: list[dict[str, Any]] = []
        for label, preproc in candidates:
            result = self.evaluate_model_cv(model_spec, preproc, groups, n_splits=self.full_splits)
            row = {
                "setting": label,
                "model_family": model_family,
                "preproc": preproc.short_tag(),
                "cv_mae": result.overall_mae,
                **{f"bin_{bin_label}": score for bin_label, score in result.per_bin.items()},
            }
            rows.append(row)
            print(
                f"Phase 3 | {label:<16} CV={result.overall_mae:.5f} "
                f"low={result.per_bin['[0.25,1.00)']:.5f} mid={result.per_bin['[1.00,2.00)']:.5f} high={result.per_bin['[2.00,2.78)']:.5f}"
            )
        df = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
        df.to_csv(PHASE_OUTPUTS[3], index=False)
        print(f"Saved {PHASE_OUTPUTS[3]}")
        return df

    def best_phase3_preproc(self) -> PreprocConfig:
        df = pd.read_csv(PHASE_OUTPUTS[3]).sort_values("cv_mae")
        tag = str(df.iloc[0]["preproc"])
        parts = tag.split("_")
        return PreprocConfig(
            load_rgb=parts[0] == "rgb",
            downsample=int(parts[1].removeprefix("ds")),
            normalize=bool(int(parts[2].removeprefix("norm"))),
            use_warp=bool(int(parts[3].removeprefix("warp"))),
            view_family=parts[4],
        )

    def phase4_search_space(self) -> dict[str, list[ModelSpec]]:
        return {
            "neighbor_ridge": [
                ModelSpec("neighbor_ridge", tuple(sorted(params.items())))
                for params in (
                    {"neighbor_count": 32, "pca_dim": 128},
                    {"neighbor_count": 40, "pca_dim": 180},
                    {"neighbor_count": 56, "pca_dim": 220},
                )
            ],
            "local_ridge": [
                ModelSpec("local_ridge", tuple(sorted(params.items())))
                for params in (
                    {"neighbor_count": 40, "pca_dim": 128, "alpha": 0.5},
                    {"neighbor_count": 48, "pca_dim": 180, "alpha": 1.0},
                    {"neighbor_count": 56, "pca_dim": 220, "alpha": 2.0},
                )
            ],
            "hgb": [
                ModelSpec("hgb", tuple(sorted(params.items())))
                for params in (
                    {"max_leaf_nodes": 31, "min_samples_leaf": 8, "learning_rate": 0.04},
                    {"max_leaf_nodes": 63, "min_samples_leaf": 8, "learning_rate": 0.03},
                    {"max_leaf_nodes": 127, "min_samples_leaf": 12, "learning_rate": 0.02},
                )
            ],
            "hgb_log": [
                ModelSpec("hgb_log", tuple(sorted(params.items())))
                for params in (
                    {"max_leaf_nodes": 31, "min_samples_leaf": 8, "learning_rate": 0.04},
                    {"max_leaf_nodes": 63, "min_samples_leaf": 8, "learning_rate": 0.03},
                    {"max_leaf_nodes": 127, "min_samples_leaf": 12, "learning_rate": 0.02},
                )
            ],
            "et": [
                ModelSpec("et", tuple(sorted(params.items())))
                for params in (
                    {"min_samples_leaf": 2, "max_features": 0.35},
                    {"min_samples_leaf": 4, "max_features": 0.45},
                    {"min_samples_leaf": 8, "max_features": 0.60},
                )
            ],
            "nys_ridge": [ModelSpec("nys_ridge", tuple())],
        }

    def run_phase4(self) -> pd.DataFrame:
        groups = self.retained_groups()
        preproc = self.best_phase3_preproc()
        rows: list[dict[str, Any]] = []
        for family in ["neighbor_ridge", "local_ridge", "hgb", "hgb_log", "et", "nys_ridge"]:
            specs = self.phase4_search_space()[family]
            for spec in specs:
                result = self.evaluate_model_cv(spec, preproc, groups, n_splits=self.full_splits)
                rows.append(
                    {
                        "model_family": family,
                        "model_tag": spec.short_tag(),
                        "preproc": preproc.short_tag(),
                        "cv_mae": result.overall_mae,
                        **{f"bin_{label}": score for label, score in result.per_bin.items()},
                    }
                )
                print(f"Phase 4 | {spec.short_tag():<48} CV={result.overall_mae:.5f}")
        df = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
        df.to_csv(PHASE_OUTPUTS[4], index=False)
        print(f"Saved {PHASE_OUTPUTS[4]}")
        return df

    def best_phase4_model(self) -> ModelSpec:
        df = pd.read_csv(PHASE_OUTPUTS[4]).sort_values("cv_mae")
        tag = str(df.iloc[0]["model_tag"])
        family = str(df.iloc[0]["model_family"])
        if "__" not in tag:
            return ModelSpec(family, tuple())
        pieces = tag.split("__")[1:]
        params: list[tuple[str, Any]] = []
        for piece in pieces:
            key, raw = piece.split("-", 1)
            if raw.replace(".", "", 1).isdigit():
                value = float(raw) if "." in raw else int(raw)
            else:
                value = raw
            params.append((key, value))
        return ModelSpec(family, tuple(params))

    def diagnostic_proxy_matrix(self, preproc: PreprocConfig) -> tuple[list[str], np.ndarray]:
        gray_bank = self.data_repo.load_gray_bank("train", preproc.load_rgb, preproc.downsample, preproc.normalize)
        names = [
            "brightness_mean",
            "brightness_std",
            "local_contrast",
            "center_dark_mass",
            "center_obstacle_proxy",
            "vertical_edge_ratio",
            "symmetry_lr_gap",
            "corridor_width_mean",
            "projection_flatness",
            "orientation_peak",
            "freq_low_high_ratio",
        ]
        rows: list[np.ndarray] = []
        for gray in gray_bank:
            gx, gy, mag = gradient_maps(gray)
            center = gray[int(0.45 * gray.shape[0]) :, int(0.20 * gray.shape[1]) : int(0.80 * gray.shape[1])]
            corridor = corridor_width_proxy_features(gray)
            orient = dominant_orientation_concentration_features(gray)
            freq = freq_energy_ratio_features(gray)
            projection = projection_features(gray, bins=16)
            row = np.array(
                [
                    float(gray.mean()),
                    float(gray.std()),
                    float(pool_grid(gray, 6, 6, reducer="std").mean()),
                    float((center <= 0.35).mean()) if center.size else 0.0,
                    float((center <= 0.50).mean()) if center.size else 0.0,
                    float(np.abs(gx).mean() / (np.abs(gx).mean() + np.abs(gy).mean() + 1e-6)),
                    float(symmetry_features(gray)[0]),
                    float(corridor[0]),
                    float(np.std(np.diff(projection[:16]))),
                    float(orient[0]),
                    float(freq[3]),
                ],
                dtype=np.float32,
            )
            rows.append(row)
        return names, np.vstack(rows).astype(np.float32)

    def run_phase5(self) -> pd.DataFrame:
        groups = self.retained_groups()
        preproc = self.best_phase3_preproc()
        model_spec = self.best_phase4_model()
        result = self.evaluate_model_cv(model_spec, preproc, groups, n_splits=self.full_splits)
        residual = np.abs(self.y - result.oof_pred)
        low_mask = self.y < 1.0
        proxy_names, proxy_matrix = self.diagnostic_proxy_matrix(preproc)
        low_resid = residual[low_mask]
        hard_cut = float(np.quantile(low_resid, 0.75)) if np.any(low_mask) else 0.0
        hard_mask = low_mask.copy()
        hard_mask[low_mask] = low_resid >= hard_cut
        rows: list[dict[str, Any]] = []
        for idx, name in enumerate(proxy_names):
            values = proxy_matrix[:, idx]
            low_values = values[low_mask]
            if low_values.size:
                for metric, value in {
                    "low_q10": float(np.percentile(low_values, 10)),
                    "low_q50": float(np.percentile(low_values, 50)),
                    "low_q90": float(np.percentile(low_values, 90)),
                }.items():
                    rows.append({"section": "low_bin_ranges", "descriptor": name, "metric": metric, "value": value})
            hard_values = values[hard_mask]
            easy_values = values[low_mask & ~hard_mask]
            if hard_values.size and easy_values.size:
                rows.extend(
                    [
                        {
                            "section": "hard_vs_easy",
                            "descriptor": name,
                            "metric": "hard_mean_minus_easy_mean",
                            "value": float(hard_values.mean() - easy_values.mean()),
                        },
                        {
                            "section": "hard_vs_easy",
                            "descriptor": name,
                            "metric": "low_bin_abs_error_corr",
                            "value": float(np.corrcoef(low_values, low_resid)[0, 1]) if low_values.std() > 0 else 0.0,
                        },
                    ]
                )
        rows.append(
            {
                "section": "summary",
                "descriptor": "low_bin_support",
                "metric": "count",
                "value": float(int(low_mask.sum())),
            }
        )
        rows.append(
            {
                "section": "summary",
                "descriptor": "hard_low_bin_support",
                "metric": "count",
                "value": float(int(hard_mask.sum())),
            }
        )
        df = pd.DataFrame(rows)
        df.to_csv(PHASE_OUTPUTS[5], index=False)
        print(f"Saved {PHASE_OUTPUTS[5]}")
        return df

    def run_phase6(self, write_submission: bool = False) -> pd.DataFrame:
        groups = self.retained_groups()
        preproc = self.best_phase3_preproc()
        model_spec = self.best_phase4_model()
        result = self.evaluate_model_cv(model_spec, preproc, groups, n_splits=self.full_splits)
        rows = [
            {
                "final_preproc": preproc.short_tag(),
                "final_model_family": model_spec.family,
                "final_model_tag": model_spec.short_tag(),
                "selected_groups": "|".join(groups),
                "cv_mae": result.overall_mae,
                **{f"bin_{label}": score for label, score in result.per_bin.items()},
            }
        ]
        df = pd.DataFrame(rows)
        df.to_csv(PHASE_OUTPUTS[6], index=False)
        print(f"Saved {PHASE_OUTPUTS[6]}")
        if write_submission:
            train_idx = np.arange(len(self.y))
            X_train, _, artifacts = self.feature_repo.assemble_fold_features("train", preproc, groups, train_idx, train_idx)
            X_test, _, _ = self.feature_repo.assemble_fold_features(
                "test", preproc, groups, train_idx, np.arange(len(self.data_repo.load_image_bank('test', preproc.load_rgb, preproc.downsample))), fit_train_artifacts=artifacts
            )
            model = make_model(model_spec, quick=self.quick)
            model.fit(X_train, self.y)
            pred = clip_predictions(model.predict(X_test), self.y)
            save_results(pred, output_path=ROOT / "prediction_experiment_driver_final.csv")
            print("Saved prediction_experiment_driver_final.csv")
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict sklearn-only phased experiment driver")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5, 6], help="Phase to run")
    parser.add_argument("--quick", action="store_true", help="Use lighter CV/search settings for smoke checks")
    parser.add_argument("--max-groups", type=int, default=None, help="Limit the number of Phase 1 groups evaluated")
    parser.add_argument("--write-submission", action="store_true", help="Write the legacy single-model Phase 6 submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    driver = ExperimentDriver(quick=args.quick, max_groups=args.max_groups)
    driver.require_previous_phase(args.phase)
    if args.phase == 1:
        driver.run_phase1()
    elif args.phase == 2:
        driver.run_phase2()
    elif args.phase == 3:
        driver.run_phase3()
    elif args.phase == 4:
        driver.run_phase4()
    elif args.phase == 5:
        driver.run_phase5()
    elif args.phase == 6:
        driver.run_phase6(write_submission=args.write_submission)
    else:
        raise ValueError(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    main()
