from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from experiment_driver import gradient_maps
from full_pipeline_eval_core import (
    ARTIFACT_DIR,
    BaselineContext,
    apply_correction,
    clip_correction,
    mid_core_mask,
)


ROOT = Path(__file__).resolve().parent
FEATURE_CACHE_DIR = ARTIFACT_DIR / "structural_strategy_cache"


def _ensure_cache() -> None:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _image_paths(split: str) -> list[Path]:
    data_dir = ROOT / "data"
    if split == "train":
        labels = pd.read_csv(data_dir / "train_labels.csv", dtype={"ID": str})
        return [data_dir / "train_images" / f"{image_id}.png" for image_id in labels["ID"].tolist()]
    return sorted((data_dir / "test_images").glob("*.png"))


def load_rgb_unit(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    out = np.empty_like(rgb, dtype=np.float32)
    for ch in range(3):
        channel = rgb[:, :, ch]
        mn = float(channel.min())
        mx = float(channel.max())
        out[:, :, ch] = (channel - mn) / (mx - mn + 1e-6)
    return out


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(arr, dtype=np.float32)
    h, w = arr.shape[:2]
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    crop = arr[:h2, :w2]
    if arr.ndim == 2:
        return crop.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3)).astype(np.float32)
    return crop.reshape(h2 // factor, factor, w2 // factor, factor, arr.shape[2]).mean(axis=(1, 3)).astype(np.float32)


def pool_grid(arr: np.ndarray, rows: int, cols: int, reducer: str = "mean") -> np.ndarray:
    h, w = arr.shape
    ys = np.linspace(0, h, rows + 1, dtype=int)
    xs = np.linspace(0, w, cols + 1, dtype=int)
    vals: list[float] = []
    for r in range(rows):
        for c in range(cols):
            patch = arr[ys[r] : ys[r + 1], xs[c] : xs[c + 1]]
            if patch.size == 0:
                vals.append(0.0)
            elif reducer == "std":
                vals.append(float(patch.std()))
            elif reducer == "max":
                vals.append(float(patch.max()))
            else:
                vals.append(float(patch.mean()))
    return np.asarray(vals, dtype=np.float32)


def _box_mean(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(arr, dtype=np.float32)
    pad = np.pad(arr, ((radius, radius), (radius, radius)), mode="edge")
    integ = np.pad(pad, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    out = integ[size:, size:] - integ[:-size, size:] - integ[size:, :-size] + integ[:-size, :-size]
    return (out / float(size * size)).astype(np.float32)


def _weighted_stats(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(6, dtype=np.float32)
    if weights is None:
        return np.asarray(
            [values.mean(), values.std(), values.min(), np.percentile(values, 25), np.percentile(values, 75), values.max()],
            dtype=np.float32,
        )
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    weights = weights / (weights.sum() + 1e-6)
    mean = float((values * weights).sum())
    std = float(np.sqrt(np.maximum(((values - mean) ** 2 * weights).sum(), 0.0)))
    return np.asarray([mean, std, values.min(), np.percentile(values, 25), np.percentile(values, 75), values.max()], dtype=np.float32)


def _flatten_parts(parts: dict[str, np.ndarray]) -> tuple[np.ndarray, list[tuple[str, slice]]]:
    blocks: list[np.ndarray] = []
    slices: list[tuple[str, slice]] = []
    cursor = 0
    for name, values in parts.items():
        block = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        blocks.append(block)
        slices.append((name, slice(cursor, cursor + len(block))))
        cursor += len(block)
    return np.concatenate(blocks).astype(np.float32), slices


def load_or_build_feature_matrix(
    setting_id: str,
    split: str,
    builder: Callable[[np.ndarray], dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[tuple[str, slice]]]:
    _ensure_cache()
    matrix_path = FEATURE_CACHE_DIR / f"{setting_id}_{split}.npy"
    slices_path = FEATURE_CACHE_DIR / f"{setting_id}_{split}_slices.json"
    if matrix_path.exists() and slices_path.exists():
        matrix = np.load(matrix_path).astype(np.float32)
        raw = json.loads(slices_path.read_text(encoding="utf-8"))
        slices = [(row["name"], slice(int(row["start"]), int(row["stop"]))) for row in raw]
        print(f"[cache-hit] {matrix_path.name}")
        return matrix, slices

    rows: list[np.ndarray] = []
    slices: list[tuple[str, slice]] | None = None
    paths = _image_paths(split)
    for idx, path in enumerate(paths):
        flat, current_slices = _flatten_parts(builder(load_rgb_unit(path)))
        rows.append(flat)
        if slices is None:
            slices = current_slices
        if (idx + 1) % 250 == 0:
            print(f"  built {setting_id} {split}: {idx + 1}/{len(paths)}", flush=True)
    matrix = np.vstack(rows).astype(np.float32)
    np.save(matrix_path, matrix)
    serial = [{"name": name, "start": slc.start, "stop": slc.stop} for name, slc in (slices or [])]
    slices_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(f"[cache-save] {matrix_path.name}")
    return matrix, slices or []


def keypoint_descriptor_parts(rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = rgb.mean(axis=2).astype(np.float32)
    gx, gy, mag = gradient_maps(gray)
    sxx = _box_mean(gx * gx, 2)
    syy = _box_mean(gy * gy, 2)
    sxy = _box_mean(gx * gy, 2)
    response = sxx * syy - sxy * sxy - 0.04 * (sxx + syy) ** 2
    response[:5, :] = 0.0
    response[-5:, :] = 0.0
    response[:, :5] = 0.0
    response[:, -5:] = 0.0
    positive = np.maximum(response, 0.0)
    if np.any(positive > 0):
        threshold = float(np.percentile(positive[positive > 0], 88))
        candidate = positive >= threshold
        candidate &= positive >= np.roll(positive, 1, axis=0)
        candidate &= positive >= np.roll(positive, -1, axis=0)
        candidate &= positive >= np.roll(positive, 1, axis=1)
        candidate &= positive >= np.roll(positive, -1, axis=1)
        ys, xs = np.where(candidate)
        scores = positive[ys, xs]
    else:
        ys = xs = scores = np.zeros(0, dtype=np.float32)

    if len(scores) > 80:
        keep = np.argpartition(scores, -80)[-80:]
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    order = np.argsort(scores)[::-1]
    ys, xs, scores = ys[order], xs[order], scores[order].astype(np.float32)
    h, w = gray.shape
    yn = ys.astype(np.float32) / max(1.0, h - 1)
    xn = xs.astype(np.float32) / max(1.0, w - 1)
    top = min(32, len(scores))

    cell_counts = np.zeros((4, 4), dtype=np.float32)
    if len(scores):
        yy = np.clip((yn * 4).astype(int), 0, 3)
        xx = np.clip((xn * 4).astype(int), 0, 3)
        for r, c, sc in zip(yy, xx, scores):
            cell_counts[r, c] += float(sc)
        cell_counts = cell_counts / (cell_counts.sum() + 1e-6)

    descs: list[np.ndarray] = []
    if top:
        angles = (np.arctan2(gy, gx) + np.pi) % np.pi
        for y0, x0 in zip(ys[:top], xs[:top]):
            patch_ang = angles[max(0, y0 - 4) : min(h, y0 + 5), max(0, x0 - 4) : min(w, x0 + 5)].reshape(-1)
            patch_mag = mag[max(0, y0 - 4) : min(h, y0 + 5), max(0, x0 - 4) : min(w, x0 + 5)].reshape(-1)
            hist, _ = np.histogram(patch_ang, bins=8, range=(0.0, np.pi), weights=patch_mag)
            hist = hist.astype(np.float32)
            descs.append(hist / (hist.sum() + 1e-6))
    desc = np.vstack(descs).astype(np.float32) if descs else np.zeros((1, 8), dtype=np.float32)

    pair = np.zeros(18, dtype=np.float32)
    if top >= 3:
        coords = np.column_stack([yn[:top], xn[:top]])
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=2))
        tri = np.triu_indices(top, k=1)
        d = dist[tri]
        a = (np.arctan2(diff[:, :, 0], diff[:, :, 1])[tri] + np.pi) % np.pi
        ah, _ = np.histogram(a, bins=8, range=(0.0, np.pi))
        ah = ah.astype(np.float32) / (ah.sum() + 1e-6)
        nearest = np.partition(dist + np.eye(top, dtype=np.float32) * 9.0, kth=0, axis=1)[:, 0]
        pair = np.concatenate([_weighted_stats(d), _weighted_stats(nearest), ah]).astype(np.float32)

    return {
        "keypoint_response": np.concatenate([[len(scores) / 80.0], _weighted_stats(scores)]).astype(np.float32),
        "keypoint_layout": np.concatenate([_weighted_stats(yn, scores if len(scores) else None), _weighted_stats(xn, scores if len(scores) else None), cell_counts.reshape(-1)]),
        "local_descriptor_histograms": np.concatenate([desc.mean(axis=0), desc.std(axis=0), desc.max(axis=0)]).astype(np.float32),
        "keypoint_pair_geometry": pair.astype(np.float32),
    }


def geometry_aware_parts(rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = block_mean(rgb.mean(axis=2), 3)
    h, w = gray.shape
    ygrid, xgrid = np.indices(gray.shape)
    yn = ygrid.astype(np.float32) / max(1.0, h - 1)
    xn = xgrid.astype(np.float32) / max(1.0, w - 1)
    gx, gy, mag = gradient_maps(gray)
    bands = np.array_split(np.arange(h), 6)
    centers: list[tuple[float, float]] = []
    widths: list[float] = []
    masses: list[float] = []
    dark = gray <= np.quantile(gray, 0.35)
    for band in bands:
        mask = dark[band]
        mass = mask.astype(np.float32)
        total = float(mass.sum()) + 1e-6
        xs = xgrid[band].astype(np.float32) / max(1.0, w - 1)
        ys = ygrid[band].astype(np.float32) / max(1.0, h - 1)
        centers.append((float((ys * mass).sum() / total), float((xs * mass).sum() / total)))
        col_mass = mass.mean(axis=0)
        widths.append(float(np.mean(col_mass > np.quantile(col_mass, 0.65))))
        masses.append(float(mass.mean()))
    centers_arr = np.asarray(centers, dtype=np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    center_diff = np.diff(centers_arr, axis=0)
    center_dist = np.sqrt((center_diff**2).sum(axis=1))
    center_angle = np.arctan2(center_diff[:, 0], center_diff[:, 1] + 1e-6)

    row_open = (gray >= np.quantile(gray, 0.62)).mean(axis=1).astype(np.float32)
    row_dark = dark.mean(axis=1).astype(np.float32)
    horizon = int(np.argmax(np.abs(np.diff(gray.mean(axis=1), prepend=gray.mean(axis=1)[:1]))))
    left_mag = mag[:, : w // 2]
    right_mag = mag[:, w // 2 :]
    left_x = (left_mag * xn[:, : w // 2]).sum(axis=1) / (left_mag.sum(axis=1) + 1e-6)
    right_x = (right_mag * xn[:, w // 2 :]).sum(axis=1) / (right_mag.sum(axis=1) + 1e-6)
    valid = np.linspace(0.0, 1.0, h, dtype=np.float32)
    left_slope = float(np.polyfit(valid, left_x, 1)[0])
    right_slope = float(np.polyfit(valid, right_x, 1)[0])
    width_slope = float(np.polyfit(np.linspace(0.0, 1.0, len(widths_arr)), widths_arr, 1)[0])

    angles = (np.arctan2(gy, gx) + np.pi) % np.pi
    orient_hist, _ = np.histogram(angles, bins=12, range=(0.0, np.pi), weights=mag)
    orient_hist = orient_hist.astype(np.float32) / (orient_hist.sum() + 1e-6)

    return {
        "spatial_layout": np.concatenate([centers_arr.reshape(-1), widths_arr, np.asarray(masses, dtype=np.float32)]).astype(np.float32),
        "distance_features": np.concatenate([center_dist, _weighted_stats(center_dist), _weighted_stats(widths_arr), [float(abs(centers_arr[-1, 1] - 0.5))]]).astype(np.float32),
        "angle_features": np.concatenate([center_angle, orient_hist, [left_slope, right_slope, left_slope + right_slope]]).astype(np.float32),
        "perspective_cues": np.asarray(
            [
                horizon / max(1.0, h - 1),
                width_slope,
                widths_arr[-1] - widths_arr[0],
                row_open[-h // 3 :].mean() - row_open[: h // 3].mean(),
                row_dark[-h // 3 :].mean() - row_dark[: h // 3].mean(),
                np.std(centers_arr[:, 1]),
                np.mean(np.abs(np.diff(centers_arr[:, 1]))),
            ],
            dtype=np.float32,
        ),
    }


def hgb_residual_correction_with_importance(
    context: BaselineContext,
    X_train: np.ndarray,
    X_test: np.ndarray,
    group_slices: list[tuple[str, slice]],
    train_zone: np.ndarray,
    test_zone: np.ndarray,
    *,
    seed_base: int,
    cap: float,
    shrink: float,
    min_abs_raw: float,
    low_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    residual = (context.y - context.promoted_oof).astype(np.float32)
    X_train = np.nan_to_num(X_train.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    oof_correction = np.zeros(len(context.y), dtype=np.float32)
    importance_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed_base)

    for fold, (tr_idx, va_idx) in enumerate(context.splits, start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train[tr_idx])
        Xva = scaler.transform(X_train[va_idx])
        weights = 1.0 + 1.4 * train_zone[tr_idx].astype(np.float32) + low_weight * (context.y[tr_idx] < 1.0)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=140,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=24,
            l2_regularization=0.45,
            random_state=seed_base + fold,
        )
        model.fit(Xtr, residual[tr_idx], sample_weight=weights)
        raw = model.predict(Xva).astype(np.float32)
        gate = train_zone[va_idx] & (np.abs(raw) >= min_abs_raw)
        corr = np.zeros(len(va_idx), dtype=np.float32)
        corr[gate] = clip_correction(raw[gate], cap=cap, shrink=shrink)
        oof_correction[va_idx] = corr
        fold_pred = apply_correction(context, context.promoted_oof[va_idx], corr)
        base_mae = float(mean_absolute_error(context.y[va_idx], fold_pred))
        for group, slc in group_slices:
            shuffled = Xva.copy()
            order = rng.permutation(len(shuffled))
            shuffled[:, slc] = shuffled[order][:, slc]
            raw_shuf = model.predict(shuffled).astype(np.float32)
            gate_shuf = train_zone[va_idx] & (np.abs(raw_shuf) >= min_abs_raw)
            corr_shuf = np.zeros(len(va_idx), dtype=np.float32)
            corr_shuf[gate_shuf] = clip_correction(raw_shuf[gate_shuf], cap=cap, shrink=shrink)
            pred_shuf = apply_correction(context, context.promoted_oof[va_idx], corr_shuf)
            importance_rows.append(
                {
                    "fold": fold,
                    "group": group,
                    "perm_delta_mae": float(mean_absolute_error(context.y[va_idx], pred_shuf) - base_mae),
                    "baseline_fold_mae": base_mae,
                }
            )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    weights = 1.0 + 1.4 * train_zone.astype(np.float32) + low_weight * (context.y < 1.0)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=24,
        l2_regularization=0.45,
        random_state=seed_base + 99,
    )
    model.fit(Xtr, residual, sample_weight=weights)
    raw_test = model.predict(Xte).astype(np.float32)
    test_correction = np.zeros(len(context.promoted_test), dtype=np.float32)
    test_gate = test_zone & (np.abs(raw_test) >= min_abs_raw)
    test_correction[test_gate] = clip_correction(raw_test[test_gate], cap=cap, shrink=shrink)
    imp = pd.DataFrame(importance_rows)
    if not imp.empty:
        imp = imp.groupby("group", as_index=False).agg(
            mean_perm_delta_mae=("perm_delta_mae", "mean"),
            max_perm_delta_mae=("perm_delta_mae", "max"),
            mean_fold_mae=("baseline_fold_mae", "mean"),
        )
        imp = imp.sort_values("mean_perm_delta_mae", ascending=False).reset_index(drop=True)
    return oof_correction, test_correction, imp


def default_uncertain_zones(context: BaselineContext) -> tuple[np.ndarray, np.ndarray]:
    train_mid_safe = mid_core_mask(context, "train")
    test_mid_safe = mid_core_mask(context, "test")
    train_zone = (
        (np.abs(context.promoted_oof - 1.0) < 0.32)
        | (np.abs(context.promoted_oof - 2.0) < 0.25)
        | (context.promoted_oof < 0.95)
        | (context.promoted_oof > 1.90)
        | (~train_mid_safe)
    )
    test_zone = (
        (np.abs(context.promoted_test - 1.0) < 0.32)
        | (np.abs(context.promoted_test - 2.0) < 0.25)
        | (context.promoted_test < 0.95)
        | (context.promoted_test > 1.90)
        | (~test_mid_safe)
    )
    return train_zone.astype(bool), test_zone.astype(bool)


def append_meta_group(
    X_new_train: np.ndarray,
    X_new_test: np.ndarray,
    new_slices: list[tuple[str, slice]],
    meta_train: np.ndarray,
    meta_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, slice]]]:
    offset = X_new_train.shape[1]
    slices = list(new_slices)
    slices.append(("current_pipeline_meta", slice(offset, offset + meta_train.shape[1])))
    return (
        np.hstack([X_new_train, meta_train]).astype(np.float32),
        np.hstack([X_new_test, meta_test]).astype(np.float32),
        slices,
    )


def summarize_feature_importance(setting_id: str, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty or "mean_perm_delta_mae" not in diagnostics.columns:
        return diagnostics
    out = diagnostics.copy()
    out.insert(0, "setting_id", setting_id)
    out.to_csv(ARTIFACT_DIR / f"{setting_id}_feature_importance.csv", index=False)
    return out
