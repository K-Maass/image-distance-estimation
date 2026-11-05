from __future__ import annotations

import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from experiment_driver import ExperimentDriver, clip_predictions, get_cv_splits, per_bin_mae, save_results
from final_evidence_pipeline import BASE_MODELS, FINAL_PREPROC, active_final_groups, weighted_sum_from_dict
from final_local_competence_pipeline import (
    ROUTE_WEIGHT as PROMOTED_ROUTE_WEIGHT,
    build_current_baseline_oof,
    build_full_test_predictions,
    build_router_signals,
    evaluate_bottom_crop_oof,
    fit_router_model,
    optimize_router_threshold,
    predict_router_probability,
    train_bottom_crop_full_and_predict,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

PROMOTED_CV = 0.10042024403810501
PROMOTED_LOW_BIN = 0.2201530635356903
PROMOTED_MID_BIN = 0.08978579938411713
PROMOTED_HIGH_BIN = 0.0806371346116066
PROMOTED_THRESHOLDS = "0.50|0.45|0.55|0.50|0.55"
PROMOTED_ROUTE_FRACTION = 0.47199999999999986

FINAL_EVIDENCE_CV = 0.10096903145313263
ROBUSTNESS_REFERENCE_CV = 0.10025545209646225
ROBUSTNESS_REFERENCE_LOW_BIN = 0.21993039548397064
ROBUSTNESS_REFERENCE_MID_BIN = 0.08959274739027023
ROBUSTNESS_REFERENCE_HIGH_BIN = 0.08059201389551163

BIN_LABELS = ("[0.25,1.00)", "[1.00,2.00)", "[2.00,2.78)")
FIXED_FINAL_BLEND_WEIGHTS = np.asarray([0.483800, 0.424124, 0.049984, 0.040771, 0.001320], dtype=np.float32)


@dataclass
class BaselineContext:
    driver: ExperimentDriver
    y: np.ndarray
    splits: list[tuple[np.ndarray, np.ndarray]]
    active_groups: list[str]
    baseline_oof: np.ndarray
    base_oof_dict: dict[str, np.ndarray]
    blend_names: list[str]
    blend_weights: np.ndarray
    bottom_crop_oof: np.ndarray
    train_signals: np.ndarray
    signal_names: list[str]
    direct_blend_oof: np.ndarray
    promoted_oof: np.ndarray
    promoted_route_prob: np.ndarray
    promoted_route_mask: np.ndarray
    promoted_crop_helps: np.ndarray
    promoted_crop_help_gain: np.ndarray
    fold_ids: np.ndarray
    fold_thresholds: np.ndarray
    fold_rows: list[dict[str, float]]
    baseline_test: np.ndarray
    test_base_dict: dict[str, np.ndarray]
    bottom_crop_test: np.ndarray
    test_signals: np.ndarray
    direct_blend_test: np.ndarray
    promoted_test: np.ndarray
    test_route_prob: np.ndarray
    test_threshold: float
    test_route_fraction: float


@dataclass
class PipelineResult:
    setting_id: str
    script_path: str
    oof_pred: np.ndarray
    test_pred: np.ndarray
    route_mask: np.ndarray | None = None
    test_route_mask: np.ndarray | None = None
    diagnostics: pd.DataFrame | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def bin_codes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values < 1.0, 0, np.where(values < 2.0, 1, 2)).astype(np.int8)


def _direct_blend(y: np.ndarray, baseline_pred: np.ndarray, bottom_crop_pred: np.ndarray) -> np.ndarray:
    return clip_predictions((1.0 - PROMOTED_ROUTE_WEIGHT) * baseline_pred + PROMOTED_ROUTE_WEIGHT * bottom_crop_pred, y).astype(
        np.float32
    )


def apply_router_mask(y: np.ndarray, baseline_pred: np.ndarray, direct_blend_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = baseline_pred.astype(np.float32).copy()
    out[np.asarray(mask, dtype=bool)] = direct_blend_pred[np.asarray(mask, dtype=bool)]
    return clip_predictions(out, y).astype(np.float32)


def build_fixed_final_blend_oof(
    driver: ExperimentDriver,
    active_groups: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], np.ndarray]:
    oof_dict: dict[str, np.ndarray] = {}
    print("Evaluating current final base models with fixed promoted blend weights...")
    for spec in BASE_MODELS:
        result = driver.evaluate_model_cv(
            spec,
            FINAL_PREPROC,
            active_groups,
            n_splits=driver.full_splits,
            compute_importance=False,
        )
        oof_dict[spec.short_tag()] = result.oof_pred.astype(np.float32)
        print(f"  {spec.short_tag():<72} CV={result.overall_mae:.6f}")
    names = [spec.short_tag() for spec in BASE_MODELS]
    weights = FIXED_FINAL_BLEND_WEIGHTS.copy()
    weights = weights / weights.sum()
    baseline_oof = weighted_sum_from_dict(oof_dict, names, weights)
    return baseline_oof.astype(np.float32), oof_dict, names, weights.astype(np.float32)


def build_baseline_context(strict_reproduction: bool = True) -> BaselineContext:
    driver = ExperimentDriver(quick=False)
    y = driver.y.astype(np.float32)
    splits = get_cv_splits(y, n_splits=driver.full_splits)
    active_groups = active_final_groups(driver)

    baseline_oof, base_oof_dict, blend_names, blend_weights = build_fixed_final_blend_oof(driver, active_groups)
    bottom_crop_oof = evaluate_bottom_crop_oof(driver, active_groups)
    train_signals, signal_names = build_router_signals(baseline_oof, base_oof_dict, bottom_crop_oof)
    direct_blend_oof = _direct_blend(y, baseline_oof, bottom_crop_oof)

    baseline_abs = np.abs(y - baseline_oof)
    direct_abs = np.abs(y - direct_blend_oof)
    crop_help_gain = baseline_abs - direct_abs
    crop_helps = (crop_help_gain > 0.0).astype(bool)

    promoted_oof = np.zeros(len(y), dtype=np.float32)
    route_prob = np.zeros(len(y), dtype=np.float32)
    route_mask = np.zeros(len(y), dtype=bool)
    fold_ids = np.zeros(len(y), dtype=np.int16)
    fold_thresholds = np.zeros(len(y), dtype=np.float32)
    fold_rows: list[dict[str, float]] = []

    print("Reconstructing promoted local-competence router...", flush=True)
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        scaler, model = fit_router_model(train_signals[tr_idx], crop_helps[tr_idx].astype(np.int32), y[tr_idx], seed=42 + fold_idx)
        train_prob = predict_router_probability(scaler, model, train_signals[tr_idx])
        val_prob = predict_router_probability(scaler, model, train_signals[va_idx])
        threshold, train_mae = optimize_router_threshold(y[tr_idx], baseline_oof[tr_idx], direct_blend_oof[tr_idx], train_prob)
        val_mask = (val_prob >= threshold) & (baseline_oof[va_idx] >= 1.0)
        promoted_oof[va_idx] = apply_router_mask(y, baseline_oof[va_idx], direct_blend_oof[va_idx], val_mask)
        route_prob[va_idx] = val_prob
        route_mask[va_idx] = val_mask
        fold_ids[va_idx] = fold_idx
        fold_thresholds[va_idx] = threshold
        fold_rows.append(
            {
                "fold": float(fold_idx),
                "threshold": float(threshold),
                "train_mae_at_threshold": float(train_mae),
                "val_route_fraction": float(val_mask.mean()),
                "fold_mae": float(mean_absolute_error(y[va_idx], promoted_oof[va_idx])),
            }
        )
        print(
            f"  fold {fold_idx}: threshold={threshold:.2f} route_frac={val_mask.mean():.3f} "
            f"MAE={fold_rows[-1]['fold_mae']:.6f}",
            flush=True,
        )

    assert_promoted_reproduction(y, baseline_oof, promoted_oof, fold_rows, strict=strict_reproduction)

    print("Building promoted test predictions...", flush=True)
    baseline_test, test_base_dict = build_full_test_predictions(driver, active_groups, blend_names, blend_weights)
    bottom_crop_test = train_bottom_crop_full_and_predict(driver, active_groups)
    test_signals, _ = build_router_signals(baseline_test, test_base_dict, bottom_crop_test)
    direct_blend_test = _direct_blend(y, baseline_test, bottom_crop_test)

    scaler, model = fit_router_model(train_signals, crop_helps.astype(np.int32), y, seed=431)
    full_train_prob = predict_router_probability(scaler, model, train_signals)
    test_threshold, _ = optimize_router_threshold(y, baseline_oof, direct_blend_oof, full_train_prob)
    test_prob = predict_router_probability(scaler, model, test_signals)
    test_mask = (test_prob >= test_threshold) & (baseline_test >= 1.0)
    promoted_test = apply_router_mask(y, baseline_test, direct_blend_test, test_mask)

    return BaselineContext(
        driver=driver,
        y=y,
        splits=splits,
        active_groups=active_groups,
        baseline_oof=baseline_oof.astype(np.float32),
        base_oof_dict={k: v.astype(np.float32) for k, v in base_oof_dict.items()},
        blend_names=blend_names,
        blend_weights=blend_weights.astype(np.float32),
        bottom_crop_oof=bottom_crop_oof.astype(np.float32),
        train_signals=train_signals.astype(np.float32),
        signal_names=signal_names,
        direct_blend_oof=direct_blend_oof.astype(np.float32),
        promoted_oof=promoted_oof.astype(np.float32),
        promoted_route_prob=route_prob.astype(np.float32),
        promoted_route_mask=route_mask,
        promoted_crop_helps=crop_helps,
        promoted_crop_help_gain=crop_help_gain.astype(np.float32),
        fold_ids=fold_ids,
        fold_thresholds=fold_thresholds,
        fold_rows=fold_rows,
        baseline_test=baseline_test.astype(np.float32),
        test_base_dict={k: v.astype(np.float32) for k, v in test_base_dict.items()},
        bottom_crop_test=bottom_crop_test.astype(np.float32),
        test_signals=test_signals.astype(np.float32),
        direct_blend_test=direct_blend_test.astype(np.float32),
        promoted_test=promoted_test.astype(np.float32),
        test_route_prob=test_prob.astype(np.float32),
        test_threshold=float(test_threshold),
        test_route_fraction=float(test_mask.mean()),
    )


def variant_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    bins = per_bin_mae(y, pred)
    return {
        "cv_mae": float(mean_absolute_error(y, pred)),
        "bin_[0.25,1.00)": bins["[0.25,1.00)"],
        "bin_[1.00,2.00)": bins["[1.00,2.00)"],
        "bin_[2.00,2.78)": bins["[2.00,2.78)"],
    }


def assert_promoted_reproduction(
    y: np.ndarray,
    baseline_oof: np.ndarray,
    promoted_oof: np.ndarray,
    fold_rows: list[dict[str, float]],
    strict: bool = True,
) -> None:
    baseline_metrics = variant_metrics(y, baseline_oof)
    promoted_metrics = variant_metrics(y, promoted_oof)
    thresholds = "|".join(f"{row['threshold']:.2f}" for row in fold_rows)
    route_fraction = float(np.mean([row["val_route_fraction"] for row in fold_rows]))
    checks = [
        ("final_evidence_cv", baseline_metrics["cv_mae"], FINAL_EVIDENCE_CV, 1e-7),
        ("promoted_cv", promoted_metrics["cv_mae"], PROMOTED_CV, 1e-7),
        ("promoted_low_bin", promoted_metrics["bin_[0.25,1.00)"], PROMOTED_LOW_BIN, 1e-7),
        ("promoted_mid_bin", promoted_metrics["bin_[1.00,2.00)"], PROMOTED_MID_BIN, 1e-7),
        ("promoted_high_bin", promoted_metrics["bin_[2.00,2.78)"], PROMOTED_HIGH_BIN, 1e-7),
        ("promoted_route_fraction", route_fraction, PROMOTED_ROUTE_FRACTION, 5e-4),
    ]
    failed = [f"{name}: got {value:.12f}, expected {expected:.12f}" for name, value, expected, tol in checks if abs(value - expected) > tol]
    if thresholds != PROMOTED_THRESHOLDS:
        failed.append(f"promoted_thresholds: got {thresholds}, expected {PROMOTED_THRESHOLDS}")
    if failed:
        message = "Promoted router reproduction drifted:\n" + "\n".join(failed)
        if strict:
            raise AssertionError(message)
        print("[warning] " + message.replace("\n", "\n[warning] "), flush=True)


def base_prediction_matrix(context: BaselineContext, split: str) -> tuple[np.ndarray, list[str]]:
    pred_dict = context.base_oof_dict if split == "train" else context.test_base_dict
    names = list(pred_dict.keys())
    return np.column_stack([pred_dict[name] for name in names]).astype(np.float32), names


def standard_meta_features(context: BaselineContext, split: str) -> tuple[np.ndarray, list[str]]:
    if split == "train":
        promoted = context.promoted_oof
        baseline = context.baseline_oof
        bottom = context.bottom_crop_oof
        direct = context.direct_blend_oof
        signals = context.train_signals
        route_prob = context.promoted_route_prob
        route_mask = context.promoted_route_mask.astype(np.float32)
    else:
        promoted = context.promoted_test
        baseline = context.baseline_test
        bottom = context.bottom_crop_test
        direct = context.direct_blend_test
        signals = context.test_signals
        route_prob = context.test_route_prob
        route_mask = ((route_prob >= context.test_threshold) & (baseline >= 1.0)).astype(np.float32)

    base_matrix, base_names = base_prediction_matrix(context, split)
    base_bins = bin_codes(base_matrix)
    entropy = np.zeros(len(base_matrix), dtype=np.float32)
    for code in (0, 1, 2):
        p = (base_bins == code).mean(axis=1).astype(np.float32)
        entropy -= np.where(p > 0, p * np.log(p + 1e-6), 0.0)

    features = [
        signals,
        base_matrix,
        np.column_stack(
            [
                promoted,
                baseline,
                bottom,
                direct,
                direct - baseline,
                promoted - baseline,
                route_prob,
                route_mask,
                np.abs(promoted - 1.0),
                np.abs(promoted - 2.0),
                np.abs(baseline - 1.0),
                np.abs(baseline - 2.0),
                entropy,
                base_bins.std(axis=1),
            ]
        ),
    ]
    names = [
        *[f"signal_{name}" for name in context.signal_names],
        *[f"base_{name}" for name in base_names],
        "promoted_pred",
        "baseline_pred",
        "bottom_crop_pred",
        "direct_blend_pred",
        "direct_minus_baseline",
        "promoted_minus_baseline",
        "router_probability",
        "promoted_route_mask",
        "abs_promoted_to_1",
        "abs_promoted_to_2",
        "abs_baseline_to_1",
        "abs_baseline_to_2",
        "base_regime_entropy",
        "base_regime_code_std",
    ]
    return np.hstack(features).astype(np.float32), names


def mid_core_mask(context: BaselineContext, split: str) -> np.ndarray:
    if split == "train":
        pred = context.baseline_oof
        signals = context.train_signals
    else:
        pred = context.baseline_test
        signals = context.test_signals
    name_to_idx = {name: idx for idx, name in enumerate(context.signal_names)}
    base_min = signals[:, name_to_idx["base_model_prediction_min"]]
    base_max = signals[:, name_to_idx["base_model_prediction_max"]]
    crop_delta_abs = signals[:, name_to_idx["abs_bottom_crop_minus_baseline"]]
    return ((pred >= 1.10) & (pred < 1.90) & (base_min >= 1.0) & (base_max < 2.0) & (crop_delta_abs < 0.20))


def low_vote(context: BaselineContext, split: str) -> np.ndarray:
    signals = context.train_signals if split == "train" else context.test_signals
    return signals[:, context.signal_names.index("base_model_low_vote_fraction")]


def base_max(context: BaselineContext, split: str) -> np.ndarray:
    signals = context.train_signals if split == "train" else context.test_signals
    return signals[:, context.signal_names.index("base_model_prediction_max")]


def crop_delta(context: BaselineContext, split: str) -> np.ndarray:
    signals = context.train_signals if split == "train" else context.test_signals
    return signals[:, context.signal_names.index("bottom_crop_minus_baseline")]


def clip_correction(correction: np.ndarray, cap: float, shrink: float = 1.0) -> np.ndarray:
    return np.clip(np.asarray(correction, dtype=np.float32) * shrink, -cap, cap).astype(np.float32)


def apply_correction(context: BaselineContext, base_pred: np.ndarray, correction: np.ndarray) -> np.ndarray:
    return clip_predictions(np.asarray(base_pred, dtype=np.float32) + np.asarray(correction, dtype=np.float32), context.y).astype(np.float32)


def fit_ridge_residual(
    X_train: np.ndarray,
    residual_train: np.ndarray,
    X_query: np.ndarray,
    alpha: float = 4.0,
) -> np.ndarray:
    if len(X_train) < 20 or X_train.shape[1] == 0:
        return np.zeros(len(X_query), dtype=np.float32)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, residual_train)
    return model.predict(X_query).astype(np.float32)


def fit_hgb_residual(
    X_train: np.ndarray,
    residual_train: np.ndarray,
    X_query: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 120,
) -> np.ndarray:
    if len(X_train) < 40 or X_train.shape[1] == 0:
        return np.zeros(len(X_query), dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=max_iter,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=0.4,
        random_state=seed,
    )
    model.fit(Xs, residual_train, sample_weight=sample_weight)
    return model.predict(scaler.transform(X_query)).astype(np.float32)


def fit_hgb_classifier(
    X_train: np.ndarray,
    target: np.ndarray,
    X_query: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if len(X_train) < 40 or len(np.unique(target)) < 2:
        return np.zeros(len(X_query), dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=seed,
    )
    model.fit(Xs, target, sample_weight=sample_weight)
    return model.predict_proba(scaler.transform(X_query))[:, 1].astype(np.float32)


def fit_regime_classifier(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, seed: int) -> np.ndarray:
    target = bin_codes(y_train)
    if len(np.unique(target)) < 2:
        out = np.zeros((len(X_query), 3), dtype=np.float32)
        out[:, int(target[0])] = 1.0
        return out
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.35, max_iter=800, multi_class="auto", class_weight="balanced", random_state=seed),
    )
    model.fit(X_train, target)
    proba = np.zeros((len(X_query), 3), dtype=np.float32)
    raw = model.predict_proba(X_query).astype(np.float32)
    for col, cls in enumerate(model[-1].classes_):
        proba[:, int(cls)] = raw[:, col]
    return proba


def active_feature_fold(context: BaselineContext, tr_idx: np.ndarray, va_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xtr, _, artifacts = context.driver.feature_repo.assemble_fold_features("train", FINAL_PREPROC, context.active_groups, tr_idx, tr_idx)
    Xva, _, _ = context.driver.feature_repo.assemble_fold_features(
        "train", FINAL_PREPROC, context.active_groups, tr_idx, va_idx, fit_train_artifacts=artifacts
    )
    return Xtr.astype(np.float32), Xva.astype(np.float32)


def active_feature_test(context: BaselineContext) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.arange(len(context.y))
    test_count = len(context.baseline_test)
    test_idx = np.arange(test_count)
    Xtr, _, artifacts = context.driver.feature_repo.assemble_fold_features("train", FINAL_PREPROC, context.active_groups, train_idx, train_idx)
    Xte, _, _ = context.driver.feature_repo.assemble_fold_features(
        "test", FINAL_PREPROC, context.active_groups, train_idx, test_idx, fit_train_artifacts=artifacts
    )
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def retrieval_meta_from_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    residual_train: np.ndarray,
    train_pred: np.ndarray,
    X_query: np.ndarray,
    query_pred: np.ndarray,
    pca_dim: int = 180,
    n_neighbors: int = 35,
) -> np.ndarray:
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X_train)
    Xq = scaler.transform(X_query)
    n_comp = min(pca_dim, Xtr.shape[0], Xtr.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xq = pca.transform(Xq)
    k = min(n_neighbors, max(2, len(Xtr)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(Xtr)
    dist, idx = nn.kneighbors(Xq)
    neigh_y = y_train[idx]
    neigh_resid = residual_train[idx]
    neigh_bin = bin_codes(neigh_y)
    query_bin = bin_codes(query_pred)
    agreement = (neigh_bin == query_bin[:, None]).mean(axis=1)
    return np.column_stack(
        [
            dist[:, 0],
            dist.mean(axis=1),
            dist.std(axis=1),
            neigh_y.mean(axis=1),
            neigh_y.std(axis=1),
            neigh_y.min(axis=1),
            neigh_y.max(axis=1),
            neigh_resid.mean(axis=1),
            neigh_resid.std(axis=1),
            (neigh_bin == 0).mean(axis=1),
            (neigh_bin == 1).mean(axis=1),
            (neigh_bin == 2).mean(axis=1),
            agreement,
            np.abs(query_pred - neigh_y.mean(axis=1)),
        ]
    ).astype(np.float32)


def retrieval_meta_oof_and_test(context: BaselineContext) -> tuple[np.ndarray, np.ndarray]:
    residual = (context.y - context.promoted_oof).astype(np.float32)
    train_meta = np.zeros((len(context.y), 14), dtype=np.float32)
    for tr_idx, va_idx in context.splits:
        Xtr, Xva = active_feature_fold(context, tr_idx, va_idx)
        train_meta[va_idx] = retrieval_meta_from_features(
            Xtr,
            context.y[tr_idx],
            residual[tr_idx],
            context.promoted_oof[tr_idx],
            Xva,
            context.promoted_oof[va_idx],
        )
    Xtr_full, Xte = active_feature_test(context)
    test_meta = retrieval_meta_from_features(
        Xtr_full,
        context.y,
        residual,
        context.promoted_oof,
        Xte,
        context.promoted_test,
    )
    return train_meta, test_meta


def _geometry_one(gray: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    row_mean = gray.mean(axis=1)
    row_dark = (gray < np.quantile(gray, 0.35)).mean(axis=1)
    row_open = (gray > np.quantile(gray, 0.62)).mean(axis=1)
    grad_y = np.abs(np.diff(row_mean, prepend=row_mean[:1]))
    bands = np.array_split(np.arange(h), 6)
    cols = np.arange(w, dtype=np.float32)
    dark = gray < np.quantile(gray, 0.30)
    centers = []
    widths = []
    for band in bands:
        mass = dark[band].mean(axis=0).astype(np.float32) + 1e-6
        centers.append(float((mass * cols).sum() / mass.sum()) / max(1.0, w - 1))
        widths.append(float(np.mean(mass > np.quantile(mass, 0.65))))
    centers = np.asarray(centers, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    left = gray[:, : w // 2]
    right = np.flip(gray[:, w - w // 2 :], axis=1)
    sym = np.mean(np.abs(left - right[:, : left.shape[1]]), axis=1)
    rgb_mean = rgb.mean(axis=(0, 1)) if rgb is not None else np.zeros(3, dtype=np.float32)
    bottom = gray[int(0.65 * h) :]
    top = gray[: int(0.35 * h)]
    coeff_open = np.polyfit(y, row_open, deg=1)[0]
    coeff_dark = np.polyfit(y, row_dark, deg=1)[0]
    coeff_width = np.polyfit(np.linspace(0.0, 1.0, len(widths)), widths, deg=1)[0]
    return np.asarray(
        [
            row_mean[: h // 3].mean(),
            row_mean[h // 3 : 2 * h // 3].mean(),
            row_mean[2 * h // 3 :].mean(),
            row_dark[: h // 3].mean(),
            row_dark[h // 3 : 2 * h // 3].mean(),
            row_dark[2 * h // 3 :].mean(),
            row_open[: h // 3].mean(),
            row_open[h // 3 : 2 * h // 3].mean(),
            row_open[2 * h // 3 :].mean(),
            grad_y.max(),
            grad_y.mean(),
            y[int(np.argmax(grad_y))],
            coeff_open,
            coeff_dark,
            coeff_width,
            widths[0],
            widths[-1],
            centers[0],
            centers[-1],
            float(np.std(centers)),
            sym.mean(),
            sym[2 * h // 3 :].mean(),
            bottom.mean() - top.mean(),
            bottom.std(),
            top.std(),
            *rgb_mean.tolist(),
        ],
        dtype=np.float32,
    )


def geometry_features(context: BaselineContext) -> tuple[np.ndarray, np.ndarray]:
    gray_train = context.driver.data_repo.load_gray_bank("train", FINAL_PREPROC.load_rgb, FINAL_PREPROC.downsample, FINAL_PREPROC.normalize)
    rgb_train = context.driver.data_repo.load_rgb_bank("train", FINAL_PREPROC.downsample, FINAL_PREPROC.normalize)
    gray_test = context.driver.data_repo.load_gray_bank("test", FINAL_PREPROC.load_rgb, FINAL_PREPROC.downsample, FINAL_PREPROC.normalize)
    rgb_test = context.driver.data_repo.load_rgb_bank("test", FINAL_PREPROC.downsample, FINAL_PREPROC.normalize)
    Xtr = np.vstack([_geometry_one(gray_train[i], rgb_train[i]) for i in range(len(gray_train))]).astype(np.float32)
    Xte = np.vstack([_geometry_one(gray_test[i], rgb_test[i]) for i in range(len(gray_test))]).astype(np.float32)
    return Xtr, Xte


def keep_drop_recommendation(metrics: dict[str, float]) -> str:
    if (
        metrics["cv_mae"] <= PROMOTED_CV
        and metrics["bin_[0.25,1.00)"] < PROMOTED_LOW_BIN
        and metrics["bin_[1.00,2.00)"] <= PROMOTED_MID_BIN
        and metrics["bin_[2.00,2.78)"] <= PROMOTED_HIGH_BIN
    ):
        return "keep_for_confirmation_review"
    return "drop_after_screen"


def _summary_artifact_row(context: BaselineContext, result: PipelineResult, metrics: dict[str, float]) -> dict[str, Any]:
    correction = result.oof_pred.astype(np.float32) - context.promoted_oof.astype(np.float32)
    test_drift = result.test_pred.astype(np.float32) - context.promoted_test.astype(np.float32)
    route_mask = result.route_mask if result.route_mask is not None else (np.abs(correction) > 1e-7)
    test_route_mask = result.test_route_mask if result.test_route_mask is not None else (np.abs(test_drift) > 1e-7)
    row: dict[str, Any] = {
        "setting_id": result.setting_id,
        "script_path": result.script_path,
        **metrics,
        "delta_cv_vs_promoted": metrics["cv_mae"] - PROMOTED_CV,
        "delta_low_vs_promoted": metrics["bin_[0.25,1.00)"] - PROMOTED_LOW_BIN,
        "delta_mid_vs_promoted": metrics["bin_[1.00,2.00)"] - PROMOTED_MID_BIN,
        "delta_high_vs_promoted": metrics["bin_[2.00,2.78)"] - PROMOTED_HIGH_BIN,
        "delta_cv_vs_weight014_reference": metrics["cv_mae"] - ROBUSTNESS_REFERENCE_CV,
        "delta_low_vs_weight014_reference": metrics["bin_[0.25,1.00)"] - ROBUSTNESS_REFERENCE_LOW_BIN,
        "delta_mid_vs_weight014_reference": metrics["bin_[1.00,2.00)"] - ROBUSTNESS_REFERENCE_MID_BIN,
        "delta_high_vs_weight014_reference": metrics["bin_[2.00,2.78)"] - ROBUSTNESS_REFERENCE_HIGH_BIN,
        "route_or_correction_fraction": float(np.mean(route_mask)),
        "test_route_or_correction_fraction": float(np.mean(test_route_mask)),
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "mean_abs_correction_active": float(np.mean(np.abs(correction[route_mask]))) if np.any(route_mask) else 0.0,
        "test_mean_abs_drift_vs_promoted": float(np.mean(np.abs(test_drift))),
        "test_mean_drift_vs_promoted": float(np.mean(test_drift)),
        "keep_drop_recommendation": keep_drop_recommendation(metrics),
    }
    row.update(result.metadata)
    return row


def _regime_rows(context: BaselineContext, result: PipelineResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    true_bin = bin_codes(context.y)
    pred_bin = bin_codes(result.oof_pred)
    base_pred_bin = bin_codes(context.promoted_oof)
    test_bin = bin_codes(result.test_pred)
    test_base_bin = bin_codes(context.promoted_test)
    labels = list(BIN_LABELS)
    for i, true_label in enumerate(labels):
        mask = true_bin == i
        denom = int(mask.sum())
        for j, pred_label in enumerate(labels):
            count = int(np.sum(mask & (pred_bin == j)))
            rows.append(
                {
                    "setting_id": result.setting_id,
                    "table": "true_bin_vs_candidate_pred_bin_oof",
                    "from_bin": true_label,
                    "to_bin": pred_label,
                    "count": count,
                    "rate": count / denom if denom else np.nan,
                }
            )
    for i, from_label in enumerate(labels):
        mask = base_pred_bin == i
        denom = int(mask.sum())
        for j, to_label in enumerate(labels):
            count = int(np.sum(mask & (pred_bin == j)))
            rows.append(
                {
                    "setting_id": result.setting_id,
                    "table": "promoted_pred_bin_vs_candidate_pred_bin_oof",
                    "from_bin": from_label,
                    "to_bin": to_label,
                    "count": count,
                    "rate": count / denom if denom else np.nan,
                }
            )
    for i, from_label in enumerate(labels):
        mask = test_base_bin == i
        denom = int(mask.sum())
        for j, to_label in enumerate(labels):
            count = int(np.sum(mask & (test_bin == j)))
            rows.append(
                {
                    "setting_id": result.setting_id,
                    "table": "promoted_pred_bin_vs_candidate_pred_bin_test",
                    "from_bin": from_label,
                    "to_bin": to_label,
                    "count": count,
                    "rate": count / denom if denom else np.nan,
                }
            )
    return rows


def evaluate_pipeline_result(context: BaselineContext, result: PipelineResult) -> pd.DataFrame:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    oof = clip_predictions(result.oof_pred, context.y).astype(np.float32)
    test = clip_predictions(result.test_pred, context.y).astype(np.float32)
    result.oof_pred = oof
    result.test_pred = test

    metrics = variant_metrics(context.y, oof)
    summary_row = _summary_artifact_row(context, result, metrics)

    np.save(ARTIFACT_DIR / f"{result.setting_id}_oof.npy", oof)
    np.save(ARTIFACT_DIR / f"{result.setting_id}_test.npy", test)
    save_results(test, output_path=ARTIFACT_DIR / f"{result.setting_id}_prediction.csv")
    pd.DataFrame([summary_row]).to_csv(ARTIFACT_DIR / f"{result.setting_id}_summary.csv", index=False)
    pd.DataFrame(_regime_rows(context, result)).to_csv(ARTIFACT_DIR / f"{result.setting_id}_regime_crossing.csv", index=False)

    if result.diagnostics is not None:
        diagnostics = result.diagnostics if isinstance(result.diagnostics, pd.DataFrame) else pd.DataFrame([result.diagnostics])
        diagnostics.to_csv(ARTIFACT_DIR / f"{result.setting_id}_diagnostics.csv", index=False)

    print(
        f"{result.setting_id}: CV={metrics['cv_mae']:.6f} "
        f"low={metrics['bin_[0.25,1.00)']:.6f} mid={metrics['bin_[1.00,2.00)']:.6f} "
        f"high={metrics['bin_[2.00,2.78)']:.6f} {summary_row['keep_drop_recommendation']}",
        flush=True,
    )
    return pd.DataFrame([summary_row])


def collect_suite_summaries() -> Path:
    rows = []
    for path in sorted(ARTIFACT_DIR.glob("full_pipeline_*_summary.csv")):
        if path.name == "full_pipeline_suite_summary.csv":
            continue
        rows.append(pd.read_csv(path))
    out_path = ARTIFACT_DIR / "full_pipeline_suite_summary.csv"
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
    return out_path


def run_script(path: Path) -> int:
    return subprocess.call([sys.executable, str(path)])
