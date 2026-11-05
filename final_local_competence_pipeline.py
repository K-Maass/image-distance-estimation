from __future__ import annotations

import argparse
import hashlib
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

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from experiment_driver import (
    ExperimentDriver,
    ModelSpec,
    clip_predictions,
    feature_vector_for_group,
    get_cv_splits,
    make_model,
    per_bin_mae,
    save_results,
)
from final_evidence_pipeline import (
    BASE_MODELS,
    FINAL_GROUPS,
    FINAL_PREPROC,
    active_final_groups,
    search_blend_weights,
    train_full_and_predict,
    weighted_sum_from_dict,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
CROP_CACHE_DIR = ARTIFACT_DIR / "final_local_competence_crop_cache"
SUMMARY_PATH = ARTIFACT_DIR / "final_local_competence_pipeline_summary.csv"
QUICK_SUMMARY_PATH = ARTIFACT_DIR / "final_local_competence_pipeline_summary_quick.csv"
DEFAULT_SUBMISSION_PATH = ROOT / "prediction_final_local_competence_pipeline.csv"

CURRENT_BASELINE_NAME = "final_evidence_pipeline.py"
CURRENT_BASELINE_CV = 0.10096903145313263
CURRENT_BASELINE_KAGGLE = 0.0966
CURRENT_BASELINE_LOW_BIN = 0.2213866412639618
PREVIOUS_BASELINE_NAME = "trial98_sklearn_lowbin_specialist.py"
PREVIOUS_BASELINE_OOF = 0.10023091733455658

SETTING_ID = "final_local_competence_no_scene_diagnostic_proxies"
ROUTE_WEIGHT = 0.11
BOTTOM_CROP_SPEC = ModelSpec("neighbor_ridge", (("neighbor_count", 40), ("pca_dim", 180)))


@dataclass(frozen=True)
class CropSetting:
    setting_id: str
    region: str
    y0: float
    y1: float
    x0: float
    x1: float


BOTTOM_HALF_CROP = CropSetting("crop_bottom_half", "bottom half", 0.50, 1.00, 0.00, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final local-competence candidate pipeline")
    parser.add_argument("--quick", action="store_true", help="Use the experiment driver's lighter CV setting for a smoke test")
    parser.add_argument("--write-submission", action="store_true", help="Fit full-train candidate and write a submission CSV")
    parser.add_argument(
        "--submission-path",
        default=str(DEFAULT_SUBMISSION_PATH),
        help="Where to write the submission CSV when --write-submission is set",
    )
    return parser.parse_args()


def format_per_bin(per_bin: dict[str, float]) -> str:
    return "  ".join(f"{label}={value:.6f}" for label, value in per_bin.items())


def crop_array(arr: np.ndarray, crop: CropSetting) -> np.ndarray:
    height, width = arr.shape[:2]
    y0 = int(round(crop.y0 * height))
    y1 = int(round(crop.y1 * height))
    x0 = int(round(crop.x0 * width))
    x1 = int(round(crop.x1 * width))
    y0 = max(0, min(y0, height - 1))
    x0 = max(0, min(x0, width - 1))
    y1 = max(y0 + 1, min(y1, height))
    x1 = max(x0 + 1, min(x1, width))
    return arr[y0:y1, x0:x1].astype(np.float32, copy=False)


def crop_cache_path(split: str, crop: CropSetting, groups: list[str]) -> Path:
    group_tag = hashlib.sha1("|".join(groups).encode("utf-8")).hexdigest()[:10]
    return CROP_CACHE_DIR / f"{split}_{crop.setting_id}_{FINAL_PREPROC.short_tag()}_{group_tag}.npy"


def build_crop_feature_matrix(
    gray_bank: np.ndarray,
    rgb_bank: np.ndarray,
    crop: CropSetting,
    groups: list[str],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for idx in range(len(gray_bank)):
        gray_crop = crop_array(gray_bank[idx], crop)
        rgb_crop = crop_array(rgb_bank[idx], crop)
        parts = [
            feature_vector_for_group(group, gray_crop, rgb_crop, use_warp=FINAL_PREPROC.use_warp)
            for group in groups
        ]
        rows.append(np.concatenate(parts).astype(np.float32))
    return np.vstack(rows).astype(np.float32)


def load_or_build_crop_features(driver: ExperimentDriver, split: str, crop: CropSetting, groups: list[str]) -> np.ndarray:
    path = crop_cache_path(split, crop, groups)
    if path.exists():
        print(f"[cache-hit] {path.name}")
        return np.load(path)
    gray_bank = driver.data_repo.load_gray_bank(split, FINAL_PREPROC.load_rgb, FINAL_PREPROC.downsample, FINAL_PREPROC.normalize)
    rgb_bank = driver.data_repo.load_rgb_bank(split, FINAL_PREPROC.downsample, FINAL_PREPROC.normalize)
    X = build_crop_feature_matrix(gray_bank, rgb_bank, crop, groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    print(f"[cache-save] {path.name}")
    return X


def build_current_baseline_oof(
    driver: ExperimentDriver,
    active_groups: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], np.ndarray]:
    oof_dict: dict[str, np.ndarray] = {}
    print("Evaluating current final base models...")
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
    names, weights, _ = search_blend_weights(
        oof_dict,
        driver.y,
        n_random=4000 if driver.quick else 20000,
        seed=97,
    )
    baseline_oof = weighted_sum_from_dict(oof_dict, names, weights)
    return baseline_oof.astype(np.float32), oof_dict, names, weights


def evaluate_bottom_crop_oof(driver: ExperimentDriver, active_groups: list[str]) -> np.ndarray:
    y = driver.y.astype(np.float32)
    splits = get_cv_splits(y, n_splits=driver.full_splits)
    X = load_or_build_crop_features(driver, "train", BOTTOM_HALF_CROP, active_groups)
    oof = np.zeros(len(y), dtype=np.float32)
    print(f"Evaluating bottom-half crop expert: {BOTTOM_CROP_SPEC.short_tag()}")
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        model = make_model(BOTTOM_CROP_SPEC, quick=driver.quick)
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = clip_predictions(model.predict(X[va_idx]), y)
        print(f"  fold {fold_idx}: MAE={mean_absolute_error(y[va_idx], oof[va_idx]):.6f}")
    return oof.astype(np.float32)


def train_bottom_crop_full_and_predict(driver: ExperimentDriver, active_groups: list[str]) -> np.ndarray:
    y = driver.y.astype(np.float32)
    X_train = load_or_build_crop_features(driver, "train", BOTTOM_HALF_CROP, active_groups)
    X_test = load_or_build_crop_features(driver, "test", BOTTOM_HALF_CROP, active_groups)
    model = make_model(BOTTOM_CROP_SPEC, quick=driver.quick)
    model.fit(X_train, y)
    return clip_predictions(model.predict(X_test), y)


def build_router_signals(
    baseline_pred: np.ndarray,
    base_pred_dict: dict[str, np.ndarray],
    bottom_crop_pred: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    base_matrix = np.column_stack([base_pred_dict[name] for name in base_pred_dict]).astype(np.float32)
    base_std = base_matrix.std(axis=1)
    base_range = base_matrix.max(axis=1) - base_matrix.min(axis=1)
    sorted_base = np.sort(base_matrix, axis=1)
    low_votes = (base_matrix < 1.0).mean(axis=1)
    crop_delta = bottom_crop_pred - baseline_pred
    signals = np.column_stack(
        [
            baseline_pred,
            bottom_crop_pred,
            crop_delta,
            np.abs(crop_delta),
            base_std,
            base_range,
            sorted_base[:, 0],
            sorted_base[:, -1],
            low_votes,
        ]
    ).astype(np.float32)
    names = [
        "current_baseline_pred",
        "bottom_half_crop_pred",
        "bottom_crop_minus_baseline",
        "abs_bottom_crop_minus_baseline",
        "base_model_prediction_std",
        "base_model_prediction_range",
        "base_model_prediction_min",
        "base_model_prediction_max",
        "base_model_low_vote_fraction",
    ]
    return signals, names


def direct_crop_blend(y: np.ndarray, baseline_pred: np.ndarray, bottom_crop_pred: np.ndarray) -> np.ndarray:
    return clip_predictions((1.0 - ROUTE_WEIGHT) * baseline_pred + ROUTE_WEIGHT * bottom_crop_pred, y)


def fit_router_model(X: np.ndarray, target: np.ndarray, y: np.ndarray, seed: int) -> tuple[StandardScaler, HistGradientBoostingClassifier | None]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if len(np.unique(target)) < 2:
        return scaler, None
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=0.5,
        random_state=seed,
    )
    sample_weight = 1.0 + 2.0 * (y < 1.0)
    model.fit(Xs, target, sample_weight=sample_weight.astype(np.float32))
    return scaler, model


def predict_router_probability(
    scaler: StandardScaler,
    model: HistGradientBoostingClassifier | None,
    X: np.ndarray,
) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=np.float32)
    return model.predict_proba(scaler.transform(X))[:, 1].astype(np.float32)


def optimize_router_threshold(
    y: np.ndarray,
    baseline_pred: np.ndarray,
    direct_blend_pred: np.ndarray,
    route_prob: np.ndarray,
) -> tuple[float, float]:
    thresholds = np.linspace(0.10, 0.90, 17)
    best_threshold = 1.0
    best_mae = float(mean_absolute_error(y, baseline_pred))
    for threshold in thresholds:
        mask = (route_prob >= threshold) & (baseline_pred >= 1.0)
        pred = baseline_pred.copy()
        pred[mask] = direct_blend_pred[mask]
        mae = float(mean_absolute_error(y, pred))
        if mae < best_mae:
            best_mae = mae
            best_threshold = float(threshold)
    return best_threshold, best_mae


def apply_router(
    y: np.ndarray,
    baseline_pred: np.ndarray,
    direct_blend_pred: np.ndarray,
    route_prob: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float]:
    mask = (route_prob >= threshold) & (baseline_pred >= 1.0)
    out = baseline_pred.copy()
    out[mask] = direct_blend_pred[mask]
    return clip_predictions(out, y), float(mask.mean())


def cross_fit_local_competence_router(
    y: np.ndarray,
    baseline_oof: np.ndarray,
    bottom_crop_oof: np.ndarray,
    signals: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    splits = get_cv_splits(y, n_splits=n_splits)
    direct_blend_oof = direct_crop_blend(y, baseline_oof, bottom_crop_oof)
    crop_helps = (np.abs(y - direct_blend_oof) < np.abs(y - baseline_oof)).astype(np.int32)
    routed = np.zeros(len(y), dtype=np.float32)
    fold_rows: list[dict[str, float]] = []
    print("Cross-fitting confirmed local-competence router...")
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        scaler, model = fit_router_model(signals[tr_idx], crop_helps[tr_idx], y[tr_idx], seed=42 + fold_idx)
        train_prob = predict_router_probability(scaler, model, signals[tr_idx])
        val_prob = predict_router_probability(scaler, model, signals[va_idx])
        threshold, train_mae = optimize_router_threshold(
            y[tr_idx],
            baseline_oof[tr_idx],
            direct_blend_oof[tr_idx],
            train_prob,
        )
        routed[va_idx], val_route_fraction = apply_router(
            y,
            baseline_oof[va_idx],
            direct_blend_oof[va_idx],
            val_prob,
            threshold,
        )
        fold_rows.append(
            {
                "fold": fold_idx,
                "threshold": threshold,
                "train_mae_at_threshold": train_mae,
                "val_route_fraction": val_route_fraction,
                "fold_mae": float(mean_absolute_error(y[va_idx], routed[va_idx])),
            }
        )
        print(
            f"  fold {fold_idx}: threshold={threshold:.2f} "
            f"route_frac={val_route_fraction:.3f} MAE={fold_rows[-1]['fold_mae']:.6f}"
        )
    return routed.astype(np.float32), fold_rows


def train_full_router_and_predict_test(
    y: np.ndarray,
    train_baseline_oof: np.ndarray,
    train_bottom_crop_oof: np.ndarray,
    train_signals: np.ndarray,
    test_baseline_pred: np.ndarray,
    test_bottom_crop_pred: np.ndarray,
    test_signals: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    train_direct_blend = direct_crop_blend(y, train_baseline_oof, train_bottom_crop_oof)
    crop_helps = (np.abs(y - train_direct_blend) < np.abs(y - train_baseline_oof)).astype(np.int32)
    scaler, model = fit_router_model(train_signals, crop_helps, y, seed=431)
    train_prob = predict_router_probability(scaler, model, train_signals)
    threshold, _ = optimize_router_threshold(y, train_baseline_oof, train_direct_blend, train_prob)
    test_direct_blend = direct_crop_blend(y, test_baseline_pred, test_bottom_crop_pred)
    test_prob = predict_router_probability(scaler, model, test_signals)
    final_test, route_fraction = apply_router(y, test_baseline_pred, test_direct_blend, test_prob, threshold)
    return final_test.astype(np.float32), threshold, route_fraction


def build_full_test_predictions(
    driver: ExperimentDriver,
    active_groups: list[str],
    blend_names: list[str],
    blend_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    test_pred_dict = {
        spec.short_tag(): train_full_and_predict(driver, spec, active_groups)
        for spec in BASE_MODELS
    }
    baseline_test = weighted_sum_from_dict(test_pred_dict, blend_names, blend_weights)
    return clip_predictions(baseline_test, driver.y), test_pred_dict


def write_summary(summary_path: Path, rows: list[dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved {summary_path.resolve()}")


def main() -> None:
    args = parse_args()
    driver = ExperimentDriver(quick=args.quick)
    y = driver.y.astype(np.float32)
    active_groups = active_final_groups(driver)

    baseline_oof, base_oof_dict, blend_names, blend_weights = build_current_baseline_oof(driver, active_groups)
    bottom_crop_oof = evaluate_bottom_crop_oof(driver, active_groups)
    train_signals, signal_names = build_router_signals(baseline_oof, base_oof_dict, bottom_crop_oof)
    routed_oof, fold_rows = cross_fit_local_competence_router(
        y,
        baseline_oof,
        bottom_crop_oof,
        train_signals,
        n_splits=driver.full_splits,
    )

    baseline_mae = float(mean_absolute_error(y, baseline_oof))
    baseline_bins = per_bin_mae(y, baseline_oof)
    crop_mae = float(mean_absolute_error(y, bottom_crop_oof))
    crop_bins = per_bin_mae(y, bottom_crop_oof)
    candidate_mae = float(mean_absolute_error(y, routed_oof))
    candidate_bins = per_bin_mae(y, routed_oof)

    print("\nFinal Local Competence Candidate")
    print(f"Setting: {SETTING_ID}")
    print(f"Preprocessing: {FINAL_PREPROC.short_tag()}")
    print(f"Active feature groups: {', '.join(active_groups)}")
    print(f"Current submission baseline: {CURRENT_BASELINE_NAME} CV={CURRENT_BASELINE_CV:.6f} Kaggle={CURRENT_BASELINE_KAGGLE:.4f}")
    print(f"Previous historical baseline: {PREVIOUS_BASELINE_NAME} OOF={PREVIOUS_BASELINE_OOF:.6f}")
    print(f"Rebuilt final blend CV MAE: {baseline_mae:.6f}")
    print(f"Rebuilt final blend per-bin: {format_per_bin(baseline_bins)}")
    print(f"Bottom-half crop standalone CV MAE: {crop_mae:.6f}")
    print(f"Bottom-half crop standalone per-bin: {format_per_bin(crop_bins)}")
    print(f"Candidate CV MAE: {candidate_mae:.6f}")
    print(f"Candidate per-bin: {format_per_bin(candidate_bins)}")

    summary_path = QUICK_SUMMARY_PATH if args.quick else SUMMARY_PATH
    rows: list[dict[str, object]] = [
        {
            "setting_id": SETTING_ID,
            "variant": "current_submission_baseline_rebuilt",
            "cv_mae": baseline_mae,
            "delta_vs_current_submission_cv": baseline_mae - CURRENT_BASELINE_CV,
            "current_submission_baseline": CURRENT_BASELINE_NAME,
            "current_submission_baseline_cv": CURRENT_BASELINE_CV,
            "current_submission_baseline_kaggle": CURRENT_BASELINE_KAGGLE,
            "previous_historical_baseline": PREVIOUS_BASELINE_NAME,
            "previous_historical_baseline_oof": PREVIOUS_BASELINE_OOF,
            "preproc": FINAL_PREPROC.short_tag(),
            "active_groups": "|".join(active_groups),
            "base_models": "|".join(spec.short_tag() for spec in BASE_MODELS),
            "blend_components": "|".join(blend_names),
            "blend_weights": "|".join(f"{w:.6f}" for w in blend_weights),
            "signals_used_for_routing": "none",
            "route_weight": 0.0,
            "mean_route_fraction": 0.0,
            "fold_router_thresholds": "",
            **{f"bin_{label}": score for label, score in baseline_bins.items()},
        },
        {
            "setting_id": SETTING_ID,
            "variant": "bottom_half_crop_expert_standalone",
            "cv_mae": crop_mae,
            "delta_vs_current_submission_cv": crop_mae - CURRENT_BASELINE_CV,
            "current_submission_baseline": CURRENT_BASELINE_NAME,
            "current_submission_baseline_cv": CURRENT_BASELINE_CV,
            "current_submission_baseline_kaggle": CURRENT_BASELINE_KAGGLE,
            "previous_historical_baseline": PREVIOUS_BASELINE_NAME,
            "previous_historical_baseline_oof": PREVIOUS_BASELINE_OOF,
            "preproc": FINAL_PREPROC.short_tag(),
            "active_groups": "|".join(active_groups),
            "base_models": f"bottom_half::{BOTTOM_CROP_SPEC.short_tag()}",
            "blend_components": "",
            "blend_weights": "",
            "signals_used_for_routing": "none",
            "route_weight": 0.0,
            "mean_route_fraction": 0.0,
            "fold_router_thresholds": "",
            **{f"bin_{label}": score for label, score in crop_bins.items()},
        },
        {
            "setting_id": SETTING_ID,
            "variant": "no_scene_diagnostic_proxies_router",
            "cv_mae": candidate_mae,
            "delta_vs_current_submission_cv": candidate_mae - CURRENT_BASELINE_CV,
            "current_submission_baseline": CURRENT_BASELINE_NAME,
            "current_submission_baseline_cv": CURRENT_BASELINE_CV,
            "current_submission_baseline_kaggle": CURRENT_BASELINE_KAGGLE,
            "previous_historical_baseline": PREVIOUS_BASELINE_NAME,
            "previous_historical_baseline_oof": PREVIOUS_BASELINE_OOF,
            "preproc": FINAL_PREPROC.short_tag(),
            "active_groups": "|".join(active_groups),
            "base_models": "|".join([*base_oof_dict.keys(), f"bottom_half::{BOTTOM_CROP_SPEC.short_tag()}"]),
            "blend_components": "|".join(blend_names),
            "blend_weights": "|".join(f"{w:.6f}" for w in blend_weights),
            "signals_used_for_routing": "|".join(signal_names),
            "route_weight": ROUTE_WEIGHT,
            "mean_route_fraction": float(np.mean([row["val_route_fraction"] for row in fold_rows])),
            "fold_router_thresholds": "|".join(f"{row['threshold']:.2f}" for row in fold_rows),
            "fold_route_fractions": "|".join(f"{row['val_route_fraction']:.3f}" for row in fold_rows),
            **{f"bin_{label}": score for label, score in candidate_bins.items()},
        },
    ]

    if args.write_submission:
        print("\nFitting full-train models and writing candidate submission...")
        baseline_test, test_base_dict = build_full_test_predictions(driver, active_groups, blend_names, blend_weights)
        bottom_crop_test = train_bottom_crop_full_and_predict(driver, active_groups)
        test_signals, _ = build_router_signals(baseline_test, test_base_dict, bottom_crop_test)
        final_test, test_threshold, test_route_fraction = train_full_router_and_predict_test(
            y,
            baseline_oof,
            bottom_crop_oof,
            train_signals,
            baseline_test,
            bottom_crop_test,
            test_signals,
        )
        save_results(final_test, output_path=Path(args.submission_path))
        rows.append(
            {
                "setting_id": SETTING_ID,
                "variant": "submission_test_prediction",
                "cv_mae": candidate_mae,
                "delta_vs_current_submission_cv": candidate_mae - CURRENT_BASELINE_CV,
                "current_submission_baseline": CURRENT_BASELINE_NAME,
                "current_submission_baseline_cv": CURRENT_BASELINE_CV,
                "current_submission_baseline_kaggle": CURRENT_BASELINE_KAGGLE,
                "previous_historical_baseline": PREVIOUS_BASELINE_NAME,
                "previous_historical_baseline_oof": PREVIOUS_BASELINE_OOF,
                "preproc": FINAL_PREPROC.short_tag(),
                "active_groups": "|".join(active_groups),
                "base_models": "|".join([*test_base_dict.keys(), f"bottom_half::{BOTTOM_CROP_SPEC.short_tag()}"]),
                "blend_components": "|".join(blend_names),
                "blend_weights": "|".join(f"{w:.6f}" for w in blend_weights),
                "signals_used_for_routing": "|".join(signal_names),
                "route_weight": ROUTE_WEIGHT,
                "full_train_router_threshold": test_threshold,
                "test_route_fraction": test_route_fraction,
                "submission_path": str(Path(args.submission_path)),
                **{f"bin_{label}": score for label, score in candidate_bins.items()},
            }
        )
        print(f"Saved submission to {Path(args.submission_path).resolve()}")
        print(f"Full-train router threshold: {test_threshold:.2f}")
        print(f"Test route fraction: {test_route_fraction:.3f}")

    write_summary(summary_path, rows)


if __name__ == "__main__":
    main()
