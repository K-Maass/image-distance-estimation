from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

from experiment_driver import (
    ExperimentDriver,
    ModelSpec,
    PreprocConfig,
    clip_predictions,
    get_cv_splits,
    make_model,
    per_bin_mae,
    save_results,
)

FINAL_PREPROC = PreprocConfig(
    load_rgb=True,
    downsample=3,
    normalize=True,
    use_warp=False,
    view_family="combined",
)
FINAL_GROUPS = [
    "dark_run_continuity",
    "dominant_orientation_concentration",
    "gradient_stats",
    "gray_global_stats",
    "gray_region_stats",
    "horizon_band_transition",
    "multiview_deltas",
    "orientation_histograms",
    "percentile_spread_ladders",
    "pooled_gray_grids",
    "projection_profiles",
    "quadrant_imbalance",
    "rgb_summary",
    "rowcol_entropy",
    "threshold_mass",
    "warped_view_groups",
]
BASE_MODELS = [
    ModelSpec("neighbor_ridge", (("neighbor_count", 40), ("pca_dim", 180))),
    ModelSpec("local_ridge", (("alpha", 2.0), ("neighbor_count", 56), ("pca_dim", 220))),
    ModelSpec("hgb", (("learning_rate", 0.03), ("max_leaf_nodes", 63), ("min_samples_leaf", 8))),
    ModelSpec("hgb_log", (("learning_rate", 0.03), ("max_leaf_nodes", 63), ("min_samples_leaf", 8))),
    ModelSpec("et", (("max_features", 0.35), ("min_samples_leaf", 2))),
]
SUMMARY_PATH = Path("artifacts") / "final_pipeline_summary.csv"
QUICK_SUMMARY_PATH = Path("artifacts") / "final_pipeline_summary_quick.csv"
DEFAULT_SUBMISSION_PATH = Path("prediction_final_evidence_pipeline.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final evidence-selected sklearn-only blended pipeline")
    parser.add_argument("--quick", action="store_true", help="Use the experiment driver's lighter CV setting for a smoke test")
    parser.add_argument("--write-submission", action="store_true", help="Fit on the full train set and write a submission CSV")
    parser.add_argument(
        "--submission-path",
        default=str(DEFAULT_SUBMISSION_PATH),
        help="Where to write the submission CSV when --write-submission is set",
    )
    return parser.parse_args()


def active_final_groups(driver: ExperimentDriver) -> list[str]:
    return driver.feature_repo.supported_groups(FINAL_PREPROC, FINAL_GROUPS)


def weighted_sum_from_dict(pred_dict: dict[str, np.ndarray], names: list[str], weights: np.ndarray) -> np.ndarray:
    out = None
    for name, weight in zip(names, weights):
        term = float(weight) * np.asarray(pred_dict[name], dtype=np.float32)
        out = term if out is None else out + term
    return np.asarray(out, dtype=np.float32)


def search_blend_weights(
    pred_dict: dict[str, np.ndarray],
    y: np.ndarray,
    n_random: int,
    seed: int,
) -> tuple[list[str], np.ndarray, float]:
    names = list(pred_dict.keys())
    P = np.column_stack([pred_dict[name] for name in names]).astype(np.float32)
    rng = np.random.default_rng(seed)
    best_mae = float("inf")
    best_w = None
    candidates = [np.eye(P.shape[1], dtype=np.float32)[i] for i in range(P.shape[1])]
    candidates.append(np.ones(P.shape[1], dtype=np.float32) / P.shape[1])
    for _ in range(n_random):
        candidates.append(rng.dirichlet(np.ones(P.shape[1])).astype(np.float32))
    for weights in candidates:
        pred = P @ weights
        mae = float(mean_absolute_error(y, pred))
        if mae < best_mae:
            best_mae = mae
            best_w = weights
    return names, np.asarray(best_w, dtype=np.float32), float(best_mae)


def fit_isotonic(train_pred: np.ndarray, y_train: np.ndarray, query_pred: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(
        y_min=float(np.min(y_train)),
        y_max=float(np.max(y_train)),
        out_of_bounds="clip",
    )
    model.fit(train_pred.astype(np.float32), y_train.astype(np.float32))
    return model.predict(query_pred.astype(np.float32)).astype(np.float32)


def cross_fit_global_isotonic(base_oof: np.ndarray, y: np.ndarray, n_splits: int) -> np.ndarray:
    splits = get_cv_splits(y, n_splits=n_splits)
    oof = np.zeros(len(y), dtype=np.float32)
    for tr_idx, va_idx in splits:
        oof[va_idx] = fit_isotonic(base_oof[tr_idx], y[tr_idx], base_oof[va_idx])
    return oof.astype(np.float32)


def train_full_and_predict(driver: ExperimentDriver, model_spec: ModelSpec, active_groups: list[str]) -> np.ndarray:
    train_indices = np.arange(len(driver.y))
    X_train, _, artifacts = driver.feature_repo.assemble_fold_features(
        "train",
        FINAL_PREPROC,
        active_groups,
        train_indices,
        train_indices,
    )
    test_count = len(driver.data_repo.load_image_bank("test", FINAL_PREPROC.load_rgb, FINAL_PREPROC.downsample))
    test_indices = np.arange(test_count)
    X_test, _, _ = driver.feature_repo.assemble_fold_features(
        "test",
        FINAL_PREPROC,
        active_groups,
        train_indices,
        test_indices,
        fit_train_artifacts=artifacts,
    )
    model = make_model(model_spec, quick=driver.quick)
    model.fit(X_train, driver.y)
    return clip_predictions(model.predict(X_test), driver.y)


def format_per_bin(per_bin: dict[str, float]) -> str:
    return "  ".join(f"{label}={value:.6f}" for label, value in per_bin.items())


def write_summary(summary_path: Path, row: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    print(f"Saved {summary_path.resolve()}")


def main() -> None:
    args = parse_args()
    driver = ExperimentDriver(quick=args.quick)
    y = driver.y
    active_groups = active_final_groups(driver)

    base_results = []
    oof_dict: dict[str, np.ndarray] = {}
    print("Evaluating evidence-supported base models...")
    for spec in BASE_MODELS:
        result = driver.evaluate_model_cv(
            spec,
            FINAL_PREPROC,
            active_groups,
            n_splits=driver.full_splits,
            compute_importance=False,
        )
        base_results.append(result)
        oof_dict[spec.short_tag()] = result.oof_pred.astype(np.float32)
        print(f"  {spec.short_tag():<72} CV={result.overall_mae:.6f}")

    names, weights, raw_blend_mae = search_blend_weights(
        oof_dict,
        y,
        n_random=4000 if args.quick else 20000,
        seed=97,
    )
    raw_blend_oof = weighted_sum_from_dict(oof_dict, names, weights)
    raw_blend_mae = float(mean_absolute_error(y, raw_blend_oof))
    raw_per_bin = per_bin_mae(y, raw_blend_oof)

    calibrated_oof = cross_fit_global_isotonic(raw_blend_oof, y, n_splits=driver.full_splits)
    calibrated_mae = float(mean_absolute_error(y, calibrated_oof))
    calibrated_per_bin = per_bin_mae(y, calibrated_oof)

    use_calibration = calibrated_mae < raw_blend_mae
    final_variant = "global_isotonic" if use_calibration else "raw_blend"
    final_oof = calibrated_oof if use_calibration else raw_blend_oof
    final_mae = calibrated_mae if use_calibration else raw_blend_mae
    final_per_bin = calibrated_per_bin if use_calibration else raw_per_bin

    print("\nFinal Evidence Pipeline")
    print(f"Preprocessing: {FINAL_PREPROC.short_tag()}")
    print(f"Active feature groups: {', '.join(active_groups)}")
    print(f"Base models: {', '.join(spec.short_tag() for spec in BASE_MODELS)}")
    print("Blend weights:")
    for name, weight in zip(names, weights):
        print(f"  {name}: {weight:.4f}")
    print(f"Raw blend CV MAE: {raw_blend_mae:.6f}")
    print(f"Raw blend per-bin: {format_per_bin(raw_per_bin)}")
    print(f"Global isotonic CV MAE: {calibrated_mae:.6f}")
    print(f"Global isotonic per-bin: {format_per_bin(calibrated_per_bin)}")
    print(f"Selected final variant: {final_variant}")
    print(f"Final CV MAE: {final_mae:.6f}")
    print(f"Final per-bin: {format_per_bin(final_per_bin)}")

    summary_path = QUICK_SUMMARY_PATH if args.quick else SUMMARY_PATH
    summary_row = {
        "final_preproc": FINAL_PREPROC.short_tag(),
        "final_model_family": "blend",
        "final_model_tag": final_variant,
        "selected_groups": "|".join(active_groups),
        "retained_phase2_groups": "|".join(FINAL_GROUPS),
        "inactive_retained_groups": "|".join(group for group in FINAL_GROUPS if group not in active_groups),
        "base_models": "|".join(spec.short_tag() for spec in BASE_MODELS),
        "blend_components": "|".join(names),
        "blend_weights": "|".join(f"{w:.6f}" for w in weights),
        "raw_blend_cv_mae": raw_blend_mae,
        "global_isotonic_cv_mae": calibrated_mae,
        "cv_mae": final_mae,
    }
    summary_row.update({f"bin_{label}": score for label, score in final_per_bin.items()})
    write_summary(summary_path, summary_row)

    if args.write_submission:
        test_pred_dict = {spec.short_tag(): train_full_and_predict(driver, spec, active_groups) for spec in BASE_MODELS}
        raw_blend_test = weighted_sum_from_dict(test_pred_dict, names, weights)
        if use_calibration:
            final_test = fit_isotonic(raw_blend_oof, y, raw_blend_test)
        else:
            final_test = raw_blend_test
        save_results(final_test, output_path=Path(args.submission_path))
        print(f"Saved submission to {Path(args.submission_path).resolve()}")


if __name__ == "__main__":
    main()
