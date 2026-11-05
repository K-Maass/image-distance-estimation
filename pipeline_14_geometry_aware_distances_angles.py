from __future__ import annotations

"""Current best strict-compliant public pipeline.

This branch adds geometry-aware residual corrections to the promoted
local-competence baseline. It improved the confirmed public Kaggle score to
0.0951, mainly by helping low/high edge behavior. OOF diagnostics still show a
middle-bin tradeoff, so future variants should focus on a tighter mid-bin guard
rather than adding broader corrections.
"""

from pathlib import Path

import numpy as np

from full_pipeline_eval_core import PipelineResult, apply_correction, build_baseline_context, evaluate_pipeline_result, standard_meta_features
from structural_strategy_common import (
    append_meta_group,
    default_uncertain_zones,
    geometry_aware_parts,
    hgb_residual_correction_with_importance,
    load_or_build_feature_matrix,
    summarize_feature_importance,
)


SETTING_ID = "full_pipeline_14_geometry_aware_distances_angles"
PUBLIC_KAGGLE_MAE = 0.0951


def run(context=None):
    context = build_baseline_context(strict_reproduction=False) if context is None else context
    geom_train, geom_slices = load_or_build_feature_matrix("geometry_aware_distances_angles", "train", geometry_aware_parts)
    geom_test, _ = load_or_build_feature_matrix("geometry_aware_distances_angles", "test", geometry_aware_parts)
    meta_train, _ = standard_meta_features(context, "train")
    meta_test, _ = standard_meta_features(context, "test")
    X_train, X_test, group_slices = append_meta_group(geom_train, geom_test, geom_slices, meta_train, meta_test)
    train_zone, test_zone = default_uncertain_zones(context)
    train_zone = train_zone | (np.abs(context.promoted_oof - 1.0) < 0.40) | (np.abs(context.promoted_oof - 2.0) < 0.32)
    test_zone = test_zone | (np.abs(context.promoted_test - 1.0) < 0.40) | (np.abs(context.promoted_test - 2.0) < 0.32)

    oof_corr, test_corr, diagnostics = hgb_residual_correction_with_importance(
        context,
        X_train,
        X_test,
        group_slices,
        train_zone,
        test_zone,
        seed_base=1400,
        cap=0.040,
        shrink=0.42,
        min_abs_raw=0.011,
        low_weight=1.1,
    )
    diagnostics = summarize_feature_importance(SETTING_ID, diagnostics)
    return evaluate_pipeline_result(
        context,
        PipelineResult(
            setting_id=SETTING_ID,
            script_path=str(Path(__file__).resolve()),
            oof_pred=apply_correction(context, context.promoted_oof, oof_corr),
            test_pred=apply_correction(context, context.promoted_test, test_corr),
            route_mask=np.abs(oof_corr) > 1e-7,
            test_route_mask=np.abs(test_corr) > 1e-7,
            diagnostics=diagnostics,
            metadata={
                "idea": "geometry_aware_residual_with_distances_angles_perspective",
                "features": "spatial_layout|distance_features|angle_features|perspective_cues",
            },
        ),
    )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
