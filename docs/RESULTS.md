# Results and Evidence

These are reported historical results from the original project records.
The cleanup did not rerun training or verify scores against Kaggle.
MAE is mean absolute error; lower is better. OOF denotes development
out-of-fold predictions and is subject to the limitations below.

## Main comparison

| Method | Reported development OOF MAE | Reported public Kaggle MAE |
|---|---:|---:|
| Evidence-selected blend | 0.100969 | 0.0966 |
| Local-competence router | 0.100420 | 0.0963 |
| Pipeline 10: confidence mixture (historical comparison) | 0.099990 | 0.0958 |
| Pipeline 14: geometry-aware residual correction (retained) | 0.099819 | 0.0951 |

Pipeline 14 was selected by public score. Its reported public MAE is `0.0007`
lower than pipeline 10, approximately `0.73%` relative. No private leaderboard
score, rank, uncertainty interval, or statistical significance is established
by the retained evidence.

## Error by distance bin

| Method | Low `[0.25,1.00)` | Middle `[1.00,2.00)` | High `[2.00,2.78)` |
|---|---:|---:|---:|
| Evidence-selected blend | 0.221387 | 0.090429 | 0.080562 |
| Local-competence router | 0.220153 | 0.089786 | 0.080637 |
| Pipeline 10 | 0.219384 | 0.090064 | 0.078027 |
| Pipeline 14 | 0.212996 | 0.090770 | 0.077816 |

The high-bin label follows the original records; the implementation includes
the upper endpoint `2.78`. Pipeline 14 improved low/high MAE versus pipeline
10 while worsening middle-bin MAE. The original diagnosis counted `289`
training examples below distance `1.0`, so low-bin estimates had less support.

## What the geometry diagnostics support

Historical mean change in validation MAE after permuting a feature group:

| Pipeline 14 feature group | Permutation delta MAE |
|---|---:|
| Existing pipeline meta-features | +0.001348 |
| Perspective cues | -0.000031 |
| Distance features | -0.000032 |
| Angle features | -0.000044 |
| Spatial layout | -0.000090 |

A positive delta means permutation worsened MAE. Existing model signals
carried the clearest signal in this diagnostic. Negative values for individual
geometry groups do not establish their independent predictive value. The
public-score improvement alone does not establish why the correction worked.

## Provenance and limits

The tables were transcribed from the repository's original
`EXPERIMENT_SUMMARY.md` and the historical `structural_strategy_analysis.md`
record dated 2026-04-22. They preserve aggregate observations; original
numerical CSVs, OOF arrays, leaderboard exports, and submission identifiers
are not included in this public snapshot. The abbreviated
[experiment summary](../EXPERIMENT_SUMMARY.md) records the main ablations.

Base-model preprocessing and fitting used fixed stratified folds. However,
the router and residual layers reused global base/promoted OOF predictions
across those folds. For an outer validation fold, training meta-features can
come from models trained using that fold's labels. This is not fully nested
validation of the complete stack. Blend/variant selection on the same
OOF records and repeated public-score comparisons add selection bias.

Treat these OOF values as development diagnostics. The reported public score
is a separate historical observation, but public selection does not provide
an untouched final test. Neither source establishes unbiased generalization
or production readiness. A new untouched holdout or correctly nested run,
with preserved predictions and provenance, is required for stronger claims.
