# Experiment Summary

This is a curated record of the original distance-prediction experiments.
All numbers below are historical reports, not results rerun during repository
cleanup. [Results and limitations](docs/RESULTS.md) preserves the main comparisons.

## What was selected

The retained pipeline uses RGB images, downsampling by three, per-image
normalization, and handcrafted color, projection, occupancy, orientation,
and spatial features. Two neighbor-based regressors dominate a five-model
blend; gradient-boosted trees and ExtraTrees provide smaller contributions.

A local-competence router selectively mixes in bottom-half crop predictions
using prediction disagreement and crop deltas. Pipeline 14 adds a capped,
shrunk residual correction from geometry features and existing model signals.
It was selected on its reported public Kaggle MAE of `0.0951`.
This makes it the best reported public submission among these experiments,
not evidence that it has the best generalization error.

## Ablation findings

The original workflow screened features, preprocessing, model families, and
low-distance errors before exploring corrections. These conclusions describe
that development setup and inherit its evaluation limitations.

| Experiment | Recorded observation | Decision at the time |
|---|---|---|
| Preprocessing | RGB, downsample three, normalization, and combined features were strongest in the retained setup | Use `rgb_ds3_norm1_warp0_combined` |
| Neighbor models | Neighbor-ridge about `0.1109` OOF MAE; local-ridge about `0.1175` | Give these most blend weight |
| Global isotonic calibration | `0.102031` versus blend `0.100969` | Reject |
| Bottom-half specialist | Standalone `0.120521`; direct blend hurt low-bin MAE | Use crop predictions as router inputs |
| Router ablation | Removing scene proxies gave `0.100420`; removing disagreement gave `0.100660` | Keep crop and disagreement signals |
| Route-weight robustness | Weight `0.14` gave `0.100255`, public score still `0.0963` | Retain original router |
| Patch dictionary | Standalone `0.180295`; selected blend weight zero | Drop |
| Adversarial validation | Logistic/HGB ROC-AUC `0.508896` / `0.485717` | No useful shift signal in tested features |
| Regime residual correction | `0.100613`; low/high gains lost to middle-bin damage | Drop |
| Retrieval residual correction | `0.100825`; small edge gains, middle-bin regression | Drop |

Historical feature screening also rejected coarse PCA, spectrum summaries,
and detrended gradient-magnitude blocks. Task-warp groups were inactive in the
selected warp-off configuration, even though earlier screens favored them.

## Structural follow-ups and failed combinations

The confidence-mixture comparison (pipeline 10) reported public MAE `0.0958`.
Retrieval as a meta-signal and density/ambiguity correction reported lower
OOF MAEs (`0.098873` and `0.098579`) but public scores `0.0960` and `0.0961`.
Two heavier merge probes scored `0.0962` and `0.0965` publicly; neither
improved on pipeline 10. OOF ranking and public ranking did not agree.

Local keypoint descriptors reached `0.099659` OOF MAE but worsened the middle
bin. Broad multiscale RGB features (`0.099998`) and residual prototype
clusters (`0.100448`) offered no convincing overall improvement. These
screening implementations are not included in the curated public source.

Pipeline 14 improved reported low/high errors versus pipeline 10, while
worsening the middle bin. Its permutation analysis favored existing model
signals; individual geometry groups did not show positive importance.
The records support a useful combined correction, not a causal claim that
any particular geometry feature explains the public improvement.

## Evaluation lesson

Base models used fixed stratified folds, but downstream routers and residual
models reused global OOF predictions. Their evaluation was not fully nested:
training meta-features can depend on labels from an outer validation fold.
Repeated development and public-score selection introduce further selection
bias. An untouched holdout or fully nested evaluation is needed before
claiming an unbiased improvement; adding more correction families would not
resolve that uncertainty.
