# Reproduction and evaluation boundaries

Run from the repository root on Python 3.12. Install `requirements.txt` in
a virtual environment and use `python main.py` for the selected pipeline.
`requirements-lock.txt` records exact resolved versions from the Python 3.12
Linux validation environment; install with
`python -m pip install -r requirements.txt -c requirements-lock.txt`.
It is a version snapshot, not a hash-verified cross-platform lock. Record Python version,
`python -m pip freeze`, Git commit, hardware, dataset version, and commands with
new measurements. The active pipeline uses repository-relative `data/` directly;
its experiment settings are defined in code rather than a configuration file.

## Cache discipline

Current `artifacts/` cache keys do not fingerprint all dataset contents/order,
source code, package versions, and fold definitions. Do not reuse caches after
any of these changes. Move generated artifacts outside the checkout to preserve
them, then regenerate in an empty cache.
Never download another person's cache as verified benchmark evidence.

Pipeline 14 uses `strict_reproduction=False` when reconstructing its baseline,
so exact agreement with historical numbers is not enforced. Keep new results
distinct from original measurements.

## Evaluation caveat

`build_baseline_context` in `full_pipeline_eval_core.py` constructs global base
OOF predictions, then trains the router using these predictions and the same
folds. A meta-training row's base prediction can have been trained using labels
from the current outer validation fold. The residual stage in
`structural_strategy_common.py` similarly reuses promoted OOF predictions.
Base-level fold preprocessing does not isolate the entire stack.

Before claiming an unbiased complete-stack score, hold out new data before all
feature/model/threshold selection, or generate inner OOF predictions and fit all
stacked components only within each outer training partition. Report overall
and distance-bin MAE; do not hide the middle-bin regression. Historical public
leaderboard tuning also requires a fresh independent evaluation.

## Submission contract

The current writer generates IDs `000`, `001`, … and the test loader sorts PNGs
lexicographically. This assumes contiguous zero-padded competition IDs. Verify
the generated CSV against the official sample submission before uploading;
arbitrary filenames, gaps, or another dataset are not supported.

## Validation scope

CI checks synthetic feature/fold/routing/I/O contracts and imports the active
entry point. It does not run full fitting, establish a leaderboard score,
validate generalization, or measure runtime/memory. Exact benchmark artifacts
and a public leaderboard reference remain absent. A future independent run
should publish compact aggregate metrics and configuration, not dataset
examples or per-example labels/predictions.
