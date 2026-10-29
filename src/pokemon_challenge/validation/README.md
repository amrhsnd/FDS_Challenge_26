# Validation agreement

Use `configs/validation.json` as the initial agreement: Accuracy, five stratified
folds, shuffle enabled, seed 42. These settings alone do not guarantee the same folds:
all experiments must use the same ordered training battle IDs and labels, or reuse
an agreed fold assignment keyed by training battle ID. Log a hash of that assignment
for final candidates. Do not use battle IDs as model features.

Split at battle level, not turn level. Fit imputers, scalers, selectors, and other
learned preprocessing inside each fold using a pipeline. Inspect related/duplicate
battles before deciding whether grouping is needed. Do not use the test set or
leaderboard to tune features or model settings. Change the shared split strategy
only with team agreement and re-evaluate comparisons after a change.

Implement shared splitting/scoring modules here when modelling starts. This folder
and config establish a convention; no evaluation runner is supplied yet.
