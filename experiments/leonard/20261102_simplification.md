# Simplification experiment: fixed 2×2 comparison

Run: `20261102_122900_385150_simplification`. Owner: Leonard.
Reproduce: `.venv/bin/python notebooks/leonard/run_simplification_experiment.py`.

Prespecified four candidates, 20 fits. Original HGB: 15 leaves, min leaf 40,
L2=5. Simpler HGB: 7 leaves, min leaf 80, L2=10. Both use 200 iterations,
learning rate .06, no early stopping, seed 42. Three capacity constraints change
together, so their individual contributions cannot be isolated.

Full: 289 features, previous HGB + boost-threat columns. Compact: 45 features—
21 roster summaries, two faint counts, four final/late-mean HP summaries,
18 current/recent HP-weighted boost-threat values (including differences).
Static stats, detailed status/effects and all-time bench boost summaries removed.
Recent bench boosts may still reset after switches; team HP remains estimated.

Same five grouped folds and 7,996 development rows as notebook 06. Previously
opened holdout (2,000) plus its four development duplicates stay excluded.
Feature cache, raw-data and split hashes verified. All preprocessing fits inside
training folds. No holdout/test evaluation, adaptive tuning or new submission.

| Candidate | Features | CV accuracy | Delta vs reference |
|---|---:|---:|---:|
| Full / original | 289 | 67.234% | — |
| Full / simpler | 289 | 66.834% | -0.400 pp |
| Compact / original | 45 | 65.708% | -1.526 pp |
| Compact / simpler | 45 | 65.933% | -1.301 pp |

Retain the full HGB + boost-threat reference. Compact candidates lose in all
five folds. Simpler HGB helps within the compact set (+0.225 pp, four folds), but
remains below the reference. Full/simpler improves three folds but larger declines
in two folds reduce its mean. Reject these specific simplifications, not the
possibility that other simplifications could help.

Eight existing tests passed. Independent artifact checks confirm identical folds,
zero duplicate-group overlap, complete OOF coverage, valid predictions and
recalculated accuracies. Baseline predictions/probabilities reproduce the prior
run (column order aligned for artifact comparison). Review notebook executed
against saved results. Fold-training models are not retained.

Development data has been explored repeatedly; results remain exploratory,
not an independent performance confirmation. Only exact duplicates are grouped.

Artifacts: `outputs/leonard/20261102_122900_385150_simplification/` contains
settings/feature columns, versions/commit/dirty state/hashes, source snapshots,
fold scores, paired deltas, OOF probabilities, splits, validation and review notebook.
