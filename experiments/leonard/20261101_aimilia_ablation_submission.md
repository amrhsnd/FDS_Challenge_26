# Aimilia ablation: bounded feature/model comparison and submission

Owner: Leonard. Run: `20261101_230041_643033_aimilia_feature_ablation`.

Reproduce from the repository root with:

```bash
.venv/bin/python notebooks/leonard/run_ablation.py
```

The same cells can be run in `notebooks/leonard/05_aimilia_feature_ablation.ipynb`.
Each rerun creates a fresh timestamped artifact folder. Do not use the same holdout
for further tuning; this run already selects its final procedure before opening it.

The original nine ablation candidates are retained. Eight additional candidates
compare logistic C=0.1/10, omitted effect features, last-seen roster statuses and
recent same-species HP changes, two histogram-boosting complexities and a regularized
random forest. All use the original five development folds (8,000 battles); the
2,000-row holdout is opened only for the CV winner. Imputation/scaling/constant
filtering fit inside folds. Final submission fitting uses all 10,000 labels.

The original all-106 baseline reproduced at 0.663 CV accuracy. Roster and boost
removal reduced performance; no group deletion was adopted on the original
logistic model. Added features are hypotheses, not assumed improvements. Model
selection uses mean accuracy and stable candidate order for exact ties.

An exact-observation audit found 14 repeated training observations with consistent
labels: four duplicate groups cross development/holdout and seven cross development
folds. Original assignments were retained for comparison with earlier ablation.
This small overlap can make scores optimistic; related nonidentical battles were
not audited. Future work should use duplicate-grouped partitions agreed by the
team, with a new genuinely unused evaluation set.

Artifacts under `outputs/leonard/20261101_230041_643033_aimilia_feature_ablation/`
include exact candidate parameters, per-fold scores, paired deltas, splits and
input hashes, environment versions, source snapshots, holdout metrics, final model,
and the validated submission. The commit and dirty-tree status are in
`experiment.json`; notebook/module snapshots preserve the uncommitted experiment.

Validation: all four existing unittest checks passed. Extension checks verified
finite values, independence of IDs and labels, and exclusion of turns after 30.
The sequential Python runner was used because notebook schema dependencies were
slow to hydrate from OneDrive; it executes the actual notebook code in order and
captures textual outputs. The saved ablation plot is generated with the Agg backend.

Final outcome: `hgb_extended` selected from 17 candidates. Mean development CV
accuracy 0.66475 (baseline 0.66300; +0.175 percentage points), fold scores
0.66250 / 0.676875 / 0.66750 / 0.678125 / 0.63875. Fold standard deviation
0.0142445. Single holdout accuracy: 0.67750 (1,355 / 2,000). The CV gain is small
and does not establish a robust improvement or guarantee a higher Kaggle score.

Final fit: all 10,000 labels, 217 numeric features, 200 histogram-boosting iterations,
learning rate 0.06, 15 leaves, minimum leaf size 40, L2 regularization 5, no early
stopping, seed 42. Submission: `submission.csv`, 5,000 rows, 2,535 ones / 2,465 zeros.
CSV round-trip, sample row order, unique IDs, binary values and absence of missing
values all passed. No upload was performed.
