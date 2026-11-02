# Three controlled feature experiments

Run: `20261102_121822_824155_progressive_features`. Owner: Leonard.

Reproduce: `.venv/bin/python notebooks/leonard/run_progressive_experiments.py`.
Review: `notebooks/leonard/06_progressive_feature_experiments.ipynb`.

Each of three observation-only groups is added independently to LR106 and HGB217.
LR: C=1, max_iter=4000, lbfgs. HGB: 200 iterations, learning_rate=.06,
15 leaves, min_samples_leaf=40, L2=5, no early stopping. Seed 42.
No parameter tuning, feature combinations, holdout/test evaluation or submission generation.

StratifiedGroupKFold, five folds, 7,996 development battles. SHA-256 of canonical
observations excluding ID/target defines groups. Previously opened 2,000-row
holdout and four development duplicates of it are quarantined. No exact duplicate
group crosses folds or partitions. Models/imputation/scaling fit only on fold
training rows. Baselines rerun on these new folds; old scores are not directly comparable.

| Addition | LR accuracy | LR delta pp | HGB accuracy | HGB delta pp |
|---|---:|---:|---:|---:|
| Baseline | 66.321% | — | 66.821% | — |
| Remaining strength | 65.995% | -0.325 | 66.846% | +0.025 |
| Advantage trajectory | 66.396% | +0.075 | 66.646% | -0.175 |
| Boost threat | 66.283% | -0.037 | 67.234% | +0.413 |

Recommend HGB + boost threat for further investigation: improved 4/5 folds,
292 corrected vs 259 introduced errors (net 33/7,996). Do not automatically
combine the other groups. Small changes and development reuse limit certainty;
these results do not supply a new independent generalization estimate.
Exact duplication only is controlled; related nonidentical battles may remain.
Last-seen roster information is approximate, including boosts that may reset on switches.
Unknown opponent species never receive invented base stats.

Validation: 8 unittests passed (4 new synthetic feature tests); 4 feature tests
rerun after removing duplicate trajectory inputs. Independent verification of all
8 OOF files, finite features, duplicate-group separation, complete coverage,
probability ranges and matching training-data hash passed. Review notebook cells
executed successfully against saved artifacts.

Artifacts: `outputs/leonard/20261102_121822_824155_progressive_features/`.
`experiment.json` records code commit/dirty status, versions, hashes and settings;
`candidates.json` records exact feature columns. Source snapshots, all 40 fold
scores, paired changes, OOF probabilities, feature matrix, splits, review notebook
and `validation_report.json` are retained locally. Training fold models are not saved.
