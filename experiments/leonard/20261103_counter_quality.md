# Counter quality and redundancy beyond the 316-input matchup model

Owner: Leonard. Run `20261103_140058_542958_counter_quality`.
Reproduce: `.venv/bin/python notebooks/leonard/run_counter_experiment.py`.
Same optional local runtime and fixed prior caches as earlier experiments.
All classifiers are LR: C=.01, L2, lbfgs, max_iter=4000, random_state=42.

Four prespecified candidates: unchanged 316-input procedure; +8 redundancy inputs;
+8 HP-quality inputs; +both16. No parameter search or model-family changes.
Existing procedure: fold-fitted RFE300 + 16 team-matchup inputs. All original
316 selected inputs are verified unchanged in every variant/fold. New inputs
use training-fold species metadata, median imputation, constant removal and
standard scaling, then a joint logistic regression. Knowledge explicitly supplied
in each input battle can be overlaid, without modifying the fitted lookup.

Five same duplicate-grouped development folds, 7,996 battles. Previous 2,000-row
holdout and its four development duplicates excluded. Baseline labels AND
probabilities reproduce exactly. Twenty outer candidate/fold fits plus internal
RFE fits. No extra holdout scores, test labels, boosting or Kaggle upload.

A favourable counter has own-type best effectiveness >1. A safe counter is also
resistant to all enemy own-type options (best enemy effectiveness <1). Health
threshold fixed at .5 before evaluation. These are type-based proxies, not actual
moveset/damage/switch guarantees. Only revealed alive enemies with known types
are evaluated; unknown enemies produce missing values, not invented weaknesses.
Own unseen members start at HP=1; last HP can be stale. Quality metrics do not
adjust for sleep, paralysis, boost, speed, attack power or switch constraints.

| Candidate | Inputs | Mean CV accuracy | Delta vs current |
|---|---:|---:|---:|
| Current baseline | 316 | 67.259% | — |
| + redundancy | 324 | 67.284% | +0.025 pp |
| + HP quality | 324 | 67.396% | +0.138 pp |
| + both | 332 | 67.284% | +0.025 pp |

Select HP quality only: four folds improve, one declines. See paired_comparison.csv
for exact corrected/introduced error counts; net 11 extra correct / 7,996. The
new eight inputs measure HP of best counters, weakest covered position, healthy
safe-counter coverage, total counter HP reserves and HP-weighted uncovered enemies.
Redundancy alone changes little and its gain does not add to the quality gain.

This is a small exploratory result after repeated feature work, not independent
performance confirmation or a promise of a higher Kaggle score. Correlated derived
inputs can also alter effective regularization. Exact duplicate groups only;
related nonidentical battles may remain. Future CV must refit every selection/
preprocessing step inside folds, not use the full-data selected list.

Winning procedure refitted on all 10,000 labels: 324 inputs. It preserves the
previous full-data 316-input subset and appends the eight quality inputs.
Generated a separate 5,000-row submission. Labels, probability range, sample ID
order, unique IDs and missing values checked. Reloaded model reproduces all
5,000 predictions/probabilities; independent raw reconstruction reproduced 12.
29 tests pass, including eight new counter tests covering redundancy, fragile
vs healthy, faints, uncovered vs unknown, HP weighting, order/horizon/target
independence and fold-local metadata/schema.

Artifacts contain cv_results.csv, paired_comparison.csv, selected per-fold names,
coverage audit, final 324-feature/coefficient list, definitions for all 16 new inputs
(with a final-model flag), hashes/settings/versions, sources/tests, validation,
executed notebook, final model and checked submission.
