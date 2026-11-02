# Classical models with Pokémon and move identities

Owner: Leonard. Initial run: `20261102_123710_282289_classical_identity`.
User constraint: only simple classifiers. This run trains logistic regression
and random forest only. CatBoost was installed before the clarification but
was not trained or used; no boosting or ensemble is used in this run.

Reproduce: `.venv/bin/python notebooks/leonard/run_classical_identity_experiment.py`.
Four fixed candidates: LR / RF, each with 289 numeric baseline features or
additional species/move identity. LR C=.1, lbfgs, max_iter=4000, StandardScaler.
RF: 300 trees, min leaf 2, sqrt features, unlimited depth, two threads. Seed 42.

Added features: unordered own species membership and last HP, observed species
and exposure/HP, move rates, five-turn move rates, recency, move-type rates,
active species, opponent lead, last and most-used moves. Vocabulary/one-hot
encoding fit inside each training fold; novel validation/test values are ignored.
About 4,250 nonconstant features per training fold with identities.

Five unchanged grouped folds on 7,996 development battles. Prior 2,000-row
holdout and four duplicate development observations remain excluded. Preprocessing
and constant removal are training-fold-only. No holdout scoring or test labels.

| Candidate | Mean CV accuracy |
|---|---:|
| lr_numeric | 66.533% |
| lr_identity | 61.206% |
| rf_numeric | 66.546% |
| rf_identity | 66.508% |

Identity features in this encoding did not improve either model; LR worsened
in all five folds, suggesting too many weak/rare features for the regularization
and scaling used. RF numeric won narrowly, not convincing evidence over LR numeric.
It was fitted on all 10,000 labels and generated a validated 5,000-row submission.
This initial simple model remains below the historical 67.234% boosting result;
it must not be presented as a leaderboard improvement.

Eleven unittests passed, including three new identity invariants: team ordering,
future/target independence and fold vocabulary isolation. Subsequent artifact
verification is recorded separately. Repeated development CV is exploratory.
All settings, source snapshots, input hashes, OOF probabilities, final model,
CSV hashes and environment versions are in the run folder.

A separate bounded follow-up tests three LR C values with scaling restricted
to baseline numeric features. This is explicitly adaptive after the initial results.
