# Classical identity features and scaling follow-up

Owner: Leonard. Final run: `20261102_124639_898979_classical_identity_followup`.
Initial run: `20261102_123710_282289_classical_identity`.
Only logistic regression and random forest were fitted. No boosting or ensemble.

Species/move features are unordered own roster membership and HP, observed
species/exposure/HP, move counts/rates/recent use/recency, attack-type rates,
active species, opponent lead and last/most-used moves. Only turns 1–30;
no target or ID features. A fold-fitted DictVectorizer learns vocabularies.
New validation/test categories are ignored safely. Last observed HP can be stale.

Five original grouped development folds on 7,996 rows. All 2,000 previously
opened holdout battles plus four development duplicates remain excluded.
Initial LR/RF x numeric/identity comparison uses four prespecified candidates.
The poor LR identity result motivated a separate adaptive, bounded follow-up:
C=.01/.1/1, scaling ONLY the 289 baseline numeric inputs; identity indicators/rates
are left unscaled to avoid amplifying rare values. Three additional candidates.
No further tuning or holdout selection. Total 35 CV fits; 7 configurations.

| Candidate | CV accuracy |
|---|---:|
| lr_numeric | 66.533% |
| lr_identity | 61.206% |
| rf_numeric | 66.546% |
| rf_identity | 66.508% |
| lr_identity_base_scaled_C001 | 67.259% |
| lr_identity_base_scaled_C01 | 66.709% |
| lr_identity_base_scaled_C1 | 65.333% |

Selected LR with C=.01, L2 regularization, lbfgs, max_iter=4000, seed 42;
approximately 4,250 nonconstant numeric/one-hot features per development fold.
Base numeric features are standardized before constant filtering; identity values
remain unscaled. Final vocabulary, scaling and coefficients are refitted on all
10,000 training examples. No synthetic team-slot order or target encoding is used.

CV accuracy 67.2588%, vs initial numeric LR 66.5333% and RF 66.5460%.
The historical boosting result is 67.2337%, only 0.0251 pp lower (net two OOF
predictions); this is practically tied, not convincing evidence of improvement.
Repeated exploratory CV, especially the adaptive follow-up, limits certainty.
Only exact duplicates are controlled; related nonidentical battles may remain.

Final submission: 5,000 rows, 2,575 ones / 2,425 zeros. Correct official sample
order, unique IDs, binary labels, no missing values. Twelve tests pass, including
new target/horizon/order independence, train-only vocabulary, and selective scaling.
Independent OOF checks and model reload reproduced twelve raw test predictions
and probabilities. No test labels or new holdout scores used, no Kaggle upload.

Reproduction from repository root:
1. `.venv/bin/python notebooks/leonard/run_classical_identity_experiment.py`
2. `.venv/bin/python notebooks/leonard/run_identity_scaling_followup.py`
The follow-up uses the recorded initial run path by default; to follow a fresh
initial run, set the module's INITIAL path before calling run_followup().
Both phases use the earlier fixed split/feature cache, validated by SHA-256.
Review notebook 08 defaults to reading saved evidence and has an explicit rerun flag.

Runtime: initial four candidates used the project environment. OneDrive stalled
subsequent package imports; principal dependencies were installed at identical
versions in /tmp/pokemon_identity_runtime for the follow-up. Actual full dependency
versions are recorded in experiment.json and requirements_classical.txt. The
extra dateutil/tzdata versions differ; no date parsing/features are involved.
CatBoost had been installed before the user's simple-model clarification but was
never fitted or used, and is not a requirement for reproducing these models.

Artifacts: final output folder contains all 7 OOF files, CV/paired comparisons,
selection, model, CSV/probabilities, hashes/settings, validation, dependencies,
review notebook and source snapshots. The initial record documents rejected results.
