# Type matchup features versus compact RFE300 baseline

Owner: Leonard. Run `20261102_154044_358954_matchups`.
Reproduce: `.venv/bin/python notebooks/leonard/run_matchup_experiment.py`.
Same optional local package runtime as previous runs; source caches/splits validated
by SHA-256. All candidates use LR C=.01, L2, lbfgs, max_iter=4000, seed 42.

Four fixed variants: baseline RFE300; +10 active-matchup inputs; +16 team-coverage
inputs; +all26. Constant own-active-types-known flag removed inside training folds,
so effective active/all counts are 9/25. RFE is fitted on the same old inputs
inside every fold before new features are appended. New inputs cannot change
which 300 old inputs are selected. Matchups use fold-fitted median imputation,
constant removal and scaling, followed by one joint logistic regression.

Five identical duplicate-grouped development folds, 7,996 battles. Prior 2,000-row
holdout plus four duplicates stay excluded. Exact baseline predictions/probabilities
and every 300-feature subset reproduce the previous run. Twenty outer fits;
RFE adds internal LR fits. No boosting or hyperparameter search.

Gen-2 chart: omitted valid type pairs are neutral; dual types multiply and immunity
is zero. Own-type/STAB potential is a proxy, not a claim that the move is known
or actual damage. Observed positive-power physical/special move types give a
separate coverage measure. Hidden Power and zero-power/fixed-damage moves are
omitted; abilities/items and full move mechanics are not simulated.

Species types are recovered only from training-fold own-team and lead metadata.
At prediction, metadata explicitly supplied in that battle is overlaid without
mutating the fitted lookup. Unknown enemy types stay missing with coverage flags;
missing numeric values are imputed using training-fold medians. Own unseen members
start at HP=1; enemies are evaluated only if revealed and last seen alive. Team
weights are own last HP × enemy last HP. Observations/moves can be stale.

| Variant | Effective inputs | Mean CV accuracy | Change vs compact baseline |
|---|---:|---:|---:|
| RFE300 baseline | 300 | 67.159% | — |
| + active matchups | 309 | 66.971% | -0.187 pp |
| + team coverage | 316 | 67.259% | +0.100 pp |
| + both groups | 325 | 67.184% | +0.025 pp |

Selected team coverage. Three folds improve, one ties, one declines. Corrects 162
baseline errors and introduces 154, net eight more correct / 7,996. This is a small,
exploratory CV gain; repeated development selection is not independent confirmation
and does not establish a higher Kaggle score. Correlated derived features can also
alter effective regularization. Only exact duplicates, not related nonidentical
battles, are grouped. Do not treat LR coefficient magnitude as scale-independent
or causal importance.

Full-data winner is LR on 300 refitted baseline features plus all 16 team inputs,
316 total, trained on all 10,000 labels. Saved 5,000 official test predictions.
CSV IDs/order, binary labels, missing values checked. Serialized model reproduced
all 5,000 probabilities/labels; independent raw-feature reconstruction reproduced
12 examples. 21 tests pass, including eight new matchup checks (direction,
dual typing, immunities, neutral omissions, unknowns, real-vs-potential attacks,
HP weights, unseen own members, horizon, order and fold-local metadata).
No test labels or holdout scoring, no Kaggle upload.

Artifacts: `outputs/leonard/20261102_154044_358954_matchups/` contains all scores,
paired deltas, coverage audit, per-fold selected names, final 316-feature list,
16 new-feature definitions, sources/chart, hashes/settings/versions, validation,
executed review notebook, final model and checked submission.
