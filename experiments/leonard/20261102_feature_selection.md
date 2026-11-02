# K-best and recursive feature elimination

Owner: Leonard. Run `20261102_144359_081808_feature_selection`.
Reproduce: `.venv/bin/python notebooks/leonard/run_feature_selection_experiment.py`.
When OneDrive blocks imports, the recorded local runtime was invoked with
`PYTHONPATH=/tmp/pokemon_identity_runtime PYTHONPYCACHEPREFIX=/tmp/pokemon_identity_pycache`.
The /tmp runtime is an optional local workaround, not a persistent dependency.

Compare a fixed six-candidate set on the same 7,996 development battles and five
duplicate-grouped folds. The previously opened holdout (2,000) plus four development
duplicates remain excluded. Baseline numeric columns, identity dictionary cache,
raw data and split hashes are checked. LR settings unchanged: L2, C=.01, lbfgs,
max_iter=4000, seed 42. Numeric baseline columns standardized; identities unscaled.

Pipeline: fold-fitted identity vocabulary -> baseline-only scaling -> constant
removal -> selection -> LR. SelectKBest uses f_classif (F-test), k=100/300/1000.
RFE uses another identical LR as the ranking estimator, k=300/1000, step=.25:
remove 25% of the original input count at each round, stopping at requested k.
Selection is refitted inside every training fold, never on all labels before CV.
The final fitted classifier is separate from the estimator internal to RFE.

| Method | Features per fold | Mean CV accuracy | Change vs baseline |
|---|---:|---:|---:|
| No additional selection | ~4,250 | 67.259% | — |
| K-best | 100 | 66.471% | -0.788 pp |
| K-best | 300 | 66.921% | -0.338 pp |
| K-best | 1,000 | 66.796% | -0.463 pp |
| RFE | 300 | 67.159% | -0.100 pp |
| RFE | 1,000 | 67.159% | -0.100 pp |

No reduction improves observed mean accuracy. Retain the original recommended
submission, copied byte-for-byte into this run. RFE300 is a reasonable compact
alternative (~93% fewer features, eight fewer correct OOF predictions out of
7,996). That small difference is descriptive, not proof that either is superior.
Do not assume it is the same 300 features in every fold.

Thirty outer candidate/fold fits; RFE runs additional internal LR fits. No new
hyperparameter tuning, holdout scoring, boosting or Kaggle upload. Since the baseline
won, its exact full-data model (10,000 rows, 4,478 surviving full-vocabulary features)
and 5,000-row CSV are reused instead of refitting an unchanged configuration.

13 tests pass, including a new synthetic test for train-only vocabulary,
requested feature count and readable selected-feature mapping. Independent
artifact checks confirm all OOF scores, same splits, exact baseline reproduction,
zero duplicate-group overlap, correct per-fold selected counts and original CSV/model
hashes. Serialized model predictions reproduced eight raw test observations.

Limitations: repeated exploratory CV is not independent confirmation. KBest tests
features separately and can miss useful combinations. RFE depends on regularization
and coefficient scaling; retained mixed scaling is intentional, so RFE should not
be interpreted as a scale-independent importance ranking. Correlated/redundant
features can substitute for each other. Selection stability counts include absence
from some training vocabularies. Only exact duplicate groups are controlled.

Artifacts include cv_results.csv, paired_comparison.csv,
selected_features_by_fold.csv (actual semantic names), selection_stability.csv,
final_selected_features.csv (recommended full-data model), model/CSV/probabilities,
input hashes/settings/versions, sources/tests and an executed review notebook.

Method reference: https://scikit-learn.org/stable/modules/feature_selection.html
