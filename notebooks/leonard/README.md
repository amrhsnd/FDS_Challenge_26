# Leonard

Start with [01_basic_data_exploration.ipynb](01_basic_data_exploration.ipynb) for
battle examples, every nested field, missing values, numeric summaries, complete
categorical vocabularies, and plots. Generated CSV summaries are saved locally to
`outputs/leonard/basic_data_exploration/`.

Use [02_modelling_pipeline.ipynb](02_modelling_pipeline.ipynb) for feature-group
comparison, model comparison, five-fold validation, optional holdout evaluation,
and final submission generation. Copy `templates/modelling.ipynb` for another run.

The first complete baseline is
[03_logistic_regression_baseline.ipynb](03_logistic_regression_baseline.ipynb):
all implemented numeric predictors, logistic regression, five-fold CV, holdout
evaluation, full-data training, and a checked submission CSV.

Next, [04_three_model_comparison.ipynb](04_three_model_comparison.ipynb) compares
logistic regression, random forest, and histogram gradient boosting using identical
baseline predictors and five-fold assignments. It saves a reference for feature
engineering without evaluating the holdout or generating a submission.

Use [05_aimilia_feature_ablation.ipynb](05_aimilia_feature_ablation.ipynb) to compare
Aimilia's 106 features with versions that remove one feature group at a time.
It reports paired accuracy changes, compares added features/model settings, then
performs a final holdout evaluation and generates a submission. Optional combined
removals and permutation importance remain disabled.

[06_progressive_feature_experiments.ipynb](06_progressive_feature_experiments.ipynb)
tests remaining team strength, advantage trajectory and boost threat individually.
It uses five duplicate-grouped development folds and quarantines the previously
opened holdout plus its development duplicates.

[07_simplification_experiment.ipynb](07_simplification_experiment.ipynb) compares
289 versus 45 features and original versus more constrained boosting. These
particular simplifications did not improve CV accuracy.

[08_classical_identity_experiment.ipynb](08_classical_identity_experiment.ipynb)
compares **logistic regression and random forest only**, with and without species
and move identities. A bounded follow-up standardizes only numeric baseline
features, leaving identity indicators/rates unscaled. The selected logistic
regression (L2, C=0.01) achieved **67.259% mean development CV accuracy** and was
refitted on all 10,000 labels to generate 5,000 test predictions. It is practically
tied with the earlier boosting reference; the repeated exploratory CV does not
establish a higher Kaggle score.

The runner scripts and feature implementations live beside their notebooks.
`run_classical_identity_experiment.py` runs the initial classical comparison;
`run_identity_scaling_followup.py` runs its three-setting follow-up. Twelve tests
cover the common pipeline and custom feature/encoding/scaling invariants:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Exact settings, results, limitations and reproduction notes are recorded under
[`experiments/leonard/`](../../experiments/leonard/). Notebooks 06–08 review saved
artifacts by default and provide an explicit flag for rerunning experiments.
Their runners reference earlier local feature/split caches and run IDs: those
ignored artifacts must exist, or their paths must be updated to corresponding
recreated runs. A fresh clone alone does not include those caches.

Generated data, executed notebook snapshots, models and submissions stay under
ignored `outputs/`. The final simple-model CSV is at:
`outputs/leonard/20261102_124639_898979_classical_identity_followup/submission.csv`.

Keep your exploration notebooks here. Copy `templates/exploration.ipynb` to start. Use your own branch and descriptive notebook names.
