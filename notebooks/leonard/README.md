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

Keep your exploration notebooks here. Copy `templates/exploration.ipynb` to start. Use your own branch and descriptive notebook names.
