# Pokémon battle prediction · 300 features

Open `amirs-notebook.ipynb` with the repository `.venv` kernel and run all cells.
Put the official inputs in `data/raw/` or set `POKEMON_DATA_DIR`.

Each training fold selects exactly **300 predictors**. Selection removes missing,
constant and duplicate columns, ranks utility using linear, tree and univariate
signals, and limits every retained pair's absolute Pearson correlation to **0.90**.
All selection, imputation and scaling is fitted on training rows only.

The notebook displays the development-fit 300-column set with descriptive names,
an expandable dictionary, removal counts, four-model comparison and prediction
diagnostics. Fold selections can differ. `EVALUATE_HOLDOUT` and `MAKE_SUBMISSION`
default to False. Model choices on reused development data remain exploratory.

The helpers compute roster/timeline measurements (`features.py`), define names
and units (`feature_catalog.py`), select inputs (`feature_selection.py`), and train
duplicate-aware models (`modeling.py`). Lower-level feature and validation helpers
remain separate from the notebook's reading flow.

Each run saves `selected_features.csv` (300 columns), `feature_definitions.csv`,
`feature_selection_audit.csv`, `feature_set.json`, fold assignments and predictions,
model settings, charts and source snapshots under `outputs/amirhosein/<run>/`.
If submission is enabled, the all-label refit saves its own final 300-feature list.
Nothing is uploaded automatically.

Run checks with `.venv/bin/python -m unittest discover -s tests`.
