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

[09_feature_selection.ipynb](09_feature_selection.ipynb) compares fold-fitted
SelectKBest (100/300/1,000 features) and logistic RFE (300/1,000) with the best
simple-model reference. No reduction improved mean CV accuracy. RFE300 retains
about 7% of inputs with a descriptive 0.10 percentage-point accuracy loss.
The notebook explains the pipeline and points to actual selected-feature lists.
Run `run_feature_selection_experiment.py` to reproduce the six comparisons;
it uses the validated caches and fixed splits from earlier local runs.

The compact RFE300 version was subsequently chosen for simplicity. Run
`finalize_rfe300_model.py` to refit its selection/classifier on all labels and
save its exact 300-feature list plus a separate compact-model submission. Its
recorded procedure CV accuracy is 67.159%; the all-data list must not be reused
before future CV. See `experiments/leonard/20261102_rfe300_compact.md`.

[10_matchup_features.ipynb](10_matchup_features.ipynb) compares the compact
baseline with active matchups, team coverage and both groups. Team coverage adds
16 inputs to the unchanged fold-fitted RFE300 baseline: **316 inputs, 67.259% mean
CV accuracy**, versus 67.159% for the baseline. This small gain remains exploratory.
Type lookup, imputation and scaling are learned inside training folds; no enemy
team members are invented. Run `run_matchup_experiment.py` to reproduce.
The selected model's separate submission is at:
`outputs/leonard/20261102_154044_358954_matchups/submission.csv`.

[11_counter_quality.ipynb](11_counter_quality.ipynb) tests counter redundancy,
HP quality and both on top of the unchanged 316-input procedure. HP quality wins:
**324 inputs, 67.396% mean CV accuracy**, versus 67.259% for the 316-input model.
It improves four of five folds (net 11 additional correct predictions), a small
exploratory gain. Run `run_counter_experiment.py` to reproduce. The selected
model's separate submission is at:
`outputs/leonard/20261103_140058_542958_counter_quality/submission.csv`.

The runner scripts and feature implementations live beside their notebooks.
`run_classical_identity_experiment.py` runs the initial classical comparison;
`run_identity_scaling_followup.py` runs its three-setting follow-up. The tests
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

[13_frozen_matchup_submission.ipynb](13_frozen_matchup_submission.ipynb) is the
current submission entrypoint for Leonard's selected **316-feature matchup-only
logistic regression** (300 RFE features + 16 team-matchup features; 67.259%
historical development CV accuracy, 68.80% public accuracy reported by the user).
It contains all feature and training code, with comments and short Markdown
explanations before each of its 19 code cells. It reads the frozen configuration
from `frozen_matchup_submission.json`, rebuilds inputs from the four official
files, and refits on all 10,000 training battles. No counter-quality additions or
earlier experiment caches are used.

Prepare the [frozen runtime](#frozen-runtime-setup), then run all cells. The exact
earlier matchup CSV is saved locally to
`outputs/leonard/frozen_matchup_submission/submission.csv`, or to
`/kaggle/working/submission.csv` on Kaggle. Its expected SHA-256 is
`4744bf724de616e9857606470f953eca0ecdf2108a0dc3d9884beed165a22051`.
The notebook verifies all 316 selected feature names and the CSV checksum before
writing it, and saves a `reproduction_report.json`. All 19 code cells were
executed locally in order and reproduced all 5,000 earlier predictions exactly.
[12_frozen_submission.ipynb](12_frozen_submission.ipynb) retains the earlier
**324-feature logistic regression (67.396% historical development CV accuracy)**
as a separate alternative. Run all cells in order with the frozen runtime. It
rebuilds features and fits RFE/the model from `train.jsonl`, `test.jsonl`, `sample_submission.csv` and
`type_chart_gen2.csv`, without any earlier experiment caches. Set
`POKEMON_DATA_DIR` if the official inputs are not under `data/raw` or the detected
Kaggle competition directory.

All feature extraction, encoders, type matching, counter quality, RFE, model
fitting and CSV generation are visible in the notebook's 20 code cells. Each has
a short Markdown explanation and code comments. It does not import or call
`create_submission()` or hide model logic in repository modules.
`frozen_submission.json` supplies the frozen column order, feature names,
input checksums, reference dependencies and exact submission checksum.
The separate `frozen_submission.py` remains available as the earlier programmatic
entrypoint, but the reviewer notebook executes its own visible implementation.
The checked CSV is written to
`outputs/leonard/frozen_submission/submission.csv` locally, or
`/kaggle/working/submission.csv` on Kaggle. `reproduction_report.json` records the
successful checksum check and runtime versions.

All 324-feature notebook code cells were executed in order locally: full raw-data rebuild,
10,000 training rows, 324 final features, 5,000 predictions, exact original CSV
SHA-256 `4692f7dbde612160235b9fcb1a882da24429fdd65ce8dc322149b3ea2edc907b`.

## Frozen runtime setup

Both submission notebooks verify Python 3.12 or newer and the exact package
versions in their JSON manifests before importing the model dependencies. The
main project environment may resolve newer versions. Install the reference
packages before starting the interpreter that will execute the notebook.

### Local kernel

From the repo root, create a dedicated environment. The example uses Python 3.14
and notebook 13. For notebook 12, use `frozen_submission.json` instead.

```bash
python3.14 -m venv outputs/leonard/frozen-runtime
outputs/leonard/frozen-runtime/bin/python - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

manifest = Path('notebooks/leonard/frozen_matchup_submission.json')
versions = json.loads(manifest.read_text())['versions']
pins = [f'{name}=={version}' for name, version in versions.items()]
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ipykernel', *pins], check=True)
PY
```

Select `outputs/leonard/frozen-runtime/bin/python` in VS Code's notebook kernel
picker and start a fresh kernel.

### Kaggle

Enable Internet and attach the official competition data. If the Kaggle kernel
already has the exact versions, open the updated notebook and run all cells. It
can clone the frozen JSON configuration at a fixed Git commit.

If the versions differ, run the following in a Kaggle Python notebook. Attach
the updated submission `.ipynb` as a dataset file, or point `notebook` at its
path in an updated checkout. The child Python executes that file in order using
its own virtual environment. The fixed Git commit supplies the JSON manifest.

```python
import json
from pathlib import Path
import subprocess
import sys

if sys.version_info < (3, 12):
    raise RuntimeError('Start from a Kaggle runtime with Python 3.12 or newer.')

working = Path('/kaggle/working')
notebook = Path('/kaggle/input/your-notebook-dataset/13_frozen_matchup_submission.ipynb')
if not notebook.is_file():
    raise FileNotFoundError(f'Point notebook to the updated .ipynb file: {notebook}')

revision = '8b99ef161018a2fc07391696edb0804705025322'
checkout = working / f'fds-frozen-config-{revision[:7]}'
if not checkout.exists():
    repo = 'https://github.com/amrhsnd/FDS_Challenge_26.git'
    subprocess.run(['git', 'clone', '--no-checkout', repo, str(checkout)], check=True)
subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', revision], check=True)

manifest_name = {
    '12_frozen_submission.ipynb': 'frozen_submission.json',
    '13_frozen_matchup_submission.ipynb': 'frozen_matchup_submission.json',
}[notebook.name]
manifest = checkout / 'notebooks/leonard' / manifest_name
versions = json.loads(manifest.read_text())['versions']
pins = [f'{name}=={version}' for name, version in versions.items()]
runtime = working / 'fds-frozen-runtime'
python = runtime / 'bin/python'
if not python.is_file():
    subprocess.run([sys.executable, '-m', 'venv', str(runtime)], check=True)
subprocess.run([
    str(python), '-m', 'pip', 'install', 'ipykernel', 'matplotlib', 'nbformat', *pins,
], check=True)

runner = checkout / 'analysis/execute_notebook.py'
output = working / f'{notebook.stem}_executed.ipynb'
subprocess.run([
    str(python), '-u', str(runner), str(notebook), '--output', str(output),
], check=True)
```

Both notebooks write `/kaggle/working/submission.csv` and
`/kaggle/working/reproduction_report.json`. Save each pair before running the
other notebook if you need both submissions.

Generated data, executed notebook snapshots, models and submissions stay under
ignored `outputs/`. Earlier CSV paths in experiment sections are historical;
use notebook 13 and its `frozen_matchup_submission` output for the selected model.

Keep your exploration notebooks here. Copy `templates/exploration.ipynb` to start. Use your own branch and descriptive notebook names.
