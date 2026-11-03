# FDS Pokémon Challenge 2026

Predict the final winner of a Pokémon battle using its first 30 observed turns.

Team: Aimilia Kokkinofta, Leonard Thorner, Amirhosein Dalvand, Meghi Shehu.

## Project layout

```text
FDS_Challenge_26/
├── data/raw/                 Official local inputs; ignored by Git
├── notebooks/
│   ├── reference/            Course starter notebook
│   ├── shared/               Agreed analyses maintained by one person per change
│   ├── aimilia/              Aimilia's exploration notebooks
│   ├── leonard/              Leonard's exploration notebooks
│   ├── amirhosein/           Amirhosein's exploration notebooks
│   └── meghi/                Meghi's exploration notebooks
├── src/pokemon_challenge/
│   ├── paths.py              Shared project/data/output path handling
│   ├── data.py               JSONL reader
│   ├── features/             Reusable feature functions
│   ├── models/               Reusable model construction/training
│   └── validation/           Shared splits and evaluation code
├── configs/validation.json   Starting validation settings
├── experiments/<owner>/      One Markdown record per experiment; committed
├── outputs/<owner>/<run>/    Generated files; ignored by Git
├── reports/                  Report sections and final reproduction notes
├── templates/               Personal notebook and experiment templates
├── docs/TEAMWORK.md          Branch, review, and integration workflow
├── pyproject.toml            Installable shared Python package
└── requirements.txt          Project dependencies
```

## Set up on each machine

Open **this repository folder** in VS Code. Create/activate a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
```

XGBoost is included in the project dependencies. On macOS, its native library
also needs the OpenMP runtime: install it with `brew install libomp` if importing
`xgboost` reports a missing `libomp.dylib`.

Install VS Code's Python and Jupyter extensions and select `.venv/bin/python` as the
notebook kernel. Put the official files in `data/raw/`: `train.jsonl`, `test.jsonl`,
`sample_submission.csv`, and `type_chart_gen2.csv`. Existing `data/` files are also
supported, as are the competition's common Kaggle mount paths.

Open `notebooks/reference/starter-notebook.ipynb` for the original baseline.
Copy `templates/exploration.ipynb` into your personal notebook folder for new work.
Its setup cell also works without installing the shared package when run from a
folder inside this repository. Use the editable install for imports from elsewhere.

## Prepare shared Pokémon details

After placing the official inputs in `data/raw/`, run once from the repository
with the project environment activated:

```bash
python -m pokemon_challenge.preparation.pokemon_details
```

This reads only `train.jsonl` and saves
`data/prepared/pokemon_details/pokemon_details.json`. To specify another source or
destination, use `--train /path/to/train.jsonl --output-dir /path/to/prepared`.
Re-run preparation when the training file changes; the artifact records its SHA256.
Generated data is local and ignored by Git; commit the preparation code separately.

Every notebook can then load the saved lookup without parsing training battles:

```python
from pokemon_challenge.preparation.pokemon_details import load_pokemon_details

pokemon = load_pokemon_details()
tyranitar_types = pokemon["tyranitar"]["types"]
```

The JSON's `pokemon` dictionary is keyed by species name and stores real types
(excluding `notype`), the six base stats, observed levels/items, and initial
level/item variants with counts and source fields. Levels and items are not
species constants. A null item in a variant means that record did not supply one.
Conflicting types or base stats cause preparation to fail rather than silently
choose a value. Only P1 rosters and P2 initial leads supply initial attributes;
timeline names are checked for coverage, with missing species listed under
`coverage.timeline_species_without_initial_details`. These species remain absent
from the lookup; use `pokemon.get(name)` and handle unknowns explicitly.
No labels, test data, or external reference data are used.

## Work in parallel

For a ready-to-run modelling workflow, copy `templates/modelling.ipynb` into your
personal notebook folder. Leonard's working example is
`notebooks/leonard/02_modelling_pipeline.ipynb`. Choose feature groups and model
candidates in the notebook; the shared runner performs five-fold CV on development
data, guards the holdout evaluation, and generates submissions when explicitly
requested. See [the modelling guide](docs/MODELLING.md).

1. Start with a clean `main`: `git switch main` then `git pull --ff-only`.
2. Create a branch such as `leonard/timeline-features`.
3. Work in your notebook folder and in a separately named shared Python module.
4. Use a unique run name, e.g. `leonard/20261029_01_hp_features`. Keep generated
   predictions/models under `outputs/`, and record the experiment under `experiments/`.
5. Commit code, settings, and concise experiment notes; clear notebook outputs first.
6. Push your branch and open a pull request into `main`; ask a teammate to review it.
7. Pull the merged changes before starting the next branch.

GitHub receives changes only after you commit and push them. Personal folders reduce
conflicts; branches and pull requests keep unfinished experiments off shared `main`.
Read [the teamwork guide](docs/TEAMWORK.md) before changing shared files.

## Shared paths from Python

```python
from pokemon_challenge.paths import data_dir, output_dir
from pokemon_challenge.data import iter_battles

first_battle = next(iter_battles(data_dir() / "train.jsonl"))
run_folder = output_dir("leonard", "20261029_01_hp_features")
```

`output_dir` creates the run folder. On Kaggle it uses `/kaggle/working/outputs/`;
locally it uses this repository's `outputs/`. Set `POKEMON_DATA_DIR` when data lives
elsewhere. Battle IDs are unique within each split and may overlap between splits.

## Compare and reproduce

`configs/validation.json` sets the initial seed, metric, and fold count. Before
comparing experiments, agree on the same actual battle-level fold assignments and
preprocessing rules. See `src/pokemon_challenge/validation/README.md`.

Keep each experiment's exact code commit, configuration, environment versions, and
output filename in its record. The final Kaggle notebook must clone this repository
at a fixed commit and reproduce the selected submissions; see `reports/README.md`.
