# Official inputs

Keep downloaded files unchanged in `data/raw/`. Data files are ignored by Git;
each teammate downloads them from Kaggle independently. Expected files:
`train.jsonl`, `test.jsonl`, `sample_submission.csv`, `type_chart_gen2.csv`.
Do not add external data to this project. Use the shared path helper, not an absolute
path to someone's computer. Only this README is tracked within `data/`.

Prepared reference data lives in `data/prepared/pokemon_details/`. Generate it once
from the training file with `python -m pokemon_challenge.preparation.pokemon_details`.
The generated JSON is ignored by Git and can be rebuilt on each machine. See the
preparation section in the repository README for its schema and notebook imports.
