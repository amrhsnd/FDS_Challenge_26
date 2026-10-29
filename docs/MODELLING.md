# Run a modelling experiment from a notebook

Copy `templates/modelling.ipynb` into your personal notebook folder, select the
project `.venv` kernel, and set `OWNER`. Each execution of the setup cell creates a
timestamped run name. The setup cell locates `src/` without requiring a package
installation when you run inside this checkout.

## The notebook controls the experiment

1. Choose feature functions, or define your own in the notebook.
2. Construct `BattleExperiment(owner, run, feature_functions=...)`.
3. Define `Candidate` objects for feature groups, model settings, and optional
   automatic selection percentiles, then call `experiment.compare(candidates)`.
4. Compare mean accuracy, individual fold scores, fold variation, selection
   stability, runtime, and the majority-class reference. Add new candidate names
   with another `compare` call if needed.
5. Once decisions are complete, call `evaluate_holdout(chosen_name)` once.
6. Call `make_submission()` to refit the chosen pipeline on all labels and predict
   official test battles. It does not upload anything.

The supplied notebook defaults to CV comparison only. Holdout evaluation and final
submission generation each require explicitly enabling their notebook flags.

## Validation and feature selection

Defaults are 20% holdout and five stratified folds within the remaining 80%, seed 42.
The same ordered training data/settings produce identical splits; compare the
saved split hash, not only the seed. `splits.csv` records each battle's partition and
validation fold (`-1` for holdout). No turns are independently split.

Each fold fits median imputation, constant removal, optional ANOVA percentile
selection, and (for logistic regression) standard scaling using its training rows
only. Feature-group comparison is a manual ablation over these CV results.
`percentile=100` disables ANOVA selection; constants are still removed.
Selection frequencies report how often each input survived the fitted transforms
across folds; selected subsets need not be identical across folds.

Repeated feature/model selection can overfit CV scores, which is why the holdout
stays separate. Opening it prevents later comparisons in that Python experiment
object. Do not tune against it by creating new objects/runs. The lock is a workflow
guard, not a persistent access policy. The final full-data fit includes the former
holdout; it has no new local generalization score.

This basic runner supports numeric features, classical classifiers, accuracy, and
ordinary stratified folds. It does not yet supply categorical encoding, grouped
splits, nested CV, or an automated parameter search. Investigate duplicates/related
battles before deciding the ordinary split is appropriate.

## Custom numeric features

A feature function takes one battle without `battle_id` or `player_won` and returns
a nonempty flat dictionary with fixed string keys and numeric values. `None` and
NaN are allowed; infinity and strings are rejected. Keys must match across train
and test observations. Functions must be stateless: no dataset-wide fitting or
target-dependent selection. Learned transforms belong in the sklearn pipeline.
Feature names are prefixed with the group name to avoid collisions.

Keep experimental functions in your notebook; promote shared ones into
`src/pokemon_challenge/features/`. Built-in groups are `team_stats` and
`timeline_summary`; timeline HP describes the active observed Pokémon, not total
remaining team health.

Supported model names are `logistic_regression`, `random_forest`, and
`hist_gradient_boosting`. Pass estimator parameters in `Candidate.params`.

## Saved evidence

Each nonempty run folder is protected against accidental reuse. Artifacts include
the candidate settings, per-fold scores, feature-selection frequencies, input and
split hashes, Python/library versions, feature source where recoverable, code
commit, and dirty-tree status. Holdout and final fitting add their own metrics,
predictions, model, and final manifest. All are local under ignored `outputs/`.

For a final reproducible result, commit the implementation, retain the notebook and
exact configuration, save any custom dependencies/globals, and preserve the data
version. Source snippets and hashes are evidence, not a standalone reproduction
package. `model.joblib` expects the recorded selected input feature columns, not raw
JSON battles; the same feature functions must still run first.

## Checks

Run `python -m unittest discover -s tests -v`. Synthetic fixtures cover fixed folds,
partition separation, train-only preprocessing, the feature contract, model/feature
comparisons, holdout guards, saved artifacts, and sample submission alignment.
