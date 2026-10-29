# Team workflow

## Ownership without blocking parallel work

Each person owns their exploration folder. Write reusable functions in separate
modules such as `features/team_stats.py` and `features/timeline_hp.py`, rather than
having everyone edit one big file. Use clear inputs and return types in docstrings.
Keep feature extraction separate from labels, fitting, validation, and file writing.
Agree who edits shared analyses, validation code, the final notebook, and report
assembly before starting those changes. Roles can rotate; personal folders do not
assign anyone a modelling responsibility.

## One branch per task

From a clean `main`, pull and create `<owner>/<task>`. Commit only relevant files,
not downloaded data or outputs. Push the branch and open a pull request. Describe
what changed, how to run it, and the validation result (if it changes modelling).
A teammate reviews shared code before merging. Delete merged branches when convenient.
If your working tree has edits, commit or stash them before switching branches.

Before a long-running branch merges, bring current `main` into it with
`git fetch origin` then `git merge origin/main`, resolve conflicts, and rerun affected
checks. Avoid force-pushing shared branches. Keep direct pushes to `main` exceptional.
These are team conventions; GitHub branch protection is not configured by this folder.

## Notebook discipline

Copy `templates/exploration.ipynb` into your personal folder and use a descriptive
name (`01_team_stats.ipynb`). Do not simultaneously edit the same notebook.
Before committing, restart and run all cells to check execution order, then clear
outputs to reduce noise and large diffs. Do not embed full datasets or credentials.
Move stable logic into `src/pokemon_challenge/` so it can be imported by other
notebooks and the final Kaggle notebook. The reference starter is an example; make
experiments in your own copy rather than competing to edit it.

## One record per experiment

Copy `templates/experiment.md` to `experiments/<owner>/<unique-run>.md`.
Do not maintain a single shared spreadsheet everyone edits. Log failed experiments
too. Record exact commit, parameters, features, fold setup, per-fold scores, and
artifact locations. Use local validation to select models; Kaggle feedback is an
occasional check. Designate a submission coordinator to avoid exhausting the shared
three-submissions-per-day allowance. Record Kaggle submission IDs and selection state.

## Shared files need coordination

Validation settings, feature interfaces, requirements, and final reproduction paths
affect everyone. Discuss changes in the relevant pull request. Installing an updated
`requirements.txt` is a local step each teammate must run after pulling. Dependencies
are initially unpinned; record versions for experiments and agree on exact versions
for the final reproducible environment.

No remote publication, branch protection, or automatic CI has been configured by
this restructuring. The team must push reviewed commits and enable any GitHub rules
it wants in repository settings.
