# Four selected additions to Amir's notebook

Date: 2026-11-03, Europe/Rome. The notebook now has **32 inputs: the existing 28 plus four selected additions**. Ten was the maximum; selection stopped after four because no remaining candidate met every preset criterion. Six of seven model configurations improved; histogram gradient boosting declined slightly.

## Selected features and sources

All definitions use only the first 30 timeline entries, then retain integer turn numbers 1–30 and process them chronologically. Track each species once using its latest observed HP/status. Benched observations can be stale. Initial P1 roster members and the initial P2 lead start full and healthy; a faint forces HP to zero. Missing HP preserves the last known value, and missing observed status is unknown.

Let `h_i` be a member's latest HP fraction, clipped to [0,1], with fainted members at zero. Let `O2` contain distinct observed P2 species including its lead, `R1` the supplied P1 roster, and `T` the valid retained timeline rows.

| Added input | Exact definition | Source and purpose |
| --- | --- | --- |
| `p2_healthy_living_members` | `min(6, max(0, 6 - len(O2)) + sum(h_i > .5 and status_i == 'nostatus', i in O2))` | Healthy-member idea from `2025/de-lucia-fratello-marchio.ipynb`, cell 7. Distinguishes a healthy bench from equal total HP spread among weakened or afflicted members. Unseen slots are healthy proxies; no hidden species are invented. Unknown observed HP/status does not qualify. |
| `p1_remaining_special_bulk` | `sum(base_hp_i * base_spd_i * h_i, i in R1)` | Remaining special-bulk idea from the same 2025 cell. Weights remaining HP by supplied defensive base stats. Sleep and freeze remain living; this is a heuristic, not actual level-adjusted battle HP. Missing base stats contribute zero. |
| `p1_hp_weighted_base_special_attack` | `sum(base_spa_i * h_i) / sum(h_i)` over supplied members with numeric `base_spa`; zero when the denominator is zero | Extension of that notebook's remaining-resource/offense idea. Measures which special attackers retain HP rather than the original roster's unweighted strength. |
| `p1_leechseed_rate` | `sum('leechseed' in p1_effects_t, t in T) / len(T)`; zero for an empty prefix | Effect-exposure idea from `notebooks/aimilia/features_effects.py`. Every valid retained row contributes to the denominator; missing effects contribute zero. Measures recurring resource pressure not represented by HP totals alone. |

Final implementation: `src/pokemon_challenge/features/retained_position.py`, function `retained_position_features`. It uses no labels, IDs, learned lookups, external data, or cross-battle memory. Its outputs match the experimental candidate implementations on all 10,000 supplied training battles; missing-input behavior is explicitly defined above.

## Selection rule

The source-backed pool contained **48 candidates**: 39 active-HP, ailment, move, effect, boost, checkpoint and healthy-roster summaries from the supplied Aimilia/2025 material, plus nine remaining-resource extensions. Leonard and last year's notebooks informed the review; renamed duplicates were not counted as new information. Previously removed elemental-type counts, winning-team flags and missing-move proxies were not restored.

Greedy forward selection used logistic regression (`C=1`, `max_iter=2000`) on the original five development folds. Imputation, constant removal and scaling were fitted within each training fold. At each step, an addition had to satisfy all frozen rules:

- Incremental mean accuracy gain **at least 0.125 percentage points** (10 additional correct predictions over 8,000 development rows).
- Accuracy improved in **at least three of five folds**.
- Absolute Pearson correlation **strictly below 0.98** with every already retained input.
- No exact affine re-expression of the retained inputs (standardized projection residual mean square below `1e-12` was rejected).
- Mean log loss increased by **at most 0.0005**.
- More than one distinct value and **at least 50 nonzero development rows**.

Choose the passing candidate with highest mean accuracy, then lowest log loss, then feature name for ties. Stop when none passes, or after ten additions. The selected order is the table order above. No fifth candidate qualified. Baseline mean log loss was **0.6060575973**; the selected set achieved **0.6026735909**.

## Paired model comparison

The same input file, 8,000 development rows, five fold assignments and model settings were used before and after. RFE retains 22 original inputs; PCA retains 95% of scaled training variance. These settings were preserved.

| Model | 28-input accuracy | 32-input accuracy | Change, pp | Folds improved |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 66.3375% | **67.3625%** | +1.0250 | 4/5 |
| RFE + logistic regression | 65.9625% | 67.0750% | +1.1125 | 5/5 |
| XGBoost | 65.9125% | 66.0500% | +0.1375 | 3/5 |
| PCA + logistic regression | 64.4875% | 65.8125% | +1.3250 | 4/5 |
| Histogram gradient boosting | 66.0125% | 65.8750% | −0.1375 | 2/5 |
| Random forest | 65.3500% | 65.4625% | +0.1125 | 3/5 |
| RFE + PCA + logistic regression | 64.9250% | 65.0750% | +0.1500 | 2/5 |

Logistic regression's gain is **82 additional correct out-of-fold predictions out of 8,000**, from 5,307 to 5,389. Its fold accuracies changed from `[65.5000, 66.9375, 69.1250, 67.0625, 63.0625]%` to `[66.0000, 68.1875, 68.7500, 69.1875, 64.6875]%`. Fold standard deviation fell from 2.0041 to 1.7291 pp; this is not a confidence interval.

Evidence: `outputs/amirhosein/20261103_172150_addition_comparison/paired_comparison.csv` and `cv_results.csv` contain exact paired results and individual fold scores.

## Leave-one-out check within the selected set

Each row removes one new input and refits the logistic pipeline on the same folds. Costs are conditional and do not sum to the combined gain.

| Removed feature | Accuracy without it | Accuracy cost, pp | Folds worse without it |
| --- | ---: | ---: | ---: |
| `p2_healthy_living_members` | 66.3625% | 1.0000 | 4/5 |
| `p1_remaining_special_bulk` | 66.8125% | 0.5500 | 4/5 |
| `p1_hp_weighted_base_special_attack` | 67.1500% | 0.2125 | 4/5 |
| `p1_leechseed_rate` | 67.2375% | 0.1250 | 3/5 |

Evidence: `outputs/amirhosein/20261103_171946_feature_selection/leave_one_out.csv`; complete forward trials and screening profiles are saved beside it.

## Limits and reproducibility

These gains are **exploratory**: candidate selection and reporting reused the development folds. The comparison across seven models does not supply independent validation. No holdout was evaluated in this experiment and no official test file was read. Historical runs have evaluated this same holdout split, so it should not be described as historically untouched.

Repository commit at comparison: `338cbc16f351d74e3a56e01084157edaca013d80`; working tree was dirty. Existing unrelated edits were preserved. Recorded versions: Python 3.14.2, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0. The manifest records the full model configurations and environment.

- Training SHA256: `78f9e91c51b92adebe803bb09221347a55423dce080e433465edf527d2c20d0d`.
- Split-assignment SHA256: `43ffdf6d5ee24b4e03a4a621ad8db40edea368d9dff59c4d10e02a0dd16d0e37`.
- Original notebook SHA256: `079c60f052391dd1b0b0a1af72a3dc7d241537bf689c773e2f3af69213599766`.
- Immutable baseline: `outputs/amirhosein/20261103_171946_feature_selection/baseline_notebook.ipynb` (also copied into the comparison folder).
- Selection code/candidates, original feature rows, labels, fold assignments and paired predictions are frozen in the selection folder; `selection.json` records all criteria.

Run commands from the repository with its `.venv` interpreter:

```bash
# Selection requires the original 28-input notebook, not the current 32-input one.
# Reproduce in a separate checkout: restore the saved baseline notebook to its
# original notebook path and frozen candidate/selector files to analysis/, then:
.venv/bin/python analysis/select_amir_additions.py

# Recheck the frozen selected set directly; this reads the saved baseline.
.venv/bin/python analysis/compare_amir_additions.py outputs/amirhosein/20261103_171946_feature_selection

# Execute the current 32-input notebook completely and retain a separate preview.
.venv/bin/python analysis/execute_notebook.py notebooks/amirhosein/amirs-notebook.ipynb --output outputs/amirhosein/current32_executed.ipynb
```

The selector asserts exactly 28 baseline inputs and therefore intentionally does not rerun against the current notebook. The comparison script checks training/split hashes before comparing. Keep holdout and submission flags false while reproducing these development results.

Validation: all **37 unit tests passed**; the final notebook executed all **17 code cells** successfully, retaining **seven figures** and seven rich tables. All seven fresh CV means and every fold score reproduce the paired 32-input comparison. The saved preview and figures were visually inspected. Executed outputs are retained in `outputs/amirhosein/20261103_172150_addition_comparison/amirs_executed.ipynb`; the committed notebook has outputs cleared per repository convention.

Final notebook run: `outputs/amirhosein/20261103_172845_931462_selected_features`. Execution receipt: `outputs/amirhosein/20261103_172150_addition_comparison/amirs_executed.execution.json`.
