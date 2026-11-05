# Amir feature expansion · 4 November 2026

Base GitHub commit: 30238ab08f4afdc45c5b9639ceba84aa715b649f. Local branch: amirhosein/feature-expansion-20261104.

Expanded the original 32-feature notebook to 757 numeric
candidate columns in 20 independent feature families,
using the latest Aimilia and Leonard modules and supplied 2025 notebook ideas.
New feature providers use per-battle metadata only. The inherited original baseline
retains its fixed unlabeled species base-stat table. Overlapping providers remain
separate for ablation and redundant development vectors are reported.

Validation preserves the original 2,000-row holdout, quarantines
4 matching rows from development, and groups
identical permitted observations in five stratified development folds. Remaining
development rows: 7996. No competition holdout re-evaluation
or test prediction was performed. Repeated development selection is exploratory;
earlier team work has already examined the holdout.

Original 32-feature logistic baseline: 0.668584.
Best measured candidate: all_equal_soft_vote with 757 inputs;
mean grouped CV accuracy 0.678587, a gain of
+1.000 percentage points against that baseline.

Exact run: outputs/amirhosein/20261104_232631_018954_feature_expansion.
The run saves all candidate settings, per-fold scores, paired ablations, out-of-fold
probabilities and decisions, multiple metrics, train-validation gaps, error audit,
source snapshots and hashes, data/split hashes and environment versions.
The executed notebook and HTML verification preview are saved under outputs/amirhosein/.
All 79 repository tests pass; the complete notebook executed successfully top to bottom.

Source references: notebooks/aimilia/features_*.py and final.ipynb;
notebooks/leonard/ablation_extensions.py and progressive_features.py;
the five local 2025 reference notebooks, especially de-lucia-fratello-marchio.ipynb.
No external species database or target-derived win-rate encoding was added.
