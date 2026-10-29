"""Deterministic battle-level holdout and development-fold assignments."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_splits(ids, y, *, seed=42, n_splits=5, holdout_size=0.2):
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size must be strictly between 0 and 1")
    development, holdout = train_test_split(np.arange(len(y)), test_size=holdout_size, stratify=y, random_state=seed)
    if pd.Series(y).iloc[development].value_counts().min() < n_splits:
        raise ValueError("Too few development examples per class for the requested folds")
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(development, np.asarray(y)[development]))
    assignments = pd.DataFrame({"battle_id": list(ids), "partition": "holdout", "validation_fold": -1})
    assignments.loc[development, "partition"] = "development"
    for fold, (_, validation) in enumerate(folds):
        assignments.loc[development[validation], "validation_fold"] = fold
    return development, holdout, folds, assignments
