"""Fold-fitted membership in the most frequently winning full P1 teams."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted


def canonical_team(battle):
    """Ignore roster order; define a team by its six species, not moves or items."""
    names = [str(p.get("name", "")).lower() for p in battle.get("p1_team_details", [])]
    return tuple(sorted(names)) if len(names) == 6 and all(names) else ()


def team_key_features(battle):
    """Transport a team key as an exact numeric value; never use it as a predictor."""
    team = canonical_team(battle)
    if not team:
        return {"team_key": np.nan}
    digest = hashlib.sha256(json.dumps(team).encode()).hexdigest()
    return {"team_key": int(digest[:13], 16)}  # 52 bits, exactly representable as float


class WinningTeamFlags(TransformerMixin, BaseEstimator):
    """Replace a roster key with top-2/5/6 membership, cross-fitted for training.

    Rank full teams by observed wins, breaking ties by win rate, then team key.
    fit_transform uses internal folds so each training row's flags are learned
    without its own outcome. transform uses rankings fitted on all training rows.
    Unseen or missing teams receive three zero flags.
    """

    def __init__(self, key_column="roster__team_key", top_n=(2, 5, 6), cv=5, random_state=42):
        self.key_column = key_column
        self.top_n = top_n
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame) or self.key_column not in X:
            raise ValueError("WinningTeamFlags requires a DataFrame with the roster key column")
        labels = np.asarray(y)
        if len(labels) != len(X) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Provide one binary training label per row")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        table = pd.DataFrame({"team_key": X[self.key_column].to_numpy(), "won": labels})
        table = table.dropna(subset=["team_key"]).groupby("team_key")["won"].agg(
            battles="size", wins="sum", win_rate="mean"
        ).reset_index()
        self.ranking_ = table[table.wins > 0].sort_values(
            ["wins", "win_rate", "team_key"], ascending=[False, False, True]
        ).reset_index(drop=True)
        self.top_teams_ = {n: set(self.ranking_.head(n).team_key) for n in self.top_n}
        return self

    def transform(self, X):
        check_is_fitted(self, "top_teams_")
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Training and prediction columns must match")
        output = X.drop(columns=self.key_column).copy()
        for n in self.top_n:
            output[f"p1_top_{n}_winning_team"] = X[self.key_column].isin(self.top_teams_[n]).astype(float)
        return output

    def fit_transform(self, X, y=None, **fit_params):
        labels = np.asarray(y)
        counts = pd.Series(labels).value_counts()
        folds = min(self.cv, int(counts.min())) if len(counts) == 2 else 0
        if folds < 2:
            raise ValueError("Cross-fitted team flags need at least two examples per class")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        self.fit(X, labels)
        output = self.transform(X)
        for training, validation in splitter.split(X, labels):
            encoder = clone(self).fit(X.iloc[training], labels[training])
            output.iloc[validation] = encoder.transform(X.iloc[validation]).to_numpy()
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        names = [name for name in self.feature_names_in_ if name != self.key_column]
        return np.asarray(names + [f"p1_top_{n}_winning_team" for n in self.top_n], dtype=object)
