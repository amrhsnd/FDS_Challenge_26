"""Notebook-facing API for comparable CV experiments and guarded final evaluation."""
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import json
from numbers import Real
from pathlib import Path
import platform
import re
import subprocess

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate

from .data import iter_battles
from .features.basic import team_stats, timeline_summary
from .models.classifiers import make_pipeline
from .paths import data_dir, output_dir, project_root
from .validation.splits import make_splits


@dataclass(frozen=True)
class Candidate:
    """One feature/model configuration. Each is evaluated on every fixed CV fold."""
    name: str
    groups: tuple = ("team",)
    model: str = "logistic_regression"
    percentile: int = 100
    params: dict = field(default_factory=dict)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use names containing letters, numbers, underscores, dots, and hyphens, starting with a letter/number")
    return name


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


class BattleExperiment:
    """Load numeric features once, compare via CV, then evaluate and fit for submission.

    Feature functions take one battle with its ID and target removed, and return a
    dictionary with fixed numeric keys. They must be stateless: learned transforms
    belong in the sklearn pipeline. The holdout is excluded from compare().
    """

    def __init__(self, owner, run, *, feature_functions=None, data_path=None, validation=None):
        self.data_path = Path(data_path) if data_path is not None else data_dir()
        config_path = project_root() / "configs/validation.json"
        self.validation = json.loads(config_path.read_text())
        self.validation.setdefault("holdout_size", 0.2)
        self.validation.update(validation or {})
        if self.validation.get("metric") != "accuracy" or self.validation.get("splitter") != "StratifiedKFold":
            raise ValueError("This runner supports accuracy and shuffled StratifiedKFold only")
        if self.validation.get("shuffle") is not True:
            raise ValueError("The fixed stratified folds use shuffle=true")
        self.feature_functions = dict(feature_functions) if feature_functions is not None else {"team": team_stats, "timeline": timeline_summary}
        if not self.feature_functions or any(not callable(f) for f in self.feature_functions.values()):
            raise ValueError("Provide at least one named callable feature function")
        for name in self.feature_functions: _safe_name(name)
        self._keys = {}
        rows, ids, labels = [], [], []
        for battle in iter_battles(self.data_path / "train.jsonl"):
            label = battle.get("player_won")
            if label not in (0, 1):
                raise ValueError("Every training battle must have a binary player_won label")
            rows.append(self._feature_row(battle, self.feature_functions))
            ids.append(battle["battle_id"])
            labels.append(int(label))
        self._check_ids(ids)
        self.X = pd.DataFrame(rows, dtype=float)
        self.y = pd.Series(labels, name="player_won")
        self.ids = pd.Index(ids, name="battle_id")
        if set(self.y.unique()) != {0, 1}:
            raise ValueError("Training data must contain both target classes")
        self.development, self._holdout, self.folds, self.assignments = make_splits(
            self.ids, self.y, seed=self.validation["seed"], n_splits=self.validation["n_splits"],
            holdout_size=self.validation["holdout_size"])
        self.output_path = output_dir(_safe_name(owner), _safe_name(run))
        if any(self.output_path.iterdir()):
            raise FileExistsError(f"Run folder is not empty; choose a new run name: {self.output_path}")
        self.assignments.to_csv(self.output_path / "splits.csv", index=False)
        self._candidates = {}
        self._results = []
        self._locked_candidate = None
        self._holdout_result = None
        self._dummy_accuracy = None
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root(), text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        sources = {}
        for name, func in self.feature_functions.items():
            try: sources[name] = inspect.getsource(func)
            except (OSError, TypeError): sources[name] = "Source unavailable: retain the defining notebook or module."
        _write_json(self.output_path / "feature_sources.json", sources)
        self.manifest = {
            "owner": owner, "run": run, "validation": self.validation,
            "rows": len(self.y), "development_rows": len(self.development), "holdout_rows": len(self._holdout),
            "feature_columns": list(self.X), "git_commit": commit, "working_tree_dirty": dirty,
            "train_sha256": _digest(self.data_path / "train.jsonl"),
            "splits_sha256": _digest(self.output_path / "splits.csv"),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        }
        _write_json(self.output_path / "experiment.json", self.manifest)
        print(f"{len(self.y):,} battles | {len(self.development):,} development | {len(self._holdout):,} holdout | {len(self.folds)} folds")
        print(f"Artifacts: {self.output_path}")

    @staticmethod
    def _check_ids(ids):
        index = pd.Index(ids)
        if len(index) == 0 or index.has_duplicates or index.isna().any():
            raise ValueError("Each file needs nonmissing, unique battle IDs and at least one battle")

    def _feature_row(self, battle, groups):
        observed = {k: v for k, v in battle.items() if k not in {"player_won", "battle_id"}}
        row = {}
        for group in groups:
            features = self.feature_functions[group](observed)
            if not isinstance(features, dict) or not features or any(not isinstance(k, str) or not k for k in features):
                raise ValueError(f"Feature group {group!r} must return a nonempty dictionary with string keys")
            keys = sorted(features)
            if group not in self._keys: self._keys[group] = keys
            if keys != self._keys[group]:
                raise ValueError(f"Group {group!r} returned different feature keys across battles; use a fixed schema")
            for key in keys:
                value = features[key]
                if value is not None and not isinstance(value, (Real, np.bool_)):
                    raise ValueError(f"{group}.{key} is not numeric; encode categories inside a fold-fitted pipeline first")
                value = np.nan if value is None else float(value)
                if np.isinf(value): raise ValueError(f"Infinite value in {group}.{key}")
                row[f"{group}__{key}"] = value
        return row

    def _columns(self, candidate):
        if not candidate.groups or len(set(candidate.groups)) != len(candidate.groups):
            raise ValueError("Each candidate needs distinct feature groups")
        unknown = set(candidate.groups) - set(self.feature_functions)
        if unknown: raise ValueError(f"Unknown feature groups: {unknown}")
        return [f"{g}__{k}" for g in candidate.groups for k in self._keys[g]]

    def _pipeline(self, candidate):
        return make_pipeline(candidate.model, seed=self.validation["seed"], percentile=candidate.percentile, params=candidate.params)

    def compare(self, candidates):
        """Evaluate configurations on development folds only; append new named candidates."""
        if self._locked_candidate is not None:
            raise RuntimeError("Holdout has been opened: stop tuning this experiment")
        # Snapshot mutable parameter dictionaries so later notebook edits cannot
        # change the configuration associated with an already measured score.
        candidates = [Candidate(**asdict(candidate)) for candidate in candidates]
        names = [_safe_name(c.name) for c in candidates]
        if not names or len(names) != len(set(names)) or set(names) & set(self._candidates):
            raise ValueError("Provide nonempty candidates with new, unique names")
        X_dev, y_dev = self.X.iloc[self.development], self.y.iloc[self.development]
        if self._dummy_accuracy is None:
            self._dummy_accuracy = float(cross_validate(DummyClassifier(strategy="most_frequent"), X_dev, y_dev,
                                                       cv=self.folds, scoring="accuracy", error_score="raise")["test_score"].mean())
        for candidate in candidates:
            columns = self._columns(candidate)
            print(f"Validating {candidate.name} on {len(self.folds)} folds...", flush=True)
            scores = cross_validate(self._pipeline(candidate), X_dev[columns], y_dev, cv=self.folds,
                                    scoring="accuracy", n_jobs=1, error_score="raise", return_estimator=True)
            selected = [list(estimator[:-1].get_feature_names_out(columns)) for estimator in scores["estimator"]]
            selection = pd.DataFrame({"feature": columns,
                "selected_in_folds": [sum(column in names for names in selected) for column in columns]})
            selection.to_csv(self.output_path / f"selection_frequency_{candidate.name}.csv", index=False)
            self._candidates[candidate.name] = candidate
            result = {"candidate": candidate.name, "model": candidate.model, "groups": ", ".join(candidate.groups),
                      "input_features": len(columns), "percentile": candidate.percentile,
                      "mean_selected_features": float(np.mean([len(names) for names in selected])),
                      "cv_accuracy": float(scores["test_score"].mean()), "fold_std": float(scores["test_score"].std()),
                      "mean_fit_seconds": float(scores["fit_time"].mean()), "dummy_accuracy": self._dummy_accuracy}
            result.update({f"fold_{i+1}": float(score) for i, score in enumerate(scores["test_score"])})
            self._results.append(result)
            self.results.to_csv(self.output_path / "cv_results.csv", index=False)
            _write_json(self.output_path / "candidates.json", [asdict(c) for c in self._candidates.values()])
        return self.results

    @property
    def results(self):
        return pd.DataFrame(self._results).sort_values("cv_accuracy", ascending=False, kind="stable").reset_index(drop=True) if self._results else pd.DataFrame()

    @property
    def best_candidate(self):
        if not self._results: raise RuntimeError("Call compare() first")
        return self.results.iloc[0]["candidate"]

    def evaluate_holdout(self, candidate=None):
        """One chosen model, fitted on development data; lock further comparisons."""
        name = candidate or self.best_candidate
        if self._locked_candidate is not None:
            if name != self._locked_candidate: raise RuntimeError("Holdout already evaluated for another candidate")
            return self._holdout_result
        if name not in self._candidates: raise ValueError("Choose a candidate evaluated by compare()")
        config = self._candidates[name]
        columns = self._columns(config)
        model = self._pipeline(config)
        model.fit(self.X.iloc[self.development][columns], self.y.iloc[self.development])
        predictions = model.predict(self.X.iloc[self._holdout][columns])
        truth = self.y.iloc[self._holdout]
        # Lock before exposing the result; later compare calls are rejected.
        self._locked_candidate = name
        self._holdout_result = {"candidate": name, "accuracy": float(accuracy_score(truth, predictions)),
            "selected_features": list(model[:-1].get_feature_names_out(columns)),
            "n_battles": len(truth), "labels": [0, 1], "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
            "classification_report": classification_report(truth, predictions, labels=[0, 1], output_dict=True, zero_division=0)}
        _write_json(self.output_path / "holdout_metrics.json", self._holdout_result)
        pd.DataFrame({"battle_id": self.ids[self._holdout], "actual": np.asarray(truth), "predicted": predictions}).to_csv(self.output_path / "holdout_predictions.csv", index=False)
        return self._holdout_result

    def make_submission(self):
        """Refit the locked candidate on all labelled battles, then predict official test."""
        if self._locked_candidate is None: raise RuntimeError("Choose a model with CV, then call evaluate_holdout() first")
        config = self._candidates[self._locked_candidate]
        columns = self._columns(config)
        rows, ids = [], []
        for battle in iter_battles(self.data_path / "test.jsonl"):
            rows.append(self._feature_row(battle, config.groups))
            ids.append(battle["battle_id"])
        self._check_ids(ids)
        sample = pd.read_csv(self.data_path / "sample_submission.csv")
        if list(sample.columns) != ["battle_id", "player_won"] or sample["battle_id"].isna().any() or sample["battle_id"].duplicated().any():
            raise ValueError("Invalid sample submission format or IDs")
        if len(sample) != len(ids) or set(sample["battle_id"]) != set(ids):
            raise ValueError("Sample IDs must match test IDs exactly")
        model = self._pipeline(config)
        model.fit(self.X[columns], self.y)
        predictions = pd.Series(model.predict(pd.DataFrame(rows, dtype=float)[columns]), index=ids)
        submission = sample.copy()
        submission["player_won"] = submission["battle_id"].map(predictions).astype(int)
        if not submission["player_won"].isin([0, 1]).all(): raise ValueError("Nonbinary predictions")
        path = self.output_path / "submission.csv"
        submission.to_csv(path, index=False)
        joblib.dump(model, self.output_path / "model.joblib")
        final = {"candidate": asdict(config), "feature_columns": columns, "fit_rows": len(self.y),
                 "test_sha256": _digest(self.data_path / "test.jsonl"),
                 "sample_sha256": _digest(self.data_path / "sample_submission.csv"),
                 "submission_sha256": _digest(path), "test_rows": len(ids)}
        _write_json(self.output_path / "final_fit.json", final)
        print(f"Trained on all {len(self.y):,} labelled battles. Saved {len(submission):,} predictions to {path}")
        return submission
