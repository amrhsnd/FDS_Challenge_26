"""Every candidate uses fold-fitted imputation, selection, and preprocessing."""
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(model="logistic_regression", *, seed=42, percentile=100, params=None):
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be greater than 0 and at most 100")
    defaults = {
        "logistic_regression": (LogisticRegression, {"max_iter": 2000, "random_state": seed}),
        "random_forest": (RandomForestClassifier, {"n_estimators": 150, "n_jobs": 1, "random_state": seed}),
        "hist_gradient_boosting": (HistGradientBoostingClassifier, {"random_state": seed}),
    }
    if model not in defaults:
        raise ValueError(f"Unknown model {model!r}; choose one of {list(defaults)}")
    constructor, settings = defaults[model]
    settings = {**settings, **(params or {})}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("constant_filter", VarianceThreshold()),
        ("selector", SelectPercentile(f_classif, percentile=percentile) if percentile < 100 else "passthrough"),
        ("scaler", StandardScaler() if model == "logistic_regression" else "passthrough"),
        ("model", constructor(**settings)),
    ])
