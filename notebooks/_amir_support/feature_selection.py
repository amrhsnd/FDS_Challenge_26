"""Select a compact, useful, weakly redundant numeric feature matrix.

All decisions are fitted from the rows supplied to ``fit``. Put this transformer
inside a model pipeline so validation labels/values cannot determine selection.
Unknown values are median-filled only for selection; ``transform`` returns the
original selected values for the model's own training-fitted imputer.
"""
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


class CorrelationUtilitySelector(TransformerMixin, BaseEstimator):
    """Keep exactly ``n_features`` useful columns with bounded pair correlation.

    Remove all-missing/constant columns and exact duplicates after training-median
    imputation. Rank the remaining columns by a weighted mean of percentile ranks:
    regularized logistic coefficient magnitude (40%), shallow ExtraTrees impurity
    importance (35%), and univariate classification F score (25%). The mix supports
    both linear and nonlinear signal; it is a selection heuristic, not causal
    importance. Greedily keep the highest-ranked column whose absolute Pearson
    correlation with every retained column is at most ``correlation_threshold``.

    Raises when the requested count cannot be obtained without weakening the
    correlation limit. It never silently restores duplicate or constant columns.
    """
    def __init__(self, n_features=300, correlation_threshold=.90, random_state=42,
                 n_estimators=64):
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.n_estimators = n_estimators

    @staticmethod
    def _values(X):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            raise ValueError('Provide a nonempty two-dimensional numeric feature matrix.')
        if np.isinf(values).any():
            raise ValueError('Infinite feature values are not supported; use NaN for unknowns.')
        return values

    def fit(self, X, y):
        if isinstance(self.n_features, bool) or not isinstance(self.n_features, Integral) or self.n_features < 1:
            raise ValueError('n_features must be a positive integer.')
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError('correlation_threshold must lie in (0, 1].')
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValueError('n_estimators must be a positive integer.')
        values = self._values(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(values) or set(np.unique(target)) != {0, 1}:
            raise ValueError('Selection requires one binary 0/1 target per training row and both classes.')
        if isinstance(X, pd.DataFrame):
            if X.columns.has_duplicates or any(not isinstance(name, str) for name in X.columns):
                raise ValueError('Feature columns must have unique string names.')
            names = X.columns.to_numpy(dtype=object)
        else:
            names = np.asarray([f'x{i}' for i in range(values.shape[1])], dtype=object)
        if set(names) & {'battle_id', 'player_won', 'target', 'label', 'actual', 'predicted'}:
            raise ValueError('Identifiers and target/prediction columns must not enter feature selection.')
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = names.copy()
        self.columns_ = pd.Index(names)
        self.effective_threshold_ = float(self.correlation_threshold)
        missing = np.isnan(values)
        missing_fraction = missing.mean(axis=0)
        all_missing = missing.all(axis=0)
        medians = np.zeros(self.n_features_in_, dtype=float)
        medians[~all_missing] = np.nanmedian(values[:, ~all_missing], axis=0)
        filled = np.where(missing, medians, values)
        constant = (np.ptp(filled, axis=0) == 0) & ~all_missing
        valid = np.flatnonzero(~(all_missing | constant))

        # Prefer the least-missing version of an identical training vector.
        # Stable semantic names break ties without depending on input column order.
        representatives, duplicate_of, fingerprints = [], {}, {}
        for index in sorted(valid, key=lambda i: (missing_fraction[i], names[i])):
            vector = np.ascontiguousarray(filled[:, index])
            vector[vector == 0] = 0  # Canonicalize signed zero before hashing.
            digest = hashlib.blake2b(vector.tobytes(), digest_size=16).digest()
            matches = fingerprints.setdefault(digest, [])
            duplicate = next((other for other in matches
                              if np.array_equal(vector, filled[:, other])), None)
            if duplicate is not None:
                duplicate_of[index] = duplicate
            else:
                matches.append(index)
                representatives.append(index)
        representatives = np.asarray(sorted(representatives), dtype=int)
        if len(representatives) < self.n_features:
            raise ValueError(f'Only {len(representatives)} nonconstant, nonduplicate features are available; '
                             f'cannot retain exactly {self.n_features}.')
        matrix = filled[:, representatives]
        standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        with threadpool_limits(limits=2):
            f_score = f_classif(standardized, target)[0]
            linear = LogisticRegression(C=.05, max_iter=1500, tol=1e-5,
                                        random_state=self.random_state).fit(standardized, target)
            nonlinear = ExtraTreesClassifier(n_estimators=self.n_estimators,
                max_depth=8, min_samples_leaf=10, max_features='sqrt', n_jobs=1,
                random_state=self.random_state).fit(matrix, target)
            correlation = np.clip(standardized.T @ standardized / len(matrix), -1., 1.)
        coefficient = np.abs(linear.coef_[0])
        tree_importance = nonlinear.feature_importances_
        f_score = np.nan_to_num(f_score, nan=0., posinf=np.finfo(float).max)
        rank = lambda scores: rankdata(scores, method='average') / len(scores)
        utility = .4 * rank(coefficient) + .35 * rank(tree_importance) + .25 * rank(f_score)
        order = sorted(range(len(representatives)),
                       key=lambda i: (-utility[i], missing_fraction[representatives[i]], names[representatives[i]]))
        selected = []
        for candidate in order:
            if not selected or np.abs(correlation[candidate, selected]).max() <= self.correlation_threshold + 1e-12:
                selected.append(candidate)
                if len(selected) == self.n_features:
                    break
        if len(selected) < self.n_features:
            raise ValueError(f'Correlation limit {self.correlation_threshold:.3f} leaves only '
                f'{len(selected)} eligible features; cannot retain {self.n_features}. '
                'Increase the correlation limit or reduce n_features explicitly.')

        # Return columns in their input order, as sklearn feature selectors do;
        # utility ranking is retained separately in the audit table.
        selected.sort(key=lambda position: representatives[position])
        self.selected_indices_ = representatives[np.asarray(selected)]
        self.selected_features_ = names[self.selected_indices_].copy()
        self.n_features_out_ = len(self.selected_features_)
        self.selection_medians_ = pd.Series(medians, index=names, name='training_median')
        self.scores_ = pd.Series(utility, index=names[representatives], name='utility_score')
        selected_correlation = correlation[np.ix_(selected, selected)].copy()
        np.fill_diagonal(selected_correlation, 0)
        self.maximum_selected_correlation_ = float(np.abs(selected_correlation).max())
        self.selected_correlation_ = pd.DataFrame(correlation[np.ix_(selected, selected)],
                                                 index=self.selected_features_, columns=self.selected_features_)

        representative_position = {column: position for position, column in enumerate(representatives)}
        utility_rank = {position: rank + 1 for rank, position in enumerate(order)}
        selected_set = set(selected)
        rows = []
        for index, name in enumerate(names):
            row = {'feature': name, 'selected': False, 'reason': None,
                   'utility_score': np.nan, 'utility_rank': np.nan,
                   'missing_fraction': float(missing_fraction[index]),
                   'related_feature': None, 'absolute_correlation': np.nan,
                   'linear_coefficient': np.nan, 'tree_importance': np.nan, 'f_score': np.nan}
            if all_missing[index]:
                row['reason'] = 'all_missing'
            elif constant[index]:
                row['reason'] = 'constant'
            elif index in duplicate_of:
                row.update(reason='duplicate', related_feature=names[duplicate_of[index]], absolute_correlation=1.)
            else:
                position = representative_position[index]
                comparisons = np.abs(correlation[position, selected]).copy()
                if position in selected_set:
                    comparisons[selected.index(position)] = -1.
                strongest = int(np.argmax(comparisons))
                maximum = max(0., float(comparisons[strongest]))
                chosen = position in selected_set
                row.update(selected=chosen,
                    reason='selected' if chosen else ('correlated' if maximum > self.correlation_threshold + 1e-12 else 'lower_utility'),
                    utility_score=float(utility[position]), utility_rank=utility_rank[position],
                    related_feature=names[representatives[selected[strongest]]] if maximum > 0 else None,
                    absolute_correlation=maximum, linear_coefficient=float(coefficient[position]),
                    tree_importance=float(tree_importance[position]), f_score=float(f_score[position]))
            rows.append(row)
        self.selection_audit_ = pd.DataFrame(rows)
        self.audit_ = self.selection_audit_
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_indices_')
        values = self._values(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError('Transform input must have the same feature count as training input.')
        if isinstance(X, pd.DataFrame):
            if not np.array_equal(X.columns.to_numpy(), self.feature_names_in_):
                raise ValueError('Transform feature names and order must match training input.')
            return X.loc[:, self.selected_features_].copy()
        return values[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'selected_features_')
        if input_features is not None and not np.array_equal(input_features, self.feature_names_in_):
            raise ValueError('input_features must match the fitted feature names and order.')
        return self.selected_features_.copy()

    def get_support(self, indices=False):
        check_is_fitted(self, 'selected_indices_')
        if indices:
            return self.selected_indices_.copy()
        selected = np.zeros(self.n_features_in_, dtype=bool)
        selected[self.selected_indices_] = True
        return selected
