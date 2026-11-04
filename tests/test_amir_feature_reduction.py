"""Synthetic checks for useful, fold-fitted correlation reduction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('feature_reduction_under_test',
    ROOT / 'notebooks/amirhosein/feature_selection.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
CorrelationUtilitySelector = module.CorrelationUtilitySelector


def synthetic_features(rows=500):
    rng = np.random.default_rng(21)
    signal = rng.normal(size=rows)
    second = rng.normal(size=rows)
    X = pd.DataFrame({'signal': signal, 'signal_duplicate': signal,
        'opposite_signal': -signal, 'almost_signal': signal + rng.normal(scale=.001, size=rows),
        'secondary': second, 'constant': 1., 'unknown': np.nan,
        **{f'noise_{i}': rng.normal(size=rows) for i in range(10)}})
    y = (2.2*signal + second + rng.normal(scale=.4, size=rows) > 0).astype(int)
    return X, y


class FeatureReductionTests(unittest.TestCase):
    def test_redundant_and_uninformative_columns_are_removed_with_audit(self):
        X, y = synthetic_features()
        selector = CorrelationUtilitySelector(n_features=4, correlation_threshold=.9, n_estimators=32).fit(X, y)
        selected = set(selector.selected_features_)
        signal_family = {'signal', 'signal_duplicate', 'opposite_signal', 'almost_signal'}
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(selected & signal_family), 1)
        self.assertIn('secondary', selected)
        self.assertFalse(selected & {'constant', 'unknown'})
        self.assertLessEqual(selector.maximum_selected_correlation_, .9 + 1e-12)
        audit = selector.selection_audit_.set_index('feature')
        self.assertEqual(audit.loc['unknown', 'reason'], 'all_missing')
        self.assertEqual(audit.loc['constant', 'reason'], 'constant')
        self.assertEqual(audit.loc['signal_duplicate', 'reason'], 'duplicate')
        correlated = audit[audit.reason == 'correlated']
        self.assertGreaterEqual(len(correlated), 2)
        self.assertTrue((correlated.absolute_correlation > .9).all())
        self.assertTrue((correlated.related_feature.isin(selected)).all())
        self.assertTrue((audit[audit.reason == 'lower_utility'].utility_score <=
                         audit[audit.selected].utility_score.max()).all())

    def test_training_fitted_selection_does_not_change_on_validation_rows(self):
        X, y = synthetic_features()
        X.loc[:15, 'secondary'] = np.nan
        selector = CorrelationUtilitySelector(n_features=5, random_state=7, n_estimators=24)
        training, validation = X.iloc[:400], X.iloc[400:].copy()
        fitted = selector.fit(training, y[:400])
        features, medians, audit = (fitted.selected_features_.copy(), fitted.selection_medians_.copy(),
                                    fitted.selection_audit_.copy())
        validation.loc[:, :] = 1_000_000
        transformed = fitted.transform(validation)
        self.assertEqual(transformed.shape, (100, 5))
        self.assertEqual(transformed.index.tolist(), validation.index.tolist())
        np.testing.assert_array_equal(fitted.selected_features_, features)
        pd.testing.assert_series_equal(fitted.selection_medians_, medians)
        pd.testing.assert_frame_equal(fitted.selection_audit_, audit)
        repeat = clone(selector).fit(training, y[:400])
        np.testing.assert_array_equal(repeat.selected_features_, features)
        pd.testing.assert_frame_equal(repeat.selection_audit_, audit)
        self.assertAlmostEqual(medians['secondary'], training.secondary.median())
        self.assertNotAlmostEqual(medians['secondary'], pd.concat([training, validation]).secondary.median())
        # The transform keeps missing observations; it does not apply its scoring imputer.
        reduced_training = fitted.transform(training)
        if 'secondary' in reduced_training:
            self.assertTrue(reduced_training.secondary.iloc[:16].isna().all())

    def test_pipeline_selection_is_fitted_on_training_rows_and_tracks_names(self):
        X, y = synthetic_features()
        pipeline = Pipeline([
            ('selection', CorrelationUtilitySelector(n_features=6, n_estimators=24)),
            ('imputer', SimpleImputer()), ('model', LogisticRegression(max_iter=1000))])
        pipeline.fit(X.iloc[:350], y[:350])
        names = pipeline[:-1].get_feature_names_out(X.columns)
        np.testing.assert_array_equal(names, pipeline['selection'].selected_features_)
        self.assertEqual(len(names), 6)
        self.assertEqual(pipeline.predict_proba(X.iloc[350:]).shape, (150, 2))
        self.assertEqual(pipeline['selection'].get_support().sum(), 6)
        np.testing.assert_array_equal(X.columns[pipeline['selection'].get_support(indices=True)], names)
        np.testing.assert_array_equal(X.columns[pipeline['selection'].get_support()], names)

    def test_ambiguous_schema_and_label_columns_are_rejected(self):
        X, y = synthetic_features(100)
        selector = CorrelationUtilitySelector(n_features=3, n_estimators=8)
        for reserved in ('battle_id', 'player_won', 'target', 'actual'):
            with self.assertRaisesRegex(ValueError, 'Identifiers'):
                selector.fit(X.assign(**{reserved: y}), y)
        fitted = selector.fit(X, y)
        with self.assertRaisesRegex(ValueError, 'names and order'):
            fitted.transform(X[X.columns[::-1]])
        with self.assertRaisesRegex(ValueError, 'feature count'):
            fitted.transform(X.iloc[:, :-1])
        with self.assertRaisesRegex(ValueError, 'input_features'):
            fitted.get_feature_names_out(list(reversed(X.columns)))
        with self.assertRaisesRegex(ValueError, 'unique'):
            selector.fit(pd.concat([X, X[['signal']]], axis=1), y)

    def test_impossible_count_and_correlation_limits_never_restore_bad_columns(self):
        rng = np.random.default_rng(4)
        signal = rng.normal(size=100)
        y = (signal > 0).astype(int)
        duplicated = pd.DataFrame({'signal': signal, 'same': signal, 'empty': np.nan, 'constant': 1})
        with self.assertRaisesRegex(ValueError, 'nonconstant, nonduplicate'):
            CorrelationUtilitySelector(n_features=2).fit(duplicated, y)
        correlated = pd.DataFrame({'signal': signal, 'inverse': -signal,
                                   'near': signal + rng.normal(scale=.001, size=100)})
        with self.assertRaisesRegex(ValueError, 'Correlation limit'):
            CorrelationUtilitySelector(n_features=2, correlation_threshold=.95).fit(correlated, y)
        for target in (np.zeros(100), np.full(100, 2), y[:-1]):
            with self.assertRaisesRegex(ValueError, 'binary'):
                CorrelationUtilitySelector(n_features=1).fit(duplicated, target)
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            CorrelationUtilitySelector(n_features=1).fit(duplicated.assign(invalid=np.inf), y)

    def test_numpy_input_and_single_output_keep_exact_requested_dimension(self):
        X, y = synthetic_features(100)
        selector = CorrelationUtilitySelector(n_features=1, n_estimators=8).fit(X.to_numpy(), y)
        self.assertEqual(selector.transform(X.to_numpy()).shape, (100, 1))
        self.assertEqual(len(selector.get_feature_names_out()), 1)
        self.assertEqual(selector.maximum_selected_correlation_, 0.)


if __name__ == '__main__':
    unittest.main()
