"""Integration checks for named features and fold-local model reduction."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'notebooks/amirhosein')]
import modeling
import study
from feature_catalog import FEATURE_NAMES
from feature_selection import CorrelationUtilitySelector
from pokemon_challenge.data import iter_battles as read_battles
from pokemon_challenge.experiment import Candidate
from pokemon_challenge.validation.splits import make_splits
from team_features import _schema_fixture


class NamedFeatureLibraryTests(unittest.TestCase):
    def test_valid_observation_has_757_semantic_names_and_no_source_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            chart = Path(directory) / 'chart.csv'
            chart.write_text('attacking_type,defending_type,multiplier\nwater,fire,2\n')
            library = modeling.BattleFeatureLibrary(chart)
        battle = _schema_fixture()
        for pokemon, name in zip(battle['p1_team_details'],
                ('tyranitar', 'snorlax', 'starmie', 'skarmory', 'miltank', 'zapdos')):
            pokemon['name'] = name
        battle['p2_lead_details']['name'] = 'forretress'
        for turn in battle['battle_timeline']:
            turn['p1_pokemon_state']['name'] = 'tyranitar'
            turn['p2_pokemon_state']['name'] = 'forretress'
        original = deepcopy(battle)
        row = library(study.observed_battle(battle))
        self.assertEqual(len(row), 757)
        self.assertEqual(set(row), set(FEATURE_NAMES.values()))
        forbidden = ('aimilia', 'amilia', 'leonard', 'amir', 'selected', 'expanded_core', '__')
        for name, value in row.items():
            self.assertFalse(any(token in name for token in forbidden), name)
            self.assertRegex(name, r'^[a-z][a-z0-9_]*$')
            self.assertIsInstance(value, (int, float, np.number), name)
            self.assertFalse(math.isinf(value), name)
        self.assertEqual(battle, original)
        changed = deepcopy(battle)
        changed.update(battle_id='different', player_won=0, winner='unobserved')
        changed['battle_timeline'].reverse()
        future = deepcopy(changed['battle_timeline'][0])
        future['turn'] = 31
        future['p1_pokemon_state'].update(hp_pct=0, status='fnt')
        changed['battle_timeline'].append(future)
        actual = library(study.observed_battle(changed))
        np.testing.assert_allclose(list(row.values()), list(actual.values()), equal_nan=True)


class ReducedModelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.names = tuple(FEATURE_NAMES.values())[:10]
        random = np.random.default_rng(342)
        self.battles = []
        for group in range(160):
            values = random.normal(size=len(self.names))
            label = int(values[0] + .6 * values[1] + .15 * random.normal() > 0)
            roster = [{'name': 'Alpha', 'measurements': values.tolist()},
                      {'name': 'Beta', 'base_hp': 80}]
            for twin in range(2):
                self.battles.append({
                    'battle_id': f'{group}-{twin}', 'player_won': label,
                    'p1_team_details': roster if twin == 0 else list(reversed(roster)),
                    'p2_lead_details': {'name': 'Gamma', 'base_hp': 70},
                    'battle_timeline': [{'turn': 1,
                        'p1_pokemon_state': {'name': 'Alpha', 'hp_pct': 1.}}],
                    'private_outcome': label,
                })
        (self.data / 'train.jsonl').write_text(
            ''.join(json.dumps(battle) + '\n' for battle in self.battles))
        self.read_paths = []

        def numeric_library(chart_path):
            self.assertEqual(Path(chart_path), self.data / 'type_chart_gen2.csv')
            def values(battle):
                self.assertEqual(set(battle),
                    {'p1_team_details', 'p2_lead_details', 'battle_timeline'})
                pokemon = next(p for p in battle['p1_team_details'] if p['name'] == 'Alpha')
                return dict(zip(self.names, pokemon['measurements']))
            return values

        def monitored_reader(path):
            self.assertEqual(Path(path), self.data / 'train.jsonl')
            self.read_paths.append(Path(path))
            return read_battles(path)

        def destination(owner, run):
            folder = self.root / 'outputs' / owner / run
            folder.mkdir(parents=True, exist_ok=True)
            return folder

        for patcher in (
            patch.object(modeling, 'BattleFeatureLibrary', side_effect=numeric_library),
            patch('pokemon_challenge.experiment.output_dir', side_effect=destination),
            patch('pokemon_challenge.experiment.iter_battles', side_effect=monitored_reader),
            patch.object(study, 'iter_battles', side_effect=monitored_reader),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def experiment(self, run='one'):
        return modeling.ReducedBattleStudy(data_path=self.data, owner='synthetic', run=run,
            n_features=3, correlation_threshold=.95, validation={'n_splits': 4})

    def test_semantic_columns_grouped_folds_and_fold_local_selection(self):
        experiment = self.experiment()
        self.assertEqual(set(experiment.X), set(self.names))
        self.assertEqual(set(experiment._keys['battle']), set(self.names))
        self.assertTrue(all('__' not in column for column in experiment.X))
        _, holdout, _, _ = make_splits(experiment.ids, experiment.y, seed=42, n_splits=4)
        np.testing.assert_array_equal(experiment._holdout, holdout)
        holdout_hashes = set(experiment.observation_hashes[holdout])
        self.assertGreater(experiment.manifest['quarantined_rows'], 0)
        expected_fit_rows = []
        for train, valid in experiment.folds:
            training, validation = experiment.development[train], experiment.development[valid]
            expected_fit_rows.append(set(training))
            self.assertFalse(set(experiment.observation_hashes[training]) &
                             set(experiment.observation_hashes[validation]))
            self.assertFalse(set(experiment.observation_hashes[training]) & holdout_hashes)
        candidate = Candidate('linear', groups=('battle',), params={'C': .1})
        self.assertFalse(hasattr(experiment, 'feature_selector'))
        fit_rows = []
        original_fit = CorrelationUtilitySelector.fit
        def observed_fit(selector, X, y):
            fit_rows.append(set(X.index))
            return original_fit(selector, X, y)
        with patch.object(CorrelationUtilitySelector, 'fit', new=observed_fit):
            results = experiment.compare([candidate])
        self.assertEqual(fit_rows, expected_fit_rows)
        result = results.iloc[0]
        self.assertEqual(result.input_features, 3)
        self.assertEqual(result.candidate_features, len(self.names))
        self.assertEqual(result.mean_selected_features, 3)
        saved = pd.read_csv(experiment.output_path / 'oof_linear.csv')
        np.testing.assert_array_equal(saved.predicted, experiment.oof_predictions['linear'])
        np.testing.assert_array_equal(saved.actual,
                                      experiment.y.iloc[experiment.development].to_numpy())
        self.assertAlmostEqual(result.oof_accuracy,
                               accuracy_score(saved.actual, saved.predicted))
        train, valid = experiment.folds[0]
        pipeline = clone(experiment._pipeline(candidate)).set_params(memory=None)
        X = experiment.X.iloc[experiment.development]
        y = experiment.y.iloc[experiment.development].to_numpy()
        pipeline.fit(X.iloc[train], y[train])
        self.assertEqual(len(pipeline['select_features'].get_feature_names_out()), 3)
        self.assertEqual(pipeline['model'].n_features_in_, 3)
        np.testing.assert_allclose(saved.probability.iloc[valid],
                                   pipeline.predict_proba(X.iloc[valid])[:, 1])
        self.assertFalse(experiment.manifest['holdout_evaluated'])
        self.assertFalse(experiment.manifest['test_read'])
        with self.assertRaisesRegex(ValueError, 'battle feature group'):
            experiment._columns(Candidate('invalid', groups=('unknown',)))

    def test_development_feature_set_is_documented_without_opening_holdout(self):
        experiment = self.experiment()
        fit_rows = []
        original_fit = CorrelationUtilitySelector.fit
        def observed_fit(selector, X, y):
            fit_rows.append(set(X.index))
            return original_fit(selector, X, y)
        with patch.object(CorrelationUtilitySelector, 'fit', new=observed_fit):
            matrix = experiment.select_features()
        self.assertEqual(fit_rows, [set(experiment.development)])
        self.assertEqual(matrix.shape, (len(self.battles), 3))
        self.assertEqual(matrix.columns.tolist(),
                         experiment.feature_definitions.feature.tolist())
        self.assertEqual(set(experiment.feature_definitions),
                         {'feature', 'category', 'description', 'units'})
        self.assertTrue(experiment.feature_definitions.description.str.len().gt(10).all())
        saved = json.loads((experiment.output_path / 'feature_set.json').read_text())
        self.assertEqual(saved['features'], matrix.columns.tolist())
        self.assertEqual(saved['count'], 3)
        self.assertEqual(saved['selection_fit_rows'], len(experiment.development))
        self.assertEqual(saved['effective_correlation_threshold'], .95)
        self.assertFalse(experiment.manifest['holdout_evaluated'])
        self.assertFalse(experiment.manifest['test_read'])

    def test_soft_voting_fits_one_selector_shared_by_all_members(self):
        experiment = self.experiment()
        members = [Candidate('linear', groups=('battle',), params={'C': .1}),
                   Candidate('tree', groups=('battle',), model='hist_gradient_boosting',
                     params={'max_iter': 8, 'max_leaf_nodes': 5,
                             'min_samples_leaf': 8, 'early_stopping': False})]
        candidate = Candidate('vote', groups=('battle',), model='soft_voting',
                              params={'members': [asdict(member) for member in members]})
        model = experiment._pipeline(candidate).set_params(memory=None)
        self.assertEqual(list(model.named_steps), ['select_features', 'imputer', 'model'])
        X, y = experiment.X.iloc[experiment.development], experiment.y.iloc[experiment.development]
        fit_rows = []
        original_fit = CorrelationUtilitySelector.fit
        def observed_fit(selector, X, y):
            fit_rows.append(set(X.index))
            return original_fit(selector, X, y)
        with patch.object(CorrelationUtilitySelector, 'fit', new=observed_fit):
            model.fit(X, y)
        self.assertEqual(fit_rows, [set(experiment.development)])
        self.assertEqual(len(model[:-1].get_feature_names_out(X.columns)), 3)
        for member in model['model'].estimators_:
            self.assertNotIn('select_features', member.named_steps)
            self.assertEqual(member['model'].n_features_in_, 3)
        selected = model['imputer'].transform(model['select_features'].transform(X))
        expected = np.mean([member.predict_proba(selected)
                            for member in model['model'].estimators_], axis=0)
        np.testing.assert_allclose(model.predict_proba(X), expected)
        results = experiment.compare([candidate])
        self.assertEqual(results.iloc[0].input_features, 3)
        self.assertFalse(experiment.manifest['holdout_evaluated'])


if __name__ == '__main__':
    unittest.main()
