"""Synthetic checks for Amir's observation boundary and grouped evaluation.

These tests create their own labelled battles; they never open competition data.
"""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pokemon_challenge.data import iter_battles as read_battles
from pokemon_challenge.experiment import Candidate
from pokemon_challenge.validation.splits import make_splits

SPEC = importlib.util.spec_from_file_location('amir_study_under_test', ROOT / 'notebooks/_amir_support/study.py')
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class FixedTieClassifier(ClassifierMixin, BaseEstimator):
    """Exercise two valid class-decision policies at equal probabilities."""
    def __init__(self, prediction=0):
        self.prediction = prediction

    def fit(self, X, y):
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), .5)

    def predict(self, X):
        return np.full(len(X), self.prediction, dtype=int)


def example_battle():
    return {
        'battle_id': 'original', 'player_won': 1,
        'p1_team_details': [{'name': 'Alpha', 'base_hp': 70}, {'name': 'Beta', 'base_hp': 90}],
        'p2_lead_details': {'name': 'Gamma', 'base_hp': 80},
        'battle_timeline': [
            {'turn': 2, 'p1_pokemon_state': {'name': 'Alpha', 'hp_pct': .8}},
            {'turn': 1, 'p1_pokemon_state': {'name': 'Alpha', 'hp_pct': 1.0}},
            {'turn': 30, 'p2_pokemon_state': {'name': 'Gamma', 'status': 'fnt', 'hp_pct': .2}},
            {'turn': 31, 'p1_pokemon_state': {'name': 'Alpha', 'hp_pct': 0.0}},
        ],
        'winner_after_turn_30': 'private', 'unobserved_p2_roster': ['secret'],
    }


class ObservationBoundaryTests(unittest.TestCase):
    def test_whitelist_turn_window_sort_and_faint_normalization(self):
        battle = example_battle()
        original = deepcopy(battle)
        result = study.observed_battle(battle)
        self.assertEqual(set(result), {'p1_team_details', 'p2_lead_details', 'battle_timeline'})
        self.assertEqual([t['turn'] for t in result['battle_timeline']], [1, 2, 30])
        self.assertEqual(result['battle_timeline'][-1]['p2_pokemon_state']['hp_pct'], 0.0)
        self.assertEqual(battle, original)
        result['p1_team_details'][0]['name'] = 'changed'
        result['battle_timeline'][0]['p1_pokemon_state']['hp_pct'] = .1
        self.assertEqual(battle, original)

    def test_hash_ignores_ids_labels_hidden_fields_late_turns_and_order(self):
        battle = example_battle()
        reference = study.observation_digest(battle)
        changed = deepcopy(battle)
        changed['battle_id'], changed['player_won'] = 'another', 0
        changed['winner_after_turn_30'] = 'different private result'
        changed['unobserved_p2_roster'] = ['another secret']
        changed['p1_team_details'].reverse()
        changed['battle_timeline'].reverse()
        next(t for t in changed['battle_timeline'] if t['turn'] == 31)['p1_pokemon_state']['hp_pct'] = .9
        changed['battle_timeline'].extend([{'turn': 0}, {'turn': 32}, {'turn': '20'}, {}])
        original = deepcopy(changed)
        self.assertEqual(study.observation_digest(changed), reference)
        self.assertEqual(changed, original)
        self.assertEqual(study.observed_battle(changed)['p1_team_details'], changed['p1_team_details'])
        changed['battle_timeline'][next(i for i, t in enumerate(changed['battle_timeline']) if t.get('turn') == 2)]['p1_pokemon_state']['hp_pct'] = .7
        self.assertNotEqual(study.observation_digest(changed), reference)

    def test_hash_tracks_permitted_roster_and_lead_changes(self):
        battle = example_battle()
        reference = study.observation_digest(battle)
        for key in ('p1_team_details', 'p2_lead_details'):
            changed = deepcopy(battle)
            target = changed[key][0] if isinstance(changed[key], list) else changed[key]
            target['base_hp'] += 1
            self.assertNotEqual(study.observation_digest(changed), reference)

    def test_duplicate_turns_are_rejected_and_nonfinite_inputs_are_not_hashed(self):
        battle = example_battle()
        battle['battle_timeline'].append({'turn': 2})
        with self.assertRaisesRegex(ValueError, 'Duplicate turn'):
            study.observed_battle(battle)
        battle = example_battle()
        battle['p2_lead_details']['base_hp'] = float('nan')
        with self.assertRaises(ValueError):
            study.observation_digest(battle)


class AmirStudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.battles = []
        for group in range(80):
            label = group % 2
            roster = [
                {'name': 'Alpha', 'base_hp': 50 + 20 * label + group % 7, 'base_def': group},
                {'name': 'Beta', 'base_hp': 80, 'base_def': 30},
            ]
            for twin in range(2):
                self.battles.append({
                    'battle_id': f'battle-{group}-{twin}', 'player_won': label,
                    'p1_team_details': roster if twin == 0 else list(reversed(roster)),
                    'p2_lead_details': {'name': 'Gamma', 'base_hp': 65},
                    'battle_timeline': [{'turn': 1, 'p1_pokemon_state': {'name': 'Alpha', 'hp_pct': 1.0}}],
                    'private': label,
                })
        (self.data / 'train.jsonl').write_text(''.join(json.dumps(b) + '\n' for b in self.battles))
        self.read_paths = []
        def monitored_reader(path):
            self.assertEqual(Path(path).name, 'train.jsonl', 'Tests must not read official or synthetic test data.')
            self.read_paths.append(Path(path))
            return read_battles(path)
        def destination(owner, run):
            result = self.root / 'outputs' / owner / run
            result.mkdir(parents=True, exist_ok=True)
            return result
        for target, replacement in [
            ('pokemon_challenge.experiment.output_dir', destination),
            ('pokemon_challenge.experiment.iter_battles', monitored_reader),
            ('amir_study_under_test.iter_battles', monitored_reader),
        ]:
            # The dynamically imported module is not installed in sys.modules.
            if target.startswith('amir_study_under_test'):
                patcher = patch.object(study, 'iter_battles', side_effect=replacement)
            else:
                patcher = patch(target, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def experiment(self, run='one'):
        def core(battle):
            self.assertEqual(set(battle), {'p1_team_details', 'p2_lead_details', 'battle_timeline'})
            self.assertNotIn('battle_id', battle)
            alpha = next(p for p in battle['p1_team_details'] if p['name'] == 'Alpha')
            return {'signal': alpha['base_hp']}
        def extra(battle):
            alpha = next(p for p in battle['p1_team_details'] if p['name'] == 'Alpha')
            return {'marker': alpha['base_def']}
        return study.AmirStudy('synthetic', run, data_path=self.data,
            feature_functions={'core': core, 'extra': extra}, validation={'n_splits': 4})

    def test_duplicate_groups_and_holdout_quarantine_preserve_shared_holdout(self):
        source_config = (ROOT / 'configs/validation.json').read_bytes()
        experiment = self.experiment()
        _, expected_holdout, _, _ = make_splits(experiment.ids, experiment.y, seed=42, n_splits=4)
        np.testing.assert_array_equal(experiment._holdout, expected_holdout)
        holdout_hashes = set(experiment.observation_hashes[expected_holdout])
        self.assertGreater(experiment.manifest['quarantined_rows'], 0)
        self.assertEqual(experiment.manifest['duplicate_rows'], len(self.battles))
        development = set(experiment.development)
        self.assertFalse(development & set(expected_holdout))
        self.assertFalse(set(experiment.observation_hashes[experiment.development]) & holdout_hashes)
        expected_quarantine = {i for i, h in enumerate(experiment.observation_hashes)
                               if h in holdout_hashes and i not in expected_holdout}
        actual_quarantine = set(experiment.assignments.index[
            experiment.assignments.partition == 'holdout_duplicate_quarantine'])
        self.assertEqual(actual_quarantine, expected_quarantine)
        self.assertEqual(len(development) + len(expected_holdout) + len(actual_quarantine), len(self.battles))
        seen = []
        for fold, (training, validation) in enumerate(experiment.folds):
            train_rows, valid_rows = experiment.development[training], experiment.development[validation]
            self.assertFalse(set(train_rows) & set(valid_rows))
            self.assertFalse(set(experiment.observation_hashes[train_rows]) & set(experiment.observation_hashes[valid_rows]))
            self.assertTrue((experiment.assignments.loc[valid_rows, 'validation_fold'] == fold).all())
            seen.extend(validation)
        self.assertEqual(sorted(seen), list(range(len(development))))
        saved = pd.read_csv(experiment.output_path / 'splits.csv')
        pd.testing.assert_frame_equal(saved, experiment.assignments)
        repeat = self.experiment('repeat')
        pd.testing.assert_frame_equal(experiment.assignments, repeat.assignments)
        self.assertFalse(experiment.manifest['holdout_evaluated'])
        self.assertFalse(experiment.manifest['test_read'])
        self.assertFalse((experiment.output_path / 'holdout_metrics.json').exists())
        self.assertEqual(experiment.validation['splitter'], 'StratifiedGroupKFold')
        manifest = json.loads((experiment.output_path / 'experiment.json').read_text())
        self.assertEqual(manifest['validation']['splitter'], 'StratifiedGroupKFold')
        self.assertEqual(manifest['source_validation']['splitter'], 'StratifiedKFold')
        self.assertEqual(manifest['source_validation']['n_splits'], 4)
        self.assertEqual((ROOT / 'configs/validation.json').read_bytes(), source_config)

    def test_oof_probabilities_metrics_and_paired_comparison_align_with_folds(self):
        experiment = self.experiment()
        params = {'C': .5}
        baseline = Candidate('baseline', groups=('extra',), params=params)
        expanded = Candidate('expanded', groups=('core', 'extra'), params={'C': .5})
        folds_before = [(t.copy(), v.copy()) for t, v in experiment.folds]
        experiment.compare([baseline, expanded])
        params['C'] = 999
        self.assertEqual(experiment._candidates['baseline'].params['C'], .5)
        truth = experiment.y.iloc[experiment.development].to_numpy()
        scores = experiment.results.set_index('candidate')
        for candidate in (experiment._candidates['baseline'], expanded):
            saved = pd.read_csv(experiment.output_path / f'oof_{candidate.name}.csv')
            self.assertEqual(saved.battle_id.tolist(), experiment.ids[experiment.development].tolist())
            np.testing.assert_array_equal(saved.actual, truth)
            np.testing.assert_array_equal(saved.fold, experiment.assignments.loc[experiment.development, 'validation_fold'])
            columns = experiment._columns(candidate)
            expected = np.full(len(truth), np.nan)
            expected_predictions = np.full(len(truth), -1, dtype=int)
            for fold, (training, validation) in enumerate(folds_before):
                np.testing.assert_array_equal(experiment.folds[fold][0], training)
                np.testing.assert_array_equal(experiment.folds[fold][1], validation)
                model = clone(experiment._pipeline(candidate))
                X = experiment.X.iloc[experiment.development][columns]
                model.fit(X.iloc[training], truth[training])
                expected[validation] = model.predict_proba(X.iloc[validation])[:, 1]
                expected_predictions[validation] = model.predict(X.iloc[validation])
                self.assertAlmostEqual(scores.loc[candidate.name, f'fold_{fold+1}'],
                    accuracy_score(truth[validation], expected_predictions[validation]))
            np.testing.assert_allclose(saved.probability, expected)
            np.testing.assert_allclose(experiment.oof_probabilities[candidate.name], expected)
            np.testing.assert_array_equal(saved.predicted, expected_predictions)
            np.testing.assert_array_equal(experiment.oof_predictions[candidate.name], expected_predictions)
            for metric, expected_value in [
                ('oof_accuracy', accuracy_score(truth, expected_predictions)),
                ('roc_auc', roc_auc_score(truth, expected)),
                ('log_loss', log_loss(truth, expected)),
                ('brier_score', brier_score_loss(truth, expected)),
            ]:
                self.assertAlmostEqual(scores.loc[candidate.name, metric], expected_value)
        paired = experiment.paired_comparison('baseline', ['expanded']).iloc[0]
        base_correct = experiment.oof_predictions['baseline'] == truth
        new_correct = experiment.oof_predictions['expanded'] == truth
        self.assertEqual(paired.corrected_errors, int((~base_correct & new_correct).sum()))
        self.assertEqual(paired.introduced_errors, int((base_correct & ~new_correct).sum()))
        self.assertGreater(paired.corrected_errors, 0)
        self.assertEqual(paired.introduced_errors, 0)
        self.assertEqual(paired.folds_better, len(experiment.folds))
        self.assertAlmostEqual(paired.delta_pp, 100 * (scores.loc['expanded', 'cv_accuracy'] - scores.loc['baseline', 'cv_accuracy']))
        with self.assertRaisesRegex(ValueError, 'new, unique'):
            experiment.compare([baseline])
        self.assertFalse(experiment.manifest['holdout_evaluated'])

    def test_exact_probability_ties_preserve_estimator_decisions_everywhere(self):
        experiment = self.experiment()
        def tie_pipeline(candidate):
            return Pipeline([('scaler', StandardScaler()),
                             ('model', FixedTieClassifier(prediction=candidate.params['prediction']))])
        candidates = [Candidate('tie_zero', groups=('core',), params={'prediction': 0}),
                      Candidate('tie_one', groups=('core',), params={'prediction': 1})]
        with patch.object(experiment, '_pipeline', side_effect=tie_pipeline):
            experiment.compare(candidates)
        truth = experiment.y.iloc[experiment.development].to_numpy()
        scores = experiment.results.set_index('candidate')
        for name, prediction in [('tie_zero', 0), ('tie_one', 1)]:
            np.testing.assert_array_equal(experiment.oof_probabilities[name], np.full(len(truth), .5))
            np.testing.assert_array_equal(experiment.oof_predictions[name], np.full(len(truth), prediction))
            saved = pd.read_csv(experiment.output_path / f'oof_{name}.csv')
            np.testing.assert_array_equal(saved.predicted, np.full(len(truth), prediction))
            self.assertAlmostEqual(scores.loc[name, 'oof_accuracy'], float(np.mean(truth == prediction)))
            self.assertAlmostEqual(scores.loc[name, 'balanced_accuracy'], .5)
            for fold, (_, validation) in enumerate(experiment.folds, 1):
                self.assertAlmostEqual(scores.loc[name, f'fold_{fold}'],
                                       float(np.mean(truth[validation] == prediction)))
        paired = experiment.paired_comparison('tie_zero', ['tie_one']).iloc[0]
        self.assertEqual(paired.corrected_errors, int((truth == 1).sum()))
        self.assertEqual(paired.introduced_errors, int((truth == 0).sum()))
        self.assertGreater(paired.corrected_errors, 0)
        self.assertGreater(paired.introduced_errors, 0)
        self.assertAlmostEqual(paired.delta_pp,
            100 * (scores.loc['tie_one', 'cv_accuracy'] - scores.loc['tie_zero', 'cv_accuracy']))

    def test_soft_voting_averages_independent_member_subset_probabilities(self):
        experiment = self.experiment()
        members = [Candidate('signal_member', groups=('core',), params={'C': .5}),
                   Candidate('marker_member', groups=('extra',), params={'C': .5})]
        candidate = Candidate('ensemble', groups=('core', 'extra'), model='soft_voting',
                              params={'members': [asdict(member) for member in members]})
        train, valid = experiment.folds[0]
        X = experiment.X.iloc[experiment.development]
        y = experiment.y.iloc[experiment.development].to_numpy()
        expected_members = []
        for member in members:
            columns = experiment._columns(member)
            model = experiment._pipeline(member)
            model.fit(X.iloc[train][columns], y[train])
            expected_members.append(model.predict_proba(X.iloc[valid][columns]))
        model = experiment._pipeline(candidate)
        model.fit(X.iloc[train], y[train])
        np.testing.assert_allclose(model.predict_proba(X.iloc[valid]), np.mean(expected_members, axis=0))
        np.testing.assert_array_equal(model.predict(X.iloc[valid]), np.mean(expected_members, axis=0).argmax(axis=1))
        for fitted, member in zip(model.named_steps['model'].estimators_, members):
            self.assertEqual(fitted.named_steps['pipeline'].feature_names_in_.tolist(), experiment._columns(member))
        experiment.compare([candidate])
        frequency = pd.read_csv(experiment.output_path / 'selection_frequency_ensemble.csv')
        self.assertTrue(frequency.selected_in_folds.isna().all())
        np.testing.assert_allclose(experiment.oof_probabilities['ensemble'][valid],
                                   np.mean(expected_members, axis=0)[:, 1])
        invalid = Candidate('bad_ensemble', groups=('core',), model='soft_voting',
                            params={'members': [asdict(member) for member in members]})
        with self.assertRaisesRegex(ValueError, 'match the ensemble groups'):
            experiment._pipeline(invalid)

    def test_synthetic_holdout_opens_once_and_locks_comparisons(self):
        experiment = self.experiment()
        with self.assertRaises(RuntimeError):
            experiment.make_submission()
        experiment.compare([Candidate('chosen', groups=('core',)), Candidate('other', groups=('extra',))])
        result = experiment.evaluate_holdout('chosen')
        self.assertEqual(result['n_battles'], len(experiment._holdout))
        self.assertEqual(result, experiment.evaluate_holdout('chosen'))
        self.assertTrue(experiment.manifest['holdout_evaluated'])
        self.assertFalse(experiment.manifest['test_read'])
        saved = json.loads((experiment.output_path / 'experiment.json').read_text())
        self.assertTrue(saved['holdout_evaluated'])
        with self.assertRaisesRegex(RuntimeError, 'locked'):
            experiment.compare([Candidate('later', groups=('core',))])
        with self.assertRaisesRegex(RuntimeError, 'another candidate'):
            experiment.evaluate_holdout('other')
        self.assertTrue(all(path.name == 'train.jsonl' for path in self.read_paths))


if __name__ == '__main__':
    unittest.main()
