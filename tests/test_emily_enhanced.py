"""Regression checks for the new Emily-derived feature transformer."""
import copy
from pathlib import Path
import sys
import unittest
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'notebooks/_amir_support'))
from emily_enhanced import EmilyFeatures, clean_observation
from pokemon_challenge.data import iter_battles


class EnhancedFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chart = ROOT/'data/raw/type_chart_gen2.csv'
        cls.train = ROOT/'data/raw/train.jsonl'
        if not cls.chart.exists() or not cls.train.exists():
            raise unittest.SkipTest('Local competition inputs required.')
        cls.battle = next(iter_battles(cls.train))

    def test_forbidden_fields_do_not_change_features(self):
        original = pd.DataFrame({'battle': [self.battle]})
        model = EmilyFeatures(str(self.chart)).fit(original)
        modified = copy.deepcopy(self.battle)
        modified['battle_id'] = -1
        modified['player_won'] = 1 - modified['player_won']
        modified['battle_timeline'].append({'turn': 31, 'secret': 'outcome'})
        pd.testing.assert_frame_equal(model.transform(original),
            model.transform(pd.DataFrame({'battle': [modified]})))

    def test_validation_metadata_does_not_enter_fitted_lookup(self):
        model = EmilyFeatures(str(self.chart)).fit(pd.DataFrame({'battle': [self.battle]}))
        before = copy.deepcopy(model.dex_)
        validation = copy.deepcopy(self.battle)
        validation['p1_team_details'][0]['name'] = 'validation-only-species'
        model.transform(pd.DataFrame({'battle': [validation]}))
        self.assertEqual(before, model.dex_)
        self.assertNotIn('validation-only-species', model.dex_)

    def test_faint_hp_is_zero_and_missing_turn_fails(self):
        battle = copy.deepcopy(self.battle)
        state = battle['battle_timeline'][0]['p1_pokemon_state']
        state.update(status='fnt', hp_pct=.9)
        self.assertEqual(clean_observation(battle)['battle_timeline'][0]['p1_pokemon_state']['hp_pct'], 0.)
        self.assertEqual(state['hp_pct'], .9)  # Input remains immutable.
        battle['battle_timeline'].pop()
        with self.assertRaisesRegex(ValueError, 'exactly turns'):
            clean_observation(battle)


if __name__ == '__main__':
    unittest.main()
