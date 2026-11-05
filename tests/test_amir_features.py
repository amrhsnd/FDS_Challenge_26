"""Check the numeric battle-feature registry and observation boundary."""
from copy import deepcopy
import math
from numbers import Real
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'notebooks/_amir_support')]
from features import build_feature_functions
from study import observed_battle
from team_features import _schema_fixture


class FeatureRegistryTests(unittest.TestCase):
    def test_registry_values_ignore_targets_ids_and_future_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            chart = Path(directory) / 'chart.csv'
            chart.write_text('attacking_type,defending_type,multiplier\nwater,fire,2\n')
            groups = build_feature_functions(chart)
        battle = _schema_fixture()
        names = ['tyranitar', 'snorlax', 'starmie', 'skarmory', 'miltank', 'zapdos']
        for pokemon, name in zip(battle['p1_team_details'], names):
            pokemon['name'] = name
        battle['p2_lead_details']['name'] = 'forretress'
        for turn in battle['battle_timeline']:
            turn['p1_pokemon_state']['name'] = names[0]
            turn['p2_pokemon_state']['name'] = 'forretress'
        original = deepcopy(battle)
        changed = deepcopy(battle)
        changed.update(player_won=1, battle_id='different', final_winner='p1')
        future = deepcopy(changed['battle_timeline'][-1])
        future['turn'] = 31
        future['p1_pokemon_state'].update(hp_pct=0, status='fnt')
        changed['battle_timeline'].append(future)
        changed['battle_timeline'].reverse()

        def numeric_row(value):
            observed = observed_battle(value)
            return {f'{group}__{key}': number
                    for group, function in groups.items()
                    for key, number in function(observed).items()}

        expected, actual = numeric_row(battle), numeric_row(changed)
        self.assertEqual(len(groups), 20)
        self.assertEqual(len(expected), 757)
        self.assertEqual(expected.keys(), actual.keys())
        for key, value in expected.items():
            self.assertIsInstance(value, Real, key)
            self.assertFalse(math.isinf(value), key)
            if math.isnan(value):
                self.assertTrue(math.isnan(actual[key]), key)
            else:
                self.assertEqual(value, actual[key], key)
        self.assertEqual(battle, original)


if __name__ == '__main__':
    unittest.main()
