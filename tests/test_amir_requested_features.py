"""Check state tracking and observed HP changes in battle summaries."""
from pathlib import Path
import unittest
import sys



class RequestedFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(root / 'src'), str(root / 'notebooks/_amir_support')]
        import features
        cls.namespace = vars(features)

    @staticmethod
    def state(name, hp, status='nostatus', boosts=None):
        return {'name': name, 'hp_pct': hp, 'status': status, 'boosts': boosts or {}}

    def test_both_sides_switches_healing_final_boosts_and_roster_hp(self):
        state = self.state
        move = lambda category: {'category': category}
        def turn(number, p1, p2, m1=None, m2=None):
            return dict(turn=number, p1_pokemon_state=p1, p2_pokemon_state=p2,
                        p1_move_details=m1, p2_move_details=m2)
        battle = {
            'p1_team_details': [{'name': name} for name in ['A', 'D', 'E', 'F', 'G', 'H']],
            'p2_lead_details': {'name': 'B'},
            'battle_timeline': [
                turn(1, state('A', 1), state('B', 1), move('physical'), move('special')),
                turn(2, state('A', .8), state('B', .6), move('special'), move('physical')),
                # B -> C cannot be treated as damage to the same target.
                turn(3, state('A', .8), state('C', .2), move('status')),
                # A heals while P2 has a recorded move: no net damage observed.
                turn(4, state('A', 1), state('C', .2), None, move('special')),
                turn(30, state('D', .5, boosts={'atk': 2, 'def': -1, 'spa': 3, 'spd': -2}),
                     state('C', 0, 'fnt', {'atk': -2, 'spe': 1})),
                turn(31, state('D', 0, 'fnt'), state('C', 0, 'fnt'), move('special')),
            ],
        }
        f = self.namespace['extract_requested_features'](battle)
        self.assertAlmostEqual(f['p1_max_observed_move_damage'], .4)
        self.assertAlmostEqual(f['p2_max_observed_move_damage'], .2)
        self.assertEqual(f['p1_no_damage_moves'], 0)
        self.assertEqual(f['p2_no_damage_moves'], 1)
        self.assertEqual(f['p1_positive_boosts_at_turn30'], 5)
        self.assertEqual(f['p1_negative_boosts_at_turn30'], -3)
        self.assertEqual(f['p1_boost_sum_at_turn30'], 2)
        self.assertEqual(f['p2_positive_boosts_at_turn30'], 1)
        self.assertEqual(f['p2_negative_boosts_at_turn30'], -2)
        self.assertEqual(f['p2_boost_sum_at_turn30'], -1)
        self.assertEqual(f['p1_special_moves'], 1)
        self.assertEqual(f['p2_special_moves'], 2)
        self.assertEqual(f['p1_fainting_turns'], 0)
        self.assertEqual(f['p2_fainting_turns'], 1)
        self.assertAlmostEqual(f['p1_team_hp_pct_at_turn30'], 100 * 5.5 / 6)
        self.assertAlmostEqual(f['p2_team_hp_pct_at_turn30'], 100 * 4.6 / 6)

    def test_repeated_faint_states_count_once_and_null_moves_are_excluded(self):
        battle = {'battle_timeline': [
            {'turn': i, 'p1_pokemon_state': self.state('A', 0, 'fnt'),
             'p2_pokemon_state': self.state('B', 1),
             'p1_move_details': None, 'p2_move_details': None}
            for i in [1, 2, 30]
        ]}
        f = self.namespace['extract_requested_features'](battle)
        self.assertEqual(f['p1_fainting_turns'], 1)
        self.assertEqual(f['p2_fainting_turns'], 0)
        self.assertEqual(f['p1_no_damage_moves'], 0)
        self.assertEqual(f['p2_no_damage_moves'], 0)
        self.assertEqual(f['p1_team_hp_pct_at_turn30'], 100 * 5 / 6)

    def test_empty_observations_have_no_inferred_moves_or_final_boosts(self):
        f = self.namespace['extract_requested_features']({})
        self.assertEqual(len(f), 16)
        for side in ['p1', 'p2']:
            self.assertEqual(f[f'{side}_team_hp_pct_at_turn30'], 100)
            self.assertEqual(f[f'{side}_boost_sum_at_turn30'], 0)
            self.assertEqual(f[f'{side}_no_damage_moves'], 0)

    def surplus(self, timeline):
        return self.namespace['extract_damage_surplus_features'](
            {'battle_timeline': timeline})

    def test_surplus_assigns_each_turn_to_the_leading_side(self):
        result = self.surplus([
            {'turn': 1, 'p1_pokemon_state': self.state('A', 1),
             'p2_pokemon_state': self.state('B', .5)},
            {'turn': 2, 'p1_pokemon_state': self.state('A', .2),
             'p2_pokemon_state': self.state('B', .5)},
        ])
        self.assertAlmostEqual(result['p1_damage_surplus'], .5 / 30)
        self.assertAlmostEqual(result['p2_damage_surplus'], .3 * 29 / 30)

    def test_surplus_subtracts_opponent_healing_from_damage_dealt(self):
        result = self.surplus([
            {'turn': 1, 'p2_pokemon_state': self.state('B', .4)},
            {'turn': 2, 'p2_pokemon_state': self.state('B', .8)},
            {'turn': 3, 'p1_pokemon_state': self.state('A', .5)},
        ])
        self.assertAlmostEqual(result['p1_damage_surplus'], (.6 + .2) / 30)
        self.assertAlmostEqual(result['p2_damage_surplus'], .3 * 28 / 30)

    def test_surplus_weights_early_and_late_one_turn_advantage_equally(self):
        def pulse(turn):
            return self.surplus([
                {'turn': turn, 'p2_pokemon_state': self.state('B', .5)},
                {'turn': turn + 1, 'p2_pokemon_state': self.state('B', 1)},
            ])['p1_damage_surplus']
        self.assertAlmostEqual(pulse(1), .5 / 30)
        self.assertAlmostEqual(pulse(29), pulse(1))

    def test_surplus_tracks_switches_by_name_and_is_symmetric(self):
        timeline = [
            {'turn': 1, 'p2_pokemon_state': self.state('B', .5)},
            {'turn': 2, 'p2_pokemon_state': self.state('C', 1)},
            {'turn': 3, 'p2_pokemon_state': self.state('B', .7)},
        ]
        result = self.surplus(timeline)
        self.assertAlmostEqual(result['p1_damage_surplus'], (.5 * 2 + .3 * 28) / 30)
        self.assertEqual(result['p2_damage_surplus'], 0)
        swapped = self.surplus([
            {'turn': row['turn'], 'p1_pokemon_state': row['p2_pokemon_state']}
            for row in timeline])
        self.assertAlmostEqual(swapped['p2_damage_surplus'], result['p1_damage_surplus'])
        self.assertEqual(swapped['p1_damage_surplus'], 0)

    def test_surplus_empty_tied_and_after_turn_30(self):
        for timeline in ([], [
            {'turn': 1, 'p1_pokemon_state': self.state('A', .5),
             'p2_pokemon_state': self.state('B', .5)},
            {'turn': 31, 'p2_pokemon_state': self.state('B', 0)},
        ]):
            self.assertEqual(self.surplus(timeline),
                             {'p1_damage_surplus': 0., 'p2_damage_surplus': 0.})
        features = self.namespace['extract_selected_features']({})
        self.assertEqual(len(features), 32)
        for removed in ('p1_fainting_turns', 'p1_positive_boosts_at_turn30',
                        'p2_positive_boosts_at_turn30'):
            self.assertNotIn(removed, features)
        self.assertNotIn('p2_lead_spe', features)
        self.assertIn('p2_mean_spe', features)
        self.assertEqual(features['first_ko_advantage'], 0.0)
        for name in self.namespace['NEW_FEATURE_COLUMNS']:
            self.assertIn(name, features)
        self.assertIn('p1_damage_surplus', features)
        self.assertIn('p2_damage_surplus', features)


if __name__ == '__main__':
    unittest.main()
