"""Check semantic names, physical units and deterministic matrix renaming."""
from pathlib import Path
import re
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks/_amir_support'))
from feature_catalog import (FEATURE_NAMES, FEATURE_DOCUMENTATION,
                             feature_dictionary, rename_frame)


class FeatureCatalogTests(unittest.TestCase):
    def test_names_are_unique_semantic_identifiers_and_have_definitions(self):
        names = list(FEATURE_NAMES.values())
        self.assertEqual(len(names),757)
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(set(names),set(FEATURE_DOCUMENTATION))
        for name in names:
            self.assertRegex(name,r'^[a-z][a-z0-9_]*$')
            self.assertFalse(any(word in name for word in ('aimilia','leonard','selected','expanded_core')))
            definition = FEATURE_DOCUMENTATION[name]
            self.assertGreater(len(definition['description']),30)
            self.assertNotIn(definition['units'],('','numeric proxy','indicator or count'))
            self.assertTrue(definition['category'])

    def test_base_and_level_scaled_stats_have_different_precise_names_and_units(self):
        base = FEATURE_NAMES['selected__p1_mean_atk']
        scaled = FEATURE_NAMES['aimilia_static__p1_atk_mean']
        self.assertEqual(base,'roster_p1_supplied_base_attack_mean')
        self.assertIn('level_scaled',scaled)
        self.assertNotEqual(base,scaled)
        self.assertEqual(FEATURE_DOCUMENTATION[base]['units'],'base stat')
        self.assertEqual(FEATURE_DOCUMENTATION[scaled]['units'],'level-scaled base stat')

    def test_reverse_signed_status_gaps_are_named_in_the_correct_direction(self):
        for key in ('aimilia_hazards__diff_doomed','aimilia_hazards__diff_tox_rate',
                    'aimilia_moves__diff_slp_dead_rate'):
            self.assertIn('p2_minus_p1',FEATURE_NAMES[key])
        self.assertIn('p1_minus_p2',FEATURE_NAMES['aimilia_roster__alive_diff_t30'])

    def test_hp_percent_bars_and_fraction_units_are_not_conflated(self):
        def unit(key):return FEATURE_DOCUMENTATION[FEATURE_NAMES[key]]['units']
        self.assertEqual(unit('selected__p1_team_hp_pct_at_turn30'),'percent of six full HP bars')
        self.assertEqual(unit('aimilia_roster__p1_teamhp_t30'),'HP bars')
        self.assertEqual(unit('expanded_core__p1_max_observed_move_damage'),'HP fraction')
        self.assertEqual(unit('leonard_battle_state__p1_observed_hp_sq'),'squared HP fraction')
        self.assertEqual(unit('aimilia_static__p1_hp_mean'),'level-scaled base stat')

    def test_renaming_preserves_values_order_and_input_frame(self):
        keys = ['aimilia_static__p1_hp_mean','selected__p1_mean_atk']
        original = pd.DataFrame([[80.,90.],[70.,100.]],columns=keys,index=[8,3])
        renamed = rename_frame(original)
        self.assertEqual(list(renamed),[FEATURE_NAMES[key] for key in keys])
        pd.testing.assert_frame_equal(renamed.set_axis(keys,axis=1),original)
        self.assertEqual(list(original),keys)
        renamed.iloc[0,0]=1
        self.assertEqual(original.iloc[0,0],80.)
        with self.assertRaises(ValueError):
            rename_frame(pd.DataFrame({'not_a_feature':[1]}))

    def test_dictionary_has_requested_order_without_extraction_keys(self):
        names=[FEATURE_NAMES['aimilia_hazards__p1_spikes_turn'],
               FEATURE_NAMES['aimilia_speed__p1_speed_adv_rate']]
        table=feature_dictionary(names)
        self.assertEqual(table.feature.tolist(),names)
        self.assertEqual(list(table),['feature','category','description','units'])
        self.assertIn('31',table.iloc[0].description)
        with self.assertRaises(ValueError):feature_dictionary(['unknown'])


if __name__=='__main__':unittest.main()
