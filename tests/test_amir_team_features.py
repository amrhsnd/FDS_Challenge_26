"""Integration checks for observation-only team-feature adapters."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/amirhosein"))
from team_features import FEATURE_DESCRIPTIONS, _schema_fixture, make_team_feature_groups, sanitize_battle


class TeamFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.chart = Path(cls.directory.name) / "chart.csv"
        cls.chart.write_text("attacking_type,defending_type,multiplier\nwater,fire,2\nfire,water,0.5\n")
        cls.groups = make_team_feature_groups(cls.chart)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def assert_rows_equal(self, a, b):
        self.assertEqual(tuple(a), tuple(b))
        np.testing.assert_allclose(list(a.values()), list(b.values()), equal_nan=True)

    def test_synthetic_schema_counts_and_numeric_contract(self):
        self.assertEqual(set(self.groups), set(FEATURE_DESCRIPTIONS))
        self.assertEqual(sum(len(group.columns) for name, group in self.groups.items()
                             if name.startswith("aimilia_")), 274)
        for name, group in self.groups.items():
            row = group(_schema_fixture())
            self.assertEqual(tuple(row), group.columns)
            self.assertNotIn("battle_id", row)
            self.assertNotIn("player_won", row)
            self.assertFalse(any(math.isinf(value) for value in row.values()))

    def test_labels_ids_future_and_turn_order_do_not_change_features(self):
        baseline = _schema_fixture()
        changed = deepcopy(baseline)
        changed.update(battle_id="irrelevant", player_won=1, winner="p2")
        changed["battle_timeline"].reverse()
        future = deepcopy(changed["battle_timeline"][0])
        future["turn"] = 31
        future["p1_pokemon_state"].update(name="future", hp_pct=0, status="fnt")
        changed["battle_timeline"].append(future)
        for group in self.groups.values():
            self.assert_rows_equal(group(baseline), group(changed))

    def test_empty_and_partial_timelines_have_fixed_missing_schemas(self):
        for group in self.groups.values():
            row = group({})
            self.assertEqual(tuple(row), group.columns)
            self.assertTrue(all(math.isnan(value) for value in row.values()))
        battle = _schema_fixture()
        battle["battle_timeline"] = [turn for turn in battle["battle_timeline"] if turn["turn"] != 20]
        row = self.groups["aimilia_roster"](battle)
        self.assertTrue(all(math.isnan(value) for value in row.values()))
        self.assertTrue(any(math.isfinite(value) for value in self.groups["aimilia_flow"](battle).values()))

    def test_faint_override_is_copied_and_no_unknown_species_are_imputed(self):
        battle = _schema_fixture()
        original = deepcopy(battle)
        battle["battle_timeline"][-1]["p1_pokemon_state"].update(hp_pct=.8, status="fnt")
        self.assertEqual(self.groups["aimilia_roster"](battle)["p1_alive_t30"], 5)
        self.assertEqual(battle["battle_timeline"][-1]["p1_pokemon_state"]["hp_pct"], .8)
        self.assertEqual(original["battle_timeline"][-1]["p1_pokemon_state"]["hp_pct"], 1.)
        for turn in battle["battle_timeline"]:
            turn["p2_pokemon_state"]["name"] = "unknown_enemy"
        self.assertTrue(math.isnan(self.groups["aimilia_speed"](battle)["p2_active_spe_mean"]))

    def test_invalid_observation_is_not_silently_swallowed(self):
        battle = _schema_fixture()
        battle["battle_timeline"][0]["p1_pokemon_state"]["hp_pct"] = float("inf")
        with self.assertRaises(ValueError):
            self.groups["aimilia_flow"](battle)

    def test_real_record_horizon_target_and_schema(self):
        candidates = [ROOT / "data/raw/train.jsonl", ROOT / "data/train.jsonl",
                      Path("/Users/amrhsnd/Desktop/FDS/2026/Challange/raw data/train.jsonl")]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            self.skipTest("Official local training input not present")
        with path.open() as stream:
            battle = json.loads(next(stream))
        changed = deepcopy(battle)
        changed["battle_id"] = "changed"
        changed["player_won"] = not bool(battle.get("player_won"))
        future = deepcopy(changed["battle_timeline"][-1])
        future["turn"] = 999
        changed["battle_timeline"].append(future)
        for group in self.groups.values():
            self.assert_rows_equal(group(battle), group(changed))


if __name__ == "__main__":
    unittest.main()
