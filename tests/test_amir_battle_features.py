"""Synthetic checks for observation boundaries and resource semantics."""

from copy import deepcopy
from math import isfinite, isnan
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/_amir_support"))
from battle_features import (FEATURE_GROUPS, STATS, active_position_features,
                            observed_turns, recent_flow_features,
                            roster_shape_features, status_resource_features)


def member(name, value=100):
    return {"name": name, "level": 100, "item": "leftovers",
            **{f"base_{stat}": value for stat in STATS}}


def state(name, hp=1.0, status="nostatus", **stages):
    return {"name": name, "hp_pct": hp, "status": status,
            "boosts": {stat: stages.get(stat, 0) for stat in ("atk", "def", "spa", "spd", "spe")}}


def fixture():
    return {"p1_team_details": [member("own0"), member("own1", 50)],
            "p2_lead_details": member("lead"),
            "battle_timeline": [{"turn": turn,
                                 "p1_pokemon_state": state("own0"),
                                 "p2_pokemon_state": state("lead")}
                                for turn in range(1, 31)]}


class AmirBattleFeatureTests(unittest.TestCase):
    def assert_same_features(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if isnan(expected[key]):
                self.assertTrue(isnan(actual[key]), key)
            else:
                self.assertEqual(actual[key], expected[key], key)

    def test_horizon_target_id_and_order_independence(self):
        original = fixture()
        changed = deepcopy(original)
        changed.update(player_won=1, battle_id="ignored")
        changed["battle_timeline"].reverse()
        changed["battle_timeline"].insert(0, {"turn": 31, "p1_pokemon_state": state("own0", 0, "fnt")})
        for function in FEATURE_GROUPS.values():
            self.assert_same_features(function(changed), function(original))

    def test_fixed_schema_missing_fields_and_no_mutation(self):
        original = fixture()
        snapshot = deepcopy(original)
        for function in FEATURE_GROUPS.values():
            full, empty = function(original), function({})
            self.assertEqual(set(full), set(empty))
            self.assertTrue(all(isinstance(v, (int, float)) for v in empty.values()))
            self.assertTrue(all(isfinite(v) or isnan(v) for v in empty.values()))
        self.assertEqual(original, snapshot)

    def test_turn_validation_and_duplicate_last_wins(self):
        battle = {"battle_timeline": [{"turn": 1, "tag": "first"},
                                      {"turn": 1, "tag": "last"},
                                      {"turn": 30}, {"turn": 0},
                                      {"turn": 31}, {"turn": 1.5},
                                      {"turn": True}, {"turn": float("inf")}, {}]}
        turns = observed_turns(battle)
        self.assertEqual([number for number, _ in turns], [1, 30])
        self.assertEqual(turns[0][1]["tag"], "last")

    def test_roster_spread_uses_real_supplied_stats(self):
        result = roster_shape_features(fixture())
        self.assertEqual(result["p1_base_spa_std"], 25)
        self.assertEqual(result["p1_base_spd_min"], 50)
        self.assertEqual(result["p1_base_spe_max"], 100)
        self.assertEqual(result["p1_mean_minus_p2_lead_spa"], -25)
        self.assertEqual(result["p1_leftovers_fraction_known_items"], 1)

    def test_sleep_is_living_and_faint_overrides_positive_hp(self):
        battle = fixture()
        battle["battle_timeline"][-2]["p1_pokemon_state"] = state("own0", 0.5, "slp")
        battle["battle_timeline"][-1]["p1_pokemon_state"] = state("own1", 0.9, "fnt")
        result = status_resource_features(battle)
        self.assertEqual(result["p1_retained_slp_count"], 1)
        self.assertEqual(result["p1_retained_incapacitated_hp"], 0.5)
        self.assertEqual(result["p1_retained_healthy_hp"], 0)

    def test_unknown_opponent_has_no_invented_hp_history_or_stats(self):
        battle = fixture()
        battle["battle_timeline"][-1]["p2_pokemon_state"] = state("hidden", 0.2)
        flow = recent_flow_features(battle)
        self.assertEqual(flow["p2_t21_30_hp_loss"], 0)
        active = active_position_features(battle)
        self.assertEqual(active["p2_active_stats_supplied"], 0)
        self.assertTrue(isnan(active["p2_active_speed_proxy"]))
        status = status_resource_features(battle)
        self.assertEqual(status["p2_unrevealed_slots"], 4)
        self.assertAlmostEqual(status["p2_retained_healthy_hp"], 1.2)

    def test_window_hp_changes_track_member_not_active_slot(self):
        battle = fixture()
        battle["battle_timeline"][9]["p1_pokemon_state"] = state("own0", 0.5)
        for turn in battle["battle_timeline"][10:20]:
            turn["p1_pokemon_state"] = state("own1")
        for turn in battle["battle_timeline"][20:]:
            turn["p1_pokemon_state"] = state("own0", 0.25)
        flow = recent_flow_features(battle)
        self.assertEqual(flow["p1_t1_10_hp_loss"], 0.5)
        self.assertEqual(flow["p1_t11_20_hp_loss"], 0)
        self.assertEqual(flow["p1_t11_20_hp_gain"], 0)
        self.assertEqual(flow["p1_t21_30_hp_loss"], 0.25)
        self.assertEqual(flow["p1_t11_20_switches"], 1)
        self.assertEqual(flow["p1_t21_30_switches"], 1)

    def test_repeated_faints_only_count_once_across_windows(self):
        battle = fixture()
        for turn in battle["battle_timeline"][8:]:
            turn["p2_pokemon_state"] = state("lead", 0.9, "fnt")
        result = recent_flow_features(battle)
        self.assertEqual(result["p2_t1_10_new_faints"], 1)
        self.assertEqual(result["p2_t11_20_new_faints"], 0)
        self.assertEqual(result["p2_t21_30_new_faints"], 0)

    def test_stage_hp_and_gen2_status_proxy_interactions(self):
        battle = fixture()
        battle["battle_timeline"][-1]["p1_pokemon_state"] = state("own0", 0.5, "par", atk=2, spe=2)
        result = active_position_features(battle)
        self.assertEqual(result["p1_active_atk_hp_stage"], 1)
        self.assertEqual(result["p1_active_speed_proxy"], 50)
        self.assertEqual(result["p1_active_hp_weighted_physical_attack_proxy"], 100)
        battle["battle_timeline"][-1]["p1_pokemon_state"] = state("own0", 0.5, "brn", atk=2)
        self.assertEqual(active_position_features(battle)["p1_active_hp_weighted_physical_attack_proxy"], 50)

    def test_missing_hp_status_and_boosts_stay_unknown(self):
        battle = fixture()
        battle["battle_timeline"][-1]["p1_pokemon_state"] = {"name": "own0"}
        result = active_position_features(battle)
        self.assertTrue(isnan(result["p1_active_hp"]))
        self.assertTrue(isnan(result["p1_active_atk_stage"]))
        status = status_resource_features(battle)
        self.assertEqual(status["p1_retained_unknown_status_count"], 1)
        self.assertEqual(status["p1_retained_healthy_hp"], 1)


if __name__ == "__main__":
    unittest.main()
