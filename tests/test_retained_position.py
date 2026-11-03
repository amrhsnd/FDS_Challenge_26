"""Validate retained-roster resources and observed effect exposure."""

from copy import deepcopy
from math import isfinite
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_challenge.features.retained_position import (
    NEW_FEATURE_COLUMNS,
    retained_position_features,
)


class RetainedPositionTests(unittest.TestCase):
    @staticmethod
    def state(name, hp, status="nostatus", effects=None):
        return {"name": name, "hp_pct": hp, "status": status,
                "effects": ["noeffect"] if effects is None else effects}

    @staticmethod
    def member(name, hp=100, spd=80, spa=100):
        return {"name": name, "base_hp": hp, "base_spd": spd, "base_spa": spa}

    def test_switching_healing_and_repeated_species_keep_latest_hp(self):
        battle = {
            "p1_team_details": [self.member("a", 100, 80, 120),
                                self.member("b", 80, 100, 60)],
            "p2_lead_details": {"name": "x"},
            "battle_timeline": [
                {"turn": 1, "p1_pokemon_state": self.state("a", 0.2),
                 "p2_pokemon_state": self.state("x", 0.8)},
                {"turn": 2, "p1_pokemon_state": self.state("b", 0.5),
                 "p2_pokemon_state": self.state("y", 0.2)},
                {"turn": 3, "p1_pokemon_state": self.state("a", 0.75),
                 "p2_pokemon_state": self.state("x", 0.4)},
                {"turn": 4, "p2_pokemon_state": self.state("y", 0.9)},
            ],
        }
        features = retained_position_features(battle)
        self.assertEqual(features["p2_healthy_living_members"], 5)
        self.assertEqual(features["p1_remaining_special_bulk"], 10000)
        self.assertEqual(features["p1_hp_weighted_base_special_attack"], 96)

    def test_faint_status_overrides_positive_hp_and_sleep_is_not_faint(self):
        battle = {
            "p1_team_details": [self.member("a"), self.member("b", 80, 100, 60)],
            "battle_timeline": [
                {"turn": 1, "p1_pokemon_state": self.state("a", 0.9, "fnt"),
                 "p2_pokemon_state": self.state("x", 0.9, "fnt")},
                {"turn": 2, "p1_pokemon_state": self.state("b", 0.5, "slp"),
                 "p2_pokemon_state": self.state("y", 0.9, "slp")},
            ],
        }
        features = retained_position_features(battle)
        self.assertEqual(features["p1_remaining_special_bulk"], 4000)
        self.assertEqual(features["p1_hp_weighted_base_special_attack"], 60)
        self.assertEqual(features["p2_healthy_living_members"], 4)

    def test_missing_status_and_exact_half_hp_exclude_healthy_members(self):
        battle = {
            "p2_lead_details": {"name": "lead"},
            "battle_timeline": [
                {"turn": 1, "p2_pokemon_state": {"name": "lead", "hp_pct": 0.9}},
                {"turn": 2, "p2_pokemon_state": self.state("half", 0.5)},
                {"turn": 3, "p2_pokemon_state": self.state("unknown", None)},
            ],
        }
        self.assertEqual(retained_position_features(battle)["p2_healthy_living_members"], 3)

    def test_unknown_hp_preserves_previous_known_hp(self):
        battle = {
            "p1_team_details": [self.member("a")],
            "p2_lead_details": {"name": "lead"},
            "battle_timeline": [
                {"turn": 1, "p1_pokemon_state": self.state("a", 0.25),
                 "p2_pokemon_state": self.state("lead", 0.4)},
                {"turn": 2, "p1_pokemon_state": self.state("a", None),
                 "p2_pokemon_state": self.state("lead", None)},
            ],
        }
        features = retained_position_features(battle)
        self.assertEqual(features["p1_remaining_special_bulk"], 2000)
        self.assertEqual(features["p2_healthy_living_members"], 5)

    def test_zero_weights_and_missing_special_attack(self):
        fainted = {"p1_team_details": [self.member("a")], "battle_timeline": [
            {"turn": 1, "p1_pokemon_state": self.state("a", 0)}]}
        self.assertEqual(retained_position_features(fainted)["p1_hp_weighted_base_special_attack"], 0)
        missing = self.member("unknown")
        del missing["base_spa"]
        battle = {"p1_team_details": [self.member("a", spa=80), missing]}
        self.assertEqual(retained_position_features(battle)["p1_hp_weighted_base_special_attack"], 80)
        self.assertEqual(retained_position_features({"p1_team_details": [missing]})[
            "p1_hp_weighted_base_special_attack"], 0)

    def test_effect_sentinel_and_missing_rows_remain_in_denominator(self):
        battle = {"battle_timeline": [
            {"turn": 1, "p1_pokemon_state": self.state("a", 1, effects=["leechseed"])},
            {"turn": 2, "p1_pokemon_state": self.state("a", 1, effects=["noeffect"])},
            {"turn": 3},
            {"turn": 4, "p1_pokemon_state": self.state("a", 1, effects=["leechseed", "confusion"])},
        ]}
        self.assertEqual(retained_position_features(battle)["p1_leechseed_rate"], 0.5)

    def test_observation_boundary_precedes_valid_turn_filter(self):
        baseline = [{"turn": 1}] * 30
        excluded = {"turn": 30, "p1_pokemon_state": self.state("a", 0, effects=["leechseed"])}
        battle = {"p1_team_details": [self.member("a")],
                  "battle_timeline": baseline + [excluded]}
        features = retained_position_features(battle)
        self.assertEqual(features["p1_remaining_special_bulk"], 8000)
        self.assertEqual(features["p1_leechseed_rate"], 0)
        invalid = [{"turn": t, "p1_pokemon_state": self.state("a", 0, effects=["leechseed"])}
                   for t in (0, 31, None, float("nan"), 2.5)]
        self.assertEqual(retained_position_features({"battle_timeline": invalid})["p1_leechseed_rate"], 0)
        included = {"battle_timeline": [{"turn": 30, "p1_pokemon_state":
                                         self.state("a", 1, effects=["leechseed"])}]}
        self.assertEqual(retained_position_features(included)["p1_leechseed_rate"], 1)

    def test_chronology_and_six_member_cap(self):
        timeline = [{"turn": 20, "p2_pokemon_state": self.state("x", 0.2)},
                    {"turn": 2, "p2_pokemon_state": self.state("x", 0.9)}]
        self.assertEqual(retained_position_features({"battle_timeline": timeline})[
            "p2_healthy_living_members"], 5)
        many = [{"turn": i + 1, "p2_pokemon_state": self.state(str(i), 1)} for i in range(8)]
        self.assertEqual(retained_position_features({"battle_timeline": many})[
            "p2_healthy_living_members"], 6)

    def test_fixed_finite_outputs_without_labels_ids_or_mutation(self):
        battle = {"p1_team_details": [self.member("a")],
                  "battle_timeline": [{"turn": 1, "p1_pokemon_state": self.state("a", 0.7)}]}
        untouched = deepcopy(battle)
        result = retained_position_features(battle)
        self.assertEqual(battle, untouched)
        self.assertEqual(tuple(result), NEW_FEATURE_COLUMNS)
        for value in result.values():
            self.assertIsInstance(value, float)
            self.assertTrue(isfinite(value))
        self.assertEqual(result, retained_position_features({**battle, "battle_id": "different", "player_won": 1}))
        malformed = {"p1_team_details": [self.member("a", float("inf"), None, float("nan"))],
                     "battle_timeline": [{"turn": 1, "p1_pokemon_state": self.state("a", float("nan"))}]}
        self.assertTrue(all(isfinite(value) for value in retained_position_features(malformed).values()))
        self.assertEqual(tuple(retained_position_features({})), NEW_FEATURE_COLUMNS)
