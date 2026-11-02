"""Train-only preparation, variant preservation, conflicts, and saved lookup."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_challenge.preparation.pokemon_details import (
    extract_pokemon_details, load_pokemon_details, prepare_pokemon_details,
)


def fixture():
    pokemon = {"name": "example", "types": ["water", "notype"], "level": 100,
               "item": "leftovers", **{f"base_{s}": 50 for s in ("hp", "atk", "def", "spa", "spd", "spe")}}
    lead = {k: v for k, v in pokemon.items() if k != "item"}
    return {"battle_id": 1, "player_won": 1, "p1_team_details": [pokemon],
            "p2_lead_details": lead,
            "battle_timeline": [{"p2_pokemon_state": {"name": "unknown"}}]}


class PokemonDetailsTests(unittest.TestCase):
    def test_variants_coverage_and_no_labels(self):
        first = fixture()
        second = copy.deepcopy(first)
        second["p1_team_details"][0].update(level=50, item="berry")
        second["p2_lead_details"]["types"] = ["notype", "water"]
        result = extract_pokemon_details([first, second])
        detail = result["pokemon"]["example"]
        self.assertEqual(detail["types"], ["water"])
        self.assertEqual(detail["observed_levels"], [50, 100])
        self.assertEqual(detail["observed_items"], ["berry", "leftovers"])
        self.assertEqual(detail["initial_observations"], 4)
        self.assertEqual(sum(v["observations"] for v in detail["initial_variants"]), 4)
        self.assertTrue(any(v["item"] is None for v in detail["initial_variants"]))
        self.assertEqual(result["coverage"]["timeline_species_without_initial_details"], ["unknown"])
        self.assertNotIn("player_won", json.dumps(result))
        first["player_won"] = 0
        self.assertEqual(extract_pokemon_details([first, second]), result)

    def test_conflicting_species_attributes_rejected(self):
        for field, value in [("base_hp", 60), ("types", ["fire"])]:
            with self.subTest(field=field):
                battle = fixture()
                battle["p2_lead_details"][field] = value
                with self.assertRaisesRegex(ValueError, "Conflicting initial details"):
                    extract_pokemon_details([battle])

    def test_invalid_and_empty_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "no initial"):
            extract_pokemon_details([])
        for field in ("types", "base_hp", "level"):
            with self.subTest(field=field):
                battle = fixture()
                del battle["p1_team_details"][0][field]
                with self.assertRaises(ValueError):
                    extract_pokemon_details([battle])

    def test_save_load_and_deterministic_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            train.write_text(json.dumps(fixture()) + "\n", encoding="utf-8")
            output = root / "prepared"
            path = prepare_pokemon_details(train, output)
            first = path.read_bytes()
            payload = json.loads(first)
            self.assertEqual(payload["source"]["sha256"], hashlib.sha256(train.read_bytes()).hexdigest())
            self.assertEqual(load_pokemon_details(path)["example"]["base_hp"], 50)
            prepare_pokemon_details(train, output)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(list(root.glob("test*")), [])
            with self.assertRaises(FileNotFoundError):
                load_pokemon_details(root / "missing.json")
