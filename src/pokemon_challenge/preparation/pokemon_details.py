"""Build a species lookup from training rosters and initial opponent leads.

No labels, test battles, or external data are used. Timeline states contribute
names for the coverage check only, never initial attributes.
"""
import argparse
from collections import Counter
import hashlib
import json
from numbers import Real
from pathlib import Path

from ..data import iter_battles
from ..paths import data_dir, project_root


STATS = ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe")
FILENAME = "pokemon_details.json"


def default_output_dir():
    return project_root() / "data" / "prepared" / "pokemon_details"


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every observed Pokemon needs a nonempty name")
    return value.strip().lower()


def extract_pokemon_details(battles):
    """Collect invariant species attributes and all observed level/item variants.

    Different types or base stats for the same species are an error. Levels and
    items are battle-specific, so they are retained as variants, not collapsed
    to an arbitrary value. A null item means it was not supplied in that record.
    """
    pokemon, variants = {}, {}
    timeline_names = set()
    battle_count = initial_count = 0
    for battle in battles:
        battle_count += 1
        records = [("p1_team_details", p) for p in battle.get("p1_team_details", [])]
        if battle.get("p2_lead_details") is not None:
            records.append(("p2_lead_details", battle["p2_lead_details"]))
        for source, record in records:
            name = _name(record.get("name"))
            raw_types = record.get("types")
            if not isinstance(raw_types, list) or not raw_types:
                raise ValueError(f"Missing types for {name}")
            types = sorted({_name(t) for t in raw_types} - {"notype"})
            if not 1 <= len(types) <= 2:
                raise ValueError(f"Expected one or two real types for {name}")
            intrinsic = {"name": name, "types": types}
            for stat in STATS:
                value = record.get(stat)
                if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < float("inf"):
                    raise ValueError(f"Invalid or missing {stat} for {name}")
                intrinsic[stat] = value
            level = record.get("level")
            if isinstance(level, bool) or not isinstance(level, Real) or not 0 < level <= 100:
                raise ValueError(f"Invalid or missing level for {name}")
            item = record.get("item")
            if item is not None and not isinstance(item, str):
                raise ValueError(f"Invalid item for {name}")
            if name not in pokemon:
                pokemon[name] = intrinsic
                variants[name] = Counter()
            elif pokemon[name] != intrinsic:
                different = [key for key in intrinsic if pokemon[name][key] != intrinsic[key]]
                raise ValueError(f"Conflicting initial details for {name}: {different}; "
                                 f"battle {battle.get('battle_id')}, source {source}")
            variants[name][(level, item, source)] += 1
            initial_count += 1
        for turn in battle.get("battle_timeline", []):
            for side in ("p1", "p2"):
                state = turn.get(f"{side}_pokemon_state") or {}
                if state.get("name"):
                    timeline_names.add(_name(state["name"]))
    if not pokemon:
        raise ValueError("Training data contains no initial Pokemon details")
    for name, detail in pokemon.items():
        counts = variants[name]
        detail["observed_levels"] = sorted({level for level, _, _ in counts})
        detail["observed_items"] = sorted({item for _, item, _ in counts if item is not None})
        detail["initial_observations"] = sum(counts.values())
        detail["initial_variants"] = [
            {"level": level, "item": item, "source": source, "observations": count}
            for (level, item, source), count in sorted(
                counts.items(), key=lambda pair: (pair[0][0], pair[0][1] or "", pair[0][2]))
        ]
    return {
        "schema_version": 1,
        "pokemon": dict(sorted(pokemon.items())),
        "coverage": {
            "training_battles": battle_count,
            "initial_observations": initial_count,
            "species_with_initial_details": len(pokemon),
            "species_seen_anywhere": len(set(pokemon) | timeline_names),
            "timeline_species_without_initial_details": sorted(timeline_names - set(pokemon)),
        },
    }


def prepare_pokemon_details(train_path=None, output_dir=None):
    """Prepare and save the lookup; return its path. Reruns replace this artifact."""
    train_path = Path(train_path) if train_path is not None else data_dir() / "train.jsonl"
    result = extract_pokemon_details(iter_battles(train_path))
    digest = hashlib.sha256()
    with train_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result["source"] = {"filename": train_path.name, "sha256": digest.hexdigest(),
                        "initial_fields": ["p1_team_details", "p2_lead_details"]}
    destination = Path(output_dir) if output_dir is not None else default_output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / FILENAME
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_pokemon_details(path=None):
    """Load the prepared dictionary keyed by species name without reading JSONL.

    Unknown species are absent; use lookup.get(name), never a fabricated type.
    """
    path = Path(path) if path is not None else default_output_dir() / FILENAME
    if not path.is_file():
        raise FileNotFoundError("Prepare the lookup first: python -m "
                                "pokemon_challenge.preparation.pokemon_details")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or not isinstance(result.get("pokemon"), dict):
        raise ValueError(f"Unsupported Pokemon details file: {path}")
    return result["pokemon"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="Training JSONL; defaults to shared data_dir()/train.jsonl")
    parser.add_argument("--output-dir", type=Path, help="Destination folder for the prepared lookup")
    args = parser.parse_args()
    path = prepare_pokemon_details(args.train, args.output_dir)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"Saved {len(result['pokemon'])} Pokemon to {path}")
    missing = result["coverage"]["timeline_species_without_initial_details"]
    print(f"Timeline species without initial details ({len(missing)}): {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
