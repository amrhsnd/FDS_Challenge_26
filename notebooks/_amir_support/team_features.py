"""Observation-only adapters for Aimilia's and Leonard's feature providers.

Sources are the sibling notebooks/aimilia/features_*.py modules and Leonard's
ablation_extensions.py/progressive_features.py, synced from GitHub. Providers
remain in their original files; this adapter supplies a consistent horizon,
numeric schemas, and no-label/no-ID interface for Amir's controlled comparison.

No training-wide species lookup is fitted. Matchup, speed and expected-damage
providers receive dex={} and resolve ONLY metadata supplied in the same battle.
An unknown revealed opponent therefore produces missing estimates rather than
an invented species attribute. Missing estimates are imputed inside model folds.

These features are proxies: HP and status on the bench can be stale; unseen
members are assumed healthy in roster providers; switch counts include forced
replacements. Spikes flags mean observed use, not verified current hazards.
Toxic/seeded 'doomed' counts do not predict certain deaths. Bench stat boosts and
volatile effects can reset on switching. Speed/damage estimates use simplified
base-stat mechanics, not a battle simulator or exact move attribution.
"""

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral, Real
from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "notebooks/aimilia", ROOT / "notebooks/leonard"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


FEATURE_DESCRIPTIONS = {
    "aimilia_static": "Level-scaled team/lead base stats, composition and comparisons; no timeline.",
    "aimilia_timeline": "Active HP, early/late HP, fainting, switches, ailments and recorded move summaries.",
    "aimilia_effects": "Observed volatile-effect rates and offensive/defensive/speed boost summaries.",
    "aimilia_roster": "Latest-known team HP/alive counts at exact observed turns 10, 20 and 30; unseen slots full.",
    "aimilia_flow": "Same-species observed HP losses/healing, early/late damage and cumulative exchange.",
    "aimilia_matchup": "Recorded attack effectiveness/STAB using this battle's supplied species types only.",
    "aimilia_speed": "Approximate active speed/priority/paralysis advantage using this battle's metadata only.",
    "aimilia_damage": "Simplified move-power, accuracy, type and boosted-stat pressure; per-battle metadata only.",
    "aimilia_hazards": "Observed hazard/recovery moves, separate statuses and last-seen toxic/seeded exposure proxies.",
    "aimilia_moves": "Semantic move counts, sleep-with-action, RestTalk, cure and trapping observations.",
    "leonard_battle_state": "Last-seen living/injured/status summaries, recent comparable HP changes and active boosts.",
    "leonard_remaining_strength": "Own HP-weighted level-scaled strength, known opponent lead and observation-age summaries.",
    "leonard_trajectory": "Additional team-advantage checkpoints and recent changes; latest known HP is approximate.",
    "leonard_boost_threat": "HP-weighted offensive/speed stages, status interactions and recent observed threats.",
}


def sanitize_battle(battle):
    """Copy permitted observations, ignore labels/IDs and enforce turns 1–30.

    The first 30 supplied entries define the observation budget. Retained integer
    turns are sorted chronologically; a fainted state always has zero HP. No
    missing turn, Pokemon, move or species attribute is fabricated. Invalid
    observed state values raise rather than being hidden by missing-data fallbacks.
    """
    result = {
        "battle_id": 0,
        "p1_team_details": deepcopy(battle.get("p1_team_details") or []),
        "p2_lead_details": deepcopy(battle.get("p2_lead_details") or {}),
        "battle_timeline": [],
    }
    for source_turn in (battle.get("battle_timeline") or [])[:30]:
        number = source_turn.get("turn")
        if isinstance(number, bool) or not isinstance(number, Integral) or not 1 <= number <= 30:
            continue
        turn = deepcopy(source_turn)
        for side in ("p1", "p2"):
            state = turn.get(f"{side}_pokemon_state")
            if not isinstance(state, dict) or not state.get("name"):
                raise ValueError(f"Missing {side} state/name at observed turn {number}")
            hp = state.get("hp_pct")
            if state.get("status") == "fnt":
                state["hp_pct"] = 0.0
            elif isinstance(hp, bool) or not isinstance(hp, Real) or not math.isfinite(hp) or not 0 <= hp <= 1:
                raise ValueError(f"Invalid {side} HP at observed turn {number}")
        result["battle_timeline"].append(turn)
    result["battle_timeline"].sort(key=lambda turn: turn["turn"])
    return result


def _schema_fixture():
    """Synthetic schema definition, independent of any training observation."""
    stats = ("hp", "atk", "def", "spa", "spd", "spe")
    team = [{"name": f"own{i}", "level": 100, "item": "leftovers",
             "types": ["water", "notype"],
             **{f"base_{stat}": 100 for stat in stats}} for i in range(6)]
    lead = {**team[0], "name": "enemy", "types": ["fire", "notype"]}
    def state(name):
        return {"name": name, "hp_pct": 1.0, "status": "nostatus",
                "boosts": dict.fromkeys(("atk", "def", "spa", "spd", "spe"), 0),
                "effects": ["noeffect"]}
    turns = [{"turn": number, "p1_pokemon_state": state("own0"),
              "p2_pokemon_state": state("enemy"),
              "p1_move_details": None, "p2_move_details": None}
             for number in range(1, 31)]
    return {"battle_id": 0, "p1_team_details": team,
            "p2_lead_details": lead, "battle_timeline": turns}


@dataclass
class TeamFeatureGroup:
    """Named, fixed-schema adapter; unexpected upstream errors propagate."""
    name: str
    provider: object
    columns: tuple
    mode: str = "plain"
    chart: object = None

    def __call__(self, battle):
        observed = sanitize_battle(battle)
        turns = observed["battle_timeline"]
        numbers = {turn["turn"] for turn in turns}
        metadata_known = bool(observed["p1_team_details"] and observed["p2_lead_details"])
        # These are known provider preconditions, not exception-based recovery.
        applicable = True
        if self.name == "aimilia_static":
            applicable = metadata_known
        elif not turns:
            applicable = False
        elif self.name == "aimilia_timeline":
            applicable = any(t <= 10 for t in numbers) and any(t > 20 for t in numbers)
        elif self.name == "aimilia_roster":
            applicable = metadata_known and {10, 20, 30} <= numbers
        elif self.name in ("leonard_remaining_strength", "leonard_trajectory"):
            applicable = metadata_known
        if not applicable:
            return dict.fromkeys(self.columns, float("nan"))
        if self.mode == "static":
            row = self.provider(observed, labelled=False)
        elif self.mode == "chart_dex":
            row = self.provider(observed, self.chart, {})
        elif self.mode == "dex":
            row = self.provider(observed, {})
        else:
            row = self.provider(observed)
        clean = {key: float(value) for key, value in row.items()
                 if key not in ("battle_id", "player_won")}
        if set(clean) != set(self.columns):
            raise ValueError(f"{self.name} changed its feature schema")
        if any(math.isinf(value) for value in clean.values()):
            raise ValueError(f"{self.name} returned an infinite feature")
        return {key: clean[key] for key in self.columns}


def make_team_feature_groups(chart_path):
    """Return named callable dictionaries; runner prefixes their feature keys.

    Empty timelines preserve each group's fixed schema with NaNs, while valid
    static metadata can still produce static features. Missing exact roster
    checkpoints make only that checkpoint-dependent provider unavailable; no
    invented forward-filled states are passed to its implementation. Timeline
    summaries require at least one early and one late observation.
    """
    matchup = import_module("features_matchup")
    chart = matchup.read_chart(chart_path)
    definitions = [
        ("aimilia_static", import_module("features_static").battle_to_row, "static"),
        ("aimilia_timeline", import_module("features_timeline").battle_to_row, "plain"),
        ("aimilia_effects", import_module("features_effects").battle_to_row, "plain"),
        ("aimilia_roster", import_module("features_roster").battle_to_row, "plain"),
        ("aimilia_flow", import_module("features_flow").battle_to_row, "plain"),
        ("aimilia_matchup", matchup.battle_to_row, "chart_dex"),
        ("aimilia_speed", import_module("features_speed").battle_to_row, "dex"),
        ("aimilia_damage", import_module("features_damage").battle_to_row, "chart_dex"),
        ("aimilia_hazards", import_module("features_hazards").battle_to_row, "plain"),
        ("aimilia_moves", import_module("features_moves").battle_to_row, "plain"),
        ("leonard_battle_state", import_module("ablation_extensions").battle_state_features, "plain"),
        ("leonard_remaining_strength", import_module("progressive_features").remaining_strength, "plain"),
        ("leonard_trajectory", import_module("progressive_features").advantage_trajectory, "plain"),
        ("leonard_boost_threat", import_module("progressive_features").boost_threat, "plain"),
    ]
    fixture = _schema_fixture()
    groups = {}
    for name, provider, mode in definitions:
        if mode == "static":
            row = provider(fixture, labelled=False)
        elif mode == "chart_dex":
            row = provider(fixture, chart, {})
        elif mode == "dex":
            row = provider(fixture, {})
        else:
            row = provider(fixture)
        columns = tuple(key for key in row if key not in ("battle_id", "player_won"))
        groups[name] = TeamFeatureGroup(name, provider, columns, mode, chart)
    return groups
