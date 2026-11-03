"""Stateless, experimental candidates for Amir's existing 28-feature baseline.

Source ideas come from ``notebooks/aimilia/features_timeline.py`` (active HP,
ailments, moves and checkpoints), ``features_effects.py`` (effect exposure and
component boosts), ``features_roster.py`` (last-observed roster tracking), and
``2025/de-lucia-fratello-marchio.ipynb`` (healthy retained members).  HP-weighted
positive offensive boost is an explicit interaction of the observed final
active HP and offensive stages, rather than a claim about actual damage.

The Aimilia source paths are relative to the repository; the 2025 notebook is
in its parent workspace.  This module reads no files, labels, or fitted statistics.
It deliberately exports candidates for evaluation rather than a final choice.

Only the first 30 timeline entries are considered, followed by an explicit
turn-number filter to 1..30 and chronological ordering.  Missing numeric
observations contribute zero to active/move summaries.  Team estimates assume
unseen members are healthy at full HP, matching Amir's existing team-HP
convention; last-observed benched state may be stale.  Sleep is an ailment,
not a faint, and no hidden P2 species are invented.
"""

from __future__ import annotations

from collections.abc import Mapping
import math


SIDES = ("p1", "p2")
TEAM_SIZE = 6
MAX_TURNS = 30
EFFECTS = ("substitute", "trapped", "leechseed", "confusion")
SIDE_SUFFIXES = (
    "final_active_hp",
    "early_active_hp_mean",
    "late_active_hp_mean",
    "ailment_rate",
    "sleep_rate",
    "status_move_count",
    "move_accuracy_mean",
    "substitute_rate",
    "trapped_rate",
    "leechseed_rate",
    "confusion_rate",
    "offensive_boost_mean",
    "defensive_boost_mean",
    "speed_boost_mean",
    "alive_at_turn10",
    "alive_at_turn20",
    "late_first_observed_faints",
    "healthy_living_members",
    "final_hp_positive_offensive_boost",
)
CANDIDATE_COLUMNS = tuple(
    f"{side}_{suffix}" for side in SIDES for suffix in SIDE_SUFFIXES
) + ("late_ko_advantage",)


def _number(value, default=0.0):
    """Return a finite float without treating booleans as numeric states."""
    if isinstance(value, bool):
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _name(state):
    name = state.get("name")
    return name.strip().lower() if isinstance(name, str) and name.strip() else None


def _hp(value, default=0.0):
    return min(1.0, max(0.0, _number(value, default)))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _turns(battle):
    """Apply the observation boundary before sorting the retained entries."""
    timeline = battle.get("battle_timeline")
    if not isinstance(timeline, (list, tuple)):
        return []
    retained = []
    for entry in timeline[:MAX_TURNS]:
        entry = _mapping(entry)
        number = _number(entry.get("turn"), default=-1)
        if number.is_integer() and 1 <= number <= MAX_TURNS:
            retained.append((int(number), entry))
    return sorted(retained, key=lambda item: item[0])


def _initial_roster(battle, side):
    if side == "p1":
        pokemon = battle.get("p1_team_details") or []
        if not isinstance(pokemon, (list, tuple)):
            pokemon = []
    else:
        pokemon = [battle.get("p2_lead_details")]
    roster = {}
    for entry in pokemon:
        name = _name(_mapping(entry))
        if name:
            roster[name] = {"hp": 1.0, "status": "nostatus"}
    return roster


def extract_candidate_features(battle):
    """Return exactly 39 finite numeric candidates without using the outcome.

    Early HP is averaged over observed turns 1..10 and late HP over 21..30;
    these describe the active member, rather than team HP.  Effect, ailment,
    and sleep rates use all retained turns as denominator.  Move accuracy is
    averaged over recorded moves only, and no move yields zero.

    Alive checkpoint counts equal six minus distinct species first observed
    fainted by the checkpoint.  A late faint is a species first observed
    fainted on turns 21..30; it is counted once even if repeated.  The signed
    late KO gap is P2 late faints minus P1 late faints (positive favors P1).

    Healthy living members have last-known HP >0.5 and ``nostatus``, plus
    never-seen slots assumed healthy, capped at six.  Final HP-weighted
    offensive boost is active HP times the sum of positive atk/spa stages
    at the last retained observation.  Every output remains finite for
    empty or partially missing input.
    """
    battle = _mapping(battle)
    turns = _turns(battle)
    features = dict.fromkeys(CANDIDATE_COLUMNS, 0.0)

    for side in SIDES:
        roster = _initial_roster(battle, side)
        fainted = set()
        hp_early, hp_late, accuracies = [], [], []
        offensive, defensive, speed = [], [], []
        ailments = sleeps = status_moves = late_faints = 0
        effect_counts = dict.fromkeys(EFFECTS, 0)
        faint_turns = {}
        final_state = {}

        for turn_number, turn in turns:
            state = _mapping(turn.get(f"{side}_pokemon_state"))
            final_state = state
            name = _name(state)
            status = state.get("status", "nostatus")
            hp = _hp(state.get("hp_pct"))
            is_fainted = (state.get("hp_pct") is not None
                          and _number(state.get("hp_pct"), 1.0) <= 0) or status == "fnt"
            if name and is_fainted and name not in fainted:
                fainted.add(name)
                faint_turns[name] = turn_number
                late_faints += turn_number >= 21
            if name:
                member = roster.setdefault(name, {"hp": 1.0, "status": "nostatus"})
                if is_fainted:
                    member["hp"] = 0.0
                    member["status"] = "fnt"
                elif name not in fainted:
                    if state.get("hp_pct") is not None:
                        member["hp"] = _hp(state.get("hp_pct"), member["hp"])
                    if isinstance(status, str):
                        member["status"] = status

            if turn_number <= 10:
                hp_early.append(hp)
            if turn_number >= 21:
                hp_late.append(hp)
            ailments += isinstance(status, str) and status not in {"nostatus", "fnt"}
            sleeps += status == "slp"
            effects = state.get("effects") or []
            if not isinstance(effects, (list, tuple, set, frozenset)):
                effects = []
            for effect in EFFECTS:
                effect_counts[effect] += effect in effects

            boosts = _mapping(state.get("boosts"))
            offensive.append(_number(boosts.get("atk")) + _number(boosts.get("spa")))
            defensive.append(_number(boosts.get("def")) + _number(boosts.get("spd")))
            speed.append(_number(boosts.get("spe")))
            move = turn.get(f"{side}_move_details")
            if isinstance(move, Mapping):
                status_moves += move.get("category") == "status"
                accuracies.append(min(1.0, max(0.0, _number(move.get("accuracy")))))

        denominator = len(turns) or 1
        final_boosts = _mapping(final_state.get("boosts"))
        final_hp = _hp(final_state.get("hp_pct"))
        if final_state.get("status") == "fnt":
            final_hp = 0.0
        positive_offense = (max(0.0, _number(final_boosts.get("atk")))
                            + max(0.0, _number(final_boosts.get("spa"))))
        unseen = max(0, TEAM_SIZE - len(roster))
        healthy = unseen + sum(
            member["hp"] > 0.5 and member["status"] == "nostatus"
            for member in roster.values()
        )
        side_features = {
            "final_active_hp": final_hp,
            "early_active_hp_mean": _mean(hp_early),
            "late_active_hp_mean": _mean(hp_late),
            "ailment_rate": ailments / denominator,
            "sleep_rate": sleeps / denominator,
            "status_move_count": float(status_moves),
            "move_accuracy_mean": _mean(accuracies),
            "offensive_boost_mean": _mean(offensive),
            "defensive_boost_mean": _mean(defensive),
            "speed_boost_mean": _mean(speed),
            "alive_at_turn10": max(0, TEAM_SIZE - sum(t <= 10 for t in faint_turns.values())),
            "alive_at_turn20": max(0, TEAM_SIZE - sum(t <= 20 for t in faint_turns.values())),
            "late_first_observed_faints": float(late_faints),
            "healthy_living_members": min(TEAM_SIZE, healthy),
            "final_hp_positive_offensive_boost": final_hp * positive_offense,
            **{f"{effect}_rate": count / denominator
               for effect, count in effect_counts.items()},
        }
        for suffix, value in side_features.items():
            features[f"{side}_{suffix}"] = _number(value)

    features["late_ko_advantage"] = (
        features["p2_late_first_observed_faints"]
        - features["p1_late_first_observed_faints"]
    )
    return features
