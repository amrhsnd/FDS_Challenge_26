"""Additional, observation-only feature hypotheses for Amir's notebook.

Sources: 2025/de-lucia-fratello-marchio.ipynb (roster dispersion, status and
retained resources); Aimilia's features_speed.py and features_flow.py (speed
and recent flow); Leonard's progressive_features.py (active boost/HP
interactions). These are rewritten with fixed schemas and missing-data guards.

Every function is stateless, ignores labels/IDs, and processes only integer
turns 1..30 in chronological order. Duplicate turns use the last supplied row.
No external lookup, cross-battle species statistics or unknown P2 roster is
inferred. P1 supplied members and the P2 supplied lead start at full HP and
healthy, with last-seen bench states potentially stale. Unknown numeric values
are NaN, for imputation INSIDE each training fold. HP-change features describe
observations, not damage attributable to a particular move.
"""

from collections.abc import Mapping
from math import isfinite, nan, sqrt

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
STATUSES = ("slp", "frz", "par", "brn", "psn", "tox")
SIDES = ("p1", "p2")
WINDOWS = ((1, 10), (11, 20), (21, 30))


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _sequence(value):
    return value if isinstance(value, (list, tuple)) else ()


def _number(value):
    if value is None or isinstance(value, bool):
        return nan
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return nan
    return result if isfinite(result) else nan


def _name(state):
    name = _mapping(state).get("name")
    return name.strip().lower() if isinstance(name, str) and name.strip() else None


def _hp(state):
    state = _mapping(state)
    if state.get("status") == "fnt":
        return 0.0
    value = _number(state.get("hp_pct"))
    return min(1.0, max(0.0, value)) if isfinite(value) else nan


def observed_turns(battle):
    """Return at most 30 chronologically ordered, unique, valid turn rows."""
    by_turn = {}
    for entry in _sequence(_mapping(battle).get("battle_timeline")):
        entry = _mapping(entry)
        number = _number(entry.get("turn"))
        if isfinite(number) and number.is_integer() and 1 <= number <= 30:
            by_turn[int(number)] = entry
    return [(number, by_turn[number]) for number in sorted(by_turn)]


def _rosters(battle):
    battle = _mapping(battle)
    own = {_name(p): _mapping(p) for p in _sequence(battle.get("p1_team_details")) if _name(p)}
    lead = _mapping(battle.get("p2_lead_details"))
    return {"p1": own, "p2": {_name(lead): lead} if _name(lead) else {}}


def _mean(values):
    values = [v for v in values if isfinite(v)]
    return sum(values) / len(values) if values else nan


def _std(values):
    values = [v for v in values if isfinite(v)]
    return sqrt(sum((v - _mean(values)) ** 2 for v in values) / len(values)) if values else nan


def _difference(row, suffixes):
    for suffix in suffixes:
        row[f"gap_{suffix}"] = row[f"p1_{suffix}"] - row[f"p2_{suffix}"]
    return row


def roster_shape_features(battle):
    """P1 base-stat spread and asymmetrically available lead comparisons.

    Stats are supplied base stats, without claiming actual battle-stat values.
    Item availability comes from the supplied P1 roster only. Ratios exclude
    members missing either required stat; missing rosters produce NaN ratios.
    """
    rosters = _rosters(battle)
    members = list(rosters["p1"].values())
    lead = next(iter(rosters["p2"].values()), {})
    row = {"p1_supplied_members": float(len(members))}
    for stat in STATS:
        values = [_number(p.get(f"base_{stat}")) for p in members]
        valid = [v for v in values if isfinite(v)]
        row[f"p1_base_{stat}_std"] = _std(valid)
        row[f"p1_base_{stat}_min"] = min(valid, default=nan)
        row[f"p1_base_{stat}_max"] = max(valid, default=nan)
        row[f"p1_mean_minus_p2_lead_{stat}"] = _mean(valid) - _number(lead.get(f"base_{stat}"))
    levels = [_number(p.get("level")) for p in members]
    row["p1_level_mean"] = _mean(levels)
    row["p1_level_std"] = _std(levels)
    row["p2_supplied_lead_level"] = _number(lead.get("level"))
    for defense, label in (("def", "physical"), ("spd", "special")):
        bulk = [_number(p.get("base_hp")) * _number(p.get(f"base_{defense}")) for p in members]
        row[f"p1_initial_{label}_bulk_mean"] = _mean(bulk)
        row[f"p1_initial_{label}_bulk_std"] = _std(bulk)
    ratios = [_number(p.get("base_spa")) / _number(p.get("base_atk"))
              for p in members if _number(p.get("base_atk")) > 0]
    row["p1_special_to_physical_attack_mean"] = _mean(ratios)
    speeds = [_number(p.get("base_spe")) for p in members]
    row["p1_base_speed_ge100_count"] = float(sum(v >= 100 for v in speeds if isfinite(v)))
    supplied_items = [str(p["item"]).lower() for p in members if p.get("item") is not None]
    row["p1_item_known_count"] = float(len(supplied_items))
    row["p1_leftovers_fraction_known_items"] = _mean([float(item == "leftovers") for item in supplied_items])
    row["p1_berry_fraction_known_items"] = _mean([float("berry" in item) for item in supplied_items])
    return row


def _retained_states(battle):
    rosters = _rosters(battle)
    latest = {side: {name: {"hp": 1.0, "status": "nostatus", "seen": None}
                     for name in rosters[side]} for side in SIDES}
    turns = observed_turns(battle)
    for number, turn in turns:
        for side in SIDES:
            state = _mapping(turn.get(f"{side}_pokemon_state"))
            name = _name(state)
            if not name:
                continue
            retained = latest[side].setdefault(name, {"hp": nan, "status": None, "seen": None})
            hp = _hp(state)
            if isfinite(hp):
                retained["hp"] = hp
            # Missing status is unknown rather than silently healthy.
            retained["status"] = state.get("status")
            retained["seen"] = number
    return turns, latest


def status_resource_features(battle):
    """Last-known status/HP plus active-status exposure and uncertainty.

    P2 counts/sums cover only its supplied lead and actually revealed species;
    its unobserved slots are counted explicitly and contribute no HP or status.
    Unseen P1 supplied roster members start healthy/full. Status exposure rates
    use rows with an identified active Pokemon; missing status stays in that
    denominator and also receives an explicit unknown-status rate.
    """
    turns, latest = _retained_states(battle)
    horizon = turns[-1][0] if turns else 0
    row = {}
    gap_suffixes = []
    for side in SIDES:
        states = list(latest[side].values())
        alive = [s for s in states if isfinite(s["hp"]) and s["hp"] > 0]
        observed = [s for s in states if s["seen"] is not None]
        active = [_mapping(t.get(f"{side}_pokemon_state")) for _, t in turns]
        active = [s for s in active if _name(s)]
        for status in STATUSES:
            row[f"{side}_retained_{status}_count"] = float(sum(s["status"] == status for s in alive))
            row[f"{side}_active_{status}_rate"] = _mean([float(s.get("status") == status) for s in active])
        sums = {
            "retained_healthy_hp": sum(s["hp"] for s in alive if s["status"] == "nostatus"),
            "retained_status_hp": sum(s["hp"] for s in alive if s["status"] in STATUSES),
            "retained_incapacitated_hp": sum(s["hp"] for s in alive if s["status"] in ("slp", "frz")),
            "retained_poison_hp": sum(s["hp"] for s in alive if s["status"] in ("psn", "tox")),
        }
        for suffix, value in sums.items():
            row[f"{side}_{suffix}"] = value
        gap_suffixes = list(sums)
        row[f"{side}_revealed_members"] = float(len(observed))
        row[f"{side}_retained_unknown_status_count"] = float(sum(s["status"] not in (*STATUSES, "nostatus", "fnt") for s in states))
        row[f"{side}_retained_unknown_hp_count"] = float(sum(not isfinite(s["hp"]) for s in states))
        row[f"{side}_active_unknown_status_rate"] = _mean([float(s.get("status") not in (*STATUSES, "nostatus", "fnt")) for s in active])
        ages = [float(horizon - s["seen"]) for s in observed]
        row[f"{side}_observation_age_mean"] = _mean(ages)
        row[f"{side}_observation_age_max"] = max(ages, default=nan)
    row["p1_supplied_unobserved_members"] = float(sum(s["seen"] is None for s in latest["p1"].values()))
    row["p2_unrevealed_slots"] = float(max(0, 6 - len(latest["p2"])))
    # These differences retain the asymmetric coverage of the source values.
    return _difference(row, gap_suffixes)


def recent_flow_features(battle):
    """Three non-overlapping ten-turn windows of observed battle flow.

    HP losses/gains compare repeated sightings of the same member; supplied
    P1 members and the P2 lead use initial HP=1, while first sightings of other
    P2 members establish a baseline without inventing their prior HP. Changes
    across a gap in sightings can include unobserved time and are assigned to
    the later sighting. Switch counts require adjacent observed turn rows with
    valid active names; no recorded move can mean a switch or missing action.
    Faints count newly observed species, once per battle, even if repeated.
    """
    rosters = _rosters(battle)
    previous_hp = {side: {name: 1.0 for name in rosters[side]} for side in SIDES}
    previous_active = dict.fromkeys(SIDES)
    fainted = {side: set() for side in SIDES}
    suffixes = ("hp_loss", "hp_gain", "switches", "new_faints", "recorded_moves",
                "damaging_moves", "special_moves", "priority_moves", "rows")
    accum = {(side, stop): dict.fromkeys(suffixes, 0.0) for side in SIDES for _, stop in WINDOWS}
    last_number = None
    for number, turn in observed_turns(battle):
        stop = 10 if number <= 10 else 20 if number <= 20 else 30
        for side in SIDES:
            values = accum[side, stop]
            values["rows"] += 1
            state = _mapping(turn.get(f"{side}_pokemon_state"))
            name = _name(state)
            hp = _hp(state)
            if name:
                previous = previous_hp[side].get(name, nan)
                if isfinite(hp) and isfinite(previous):
                    values["hp_loss"] += max(previous - hp, 0.0)
                    values["hp_gain"] += max(hp - previous, 0.0)
                if isfinite(hp):
                    previous_hp[side][name] = hp
                if hp == 0 and name not in fainted[side]:
                    values["new_faints"] += 1
                    fainted[side].add(name)
                if last_number is not None and number == last_number + 1 and previous_active[side] is not None:
                    values["switches"] += previous_active[side] != name
            previous_active[side] = name
            move = turn.get(f"{side}_move_details")
            if isinstance(move, Mapping):
                values["recorded_moves"] += 1
                values["damaging_moves"] += move.get("category") in ("physical", "special")
                values["special_moves"] += move.get("category") == "special"
                values["priority_moves"] += _number(move.get("priority")) > 0
        last_number = number
    row = {}
    for start, stop in WINDOWS:
        label = f"t{start}_{stop}"
        for side in SIDES:
            values = accum[side, stop]
            for suffix in ("hp_loss", "hp_gain", "switches", "new_faints", "rows"):
                row[f"{side}_{label}_{suffix}"] = values[suffix]
            rows = values["rows"]
            moves = values["recorded_moves"]
            row[f"{side}_{label}_no_move_rate"] = 1 - moves / rows if rows else nan
            for suffix in ("damaging_moves", "special_moves", "priority_moves"):
                row[f"{side}_{label}_{suffix}_rate"] = values[suffix] / moves if moves else 0.0
        for metric in ("hp_loss", "hp_gain", "switches", "new_faints"):
            row[f"gap_{label}_{metric}"] = row[f"p1_{label}_{metric}"] - row[f"p2_{label}_{metric}"]
    for side in SIDES:
        for metric in ("hp_loss", "hp_gain"):
            row[f"{side}_{metric}_late_minus_early"] = row[f"{side}_t21_30_{metric}"] - row[f"{side}_t1_10_{metric}"]
    return row


def _stage_multiplier(stage):
    if not isfinite(stage):
        return nan
    stage = min(6.0, max(-6.0, stage))
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def active_position_features(battle):
    """Final active position, boost/HP interactions and supplied-stat proxies.

    Uses the last observed valid turn, which need not be turn 30. Only P1's
    supplied members and P2's supplied lead have stat/level proxies. Unknown
    species have NaN proxies and a zero known-stat flag. Speed is a heuristic
    base-speed * level/100 * stage multiplier, with Gen-2 paralysis factor 1/4;
    physical attack uses burn factor 1/2. Neither is an exact battle stat or a
    turn-order claim. Final active boosts avoid retaining switched-out boosts.
    """
    turns = observed_turns(battle)
    final_number, final = turns[-1] if turns else (0, {})
    rosters = _rosters(battle)
    row = {"final_observed_turn": float(final_number), "observed_turn_count": float(len(turns))}
    for side in SIDES:
        state = _mapping(final.get(f"{side}_pokemon_state"))
        member = rosters[side].get(_name(state), {})
        hp = _hp(state)
        boosts = _mapping(state.get("boosts"))
        row[f"{side}_active_hp"] = hp
        row[f"{side}_active_stats_supplied"] = float(bool(member))
        for stat in BOOST_STATS:
            stage = _number(boosts.get(stat))
            row[f"{side}_active_{stat}_stage"] = stage
            row[f"{side}_active_{stat}_hp_stage"] = hp * stage
        level = _number(member.get("level")) / 100
        for stat, label in (("atk", "physical_attack_proxy"), ("spa", "special_attack_proxy"), ("spe", "speed_proxy")):
            proxy = _number(member.get(f"base_{stat}")) * level * _stage_multiplier(_number(boosts.get(stat)))
            if stat == "spe" and state.get("status") == "par":
                proxy *= 0.25
            if stat == "atk" and state.get("status") == "brn":
                proxy *= 0.5
            if hp == 0:
                proxy = 0.0 if member else nan
            row[f"{side}_active_{label}"] = proxy
            if stat in ("atk", "spa"):
                row[f"{side}_active_hp_weighted_{label}"] = hp * proxy
    suffixes = ("active_hp", "active_atk_hp_stage", "active_spa_hp_stage", "active_spe_hp_stage",
                "active_speed_proxy", "active_hp_weighted_physical_attack_proxy",
                "active_hp_weighted_special_attack_proxy")
    return _difference(row, suffixes)


FEATURE_GROUPS = {
    "roster_shape": roster_shape_features,
    "status_resources": status_resource_features,
    "recent_flow": recent_flow_features,
    "active_position": active_position_features,
}

FEATURE_DESCRIPTIONS = {
    "roster_shape": "P1 base-stat dispersion/extrema, supplied P2 lead differences, level/item and initial bulk summaries (2025 roster ideas).",
    "status_resources": "Latest-known status/HP resources and active-status exposure, with explicit revealed-roster and observation-age coverage (2025/Aimilia).",
    "recent_flow": "Observed HP losses/gains, newly seen faints, switching and move profiles in turns 1–10, 11–20 and 21–30 (Aimilia/2025).",
    "active_position": "Final active HP, stat stages, HP-stage interactions and supplied-stat attack/speed proxies (Leonard/Aimilia).",
}
