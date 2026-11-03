"""Four selected, stateless summaries of the observed battle position.

The healthy-roster and retained-bulk ideas originate in
``2025/de-lucia-fratello-marchio.ipynb`` (cell 7).  HP-weighted base special
attack extends its remaining-resource idea.  Leech Seed exposure comes from
``notebooks/aimilia/features_effects.py``.  These are heuristic summaries of
observations, not actual battle stats: benched HP/status may be stale and
unseen P2 slots are assumed healthy.  No hidden species or stats are inferred.
"""

from collections.abc import Mapping
from math import isfinite


NEW_FEATURE_COLUMNS = (
    "p2_healthy_living_members",
    "p1_remaining_special_bulk",
    "p1_hp_weighted_base_special_attack",
    "p1_leechseed_rate",
)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _name(observation):
    value = observation.get("name")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _retained_turns(battle):
    """Limit entry count first, retain integer turns 1..30, then order them."""
    timeline = battle.get("battle_timeline")
    if not isinstance(timeline, (list, tuple)):
        return []
    retained = []
    for entry in timeline[:30]:
        entry = _mapping(entry)
        number = _number(entry.get("turn"))
        if number is not None and number.is_integer() and 1 <= number <= 30:
            retained.append((number, entry))
    return sorted(retained, key=lambda item: item[0])


def _finite_output(value):
    number = _number(value)
    return number if number is not None else 0.0


def retained_position_features(battle):
    """Return four finite numeric features, using only the observed prefix.

    * P2 healthy living members: latest-known HP strictly greater than 0.5
      and status exactly ``nostatus``, plus unseen slots in a six-member team,
      capped at six.  The lead starts full and healthy.  Missing observed
      status is unknown; a new observed species with missing HP is unknown.
    * P1 remaining special bulk: sum of supplied ``base_hp * base_spd * HP``.
      Supplied roster members start at full HP; a faint forces HP to zero.
    * P1 HP-weighted base special attack: ``sum(base_spa * HP) / sum(HP)``
      over supplied members with available numeric ``base_spa``.  Zero total
      eligible HP produces zero.  Sleep/freeze do not count as fainting.
    * P1 Leech Seed rate: fraction of retained rows whose P1 effects contain
      ``leechseed``.  Every valid retained row contributes to the denominator,
      including rows with missing state/effects.  An empty prefix gives zero.

    The first 30 entries are selected before filtering turn numbers to 1..30;
    retained rows are processed chronologically.  HP observations are clipped
    to 0..1 and repeated species update one roster entry.  Missing HP preserves
    the previous known HP.  This function has no label/ID access, fitted state,
    external lookup, or cross-battle memory.
    """
    battle = _mapping(battle)
    supplied = battle.get("p1_team_details") or []
    if not isinstance(supplied, (list, tuple)):
        supplied = []
    roster = {}
    for member in supplied:
        member = _mapping(member)
        name = _name(member)
        if name:
            roster[name] = member

    states = {
        "p1": {name: {"hp": 1.0, "status": "nostatus"} for name in roster},
        "p2": {},
    }
    lead_name = _name(_mapping(battle.get("p2_lead_details")))
    if lead_name:
        states["p2"][lead_name] = {"hp": 1.0, "status": "nostatus"}

    turns = _retained_turns(battle)
    leechseed_rows = 0
    for _, entry in turns:
        for side in ("p1", "p2"):
            observed = _mapping(entry.get(f"{side}_pokemon_state"))
            name = _name(observed)
            if name:
                latest = states[side].setdefault(name, {"hp": None, "status": None})
                hp = _number(observed.get("hp_pct"))
                if hp is not None:
                    latest["hp"] = min(1.0, max(0.0, hp))
                latest["status"] = observed.get("status")
                if latest["status"] == "fnt":
                    latest["hp"] = 0.0
        effects = _mapping(entry.get("p1_pokemon_state")).get("effects") or []
        if isinstance(effects, (list, tuple, set, frozenset)):
            leechseed_rows += "leechseed" in effects

    p2_members = list(states["p2"].values())
    unseen = max(0, 6 - len(p2_members))
    healthy = min(6, unseen + sum(
        member["hp"] is not None and member["hp"] > 0.5
        and member["status"] == "nostatus"
        for member in p2_members
    ))

    bulk_terms = []
    weighted_attack = []
    for name, member in roster.items():
        latest = states["p1"][name]
        hp = 0.0 if latest["status"] == "fnt" else (latest["hp"] or 0.0)
        bulk_terms.append((_number(member.get("base_hp")) or 0.0)
                          * (_number(member.get("base_spd")) or 0.0) * hp)
        special_attack = _number(member.get("base_spa"))
        if special_attack is not None:
            weighted_attack.append((special_attack, hp))

    bulk = sum(bulk_terms)
    attack_total = sum(value * hp for value, hp in weighted_attack)
    attack_weight = sum(hp for _, hp in weighted_attack)

    values = {
        "p2_healthy_living_members": float(healthy),
        "p1_remaining_special_bulk": bulk,
        "p1_hp_weighted_base_special_attack": (
            attack_total / attack_weight if attack_weight > 0 else 0.0
        ),
        "p1_leechseed_rate": leechseed_rows / len(turns) if turns else 0.0,
    }
    return {name: _finite_output(values[name]) for name in NEW_FEATURE_COLUMNS}
