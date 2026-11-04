"""
Step 7e: estimated damage per turn.

THE GAP THIS FILLS
------------------
This is the densest feature on the list: it folds base stats, level, types,
stat boosts, burn, move power and move accuracy into ONE number per turn -
roughly how hard that attack should have hit. Everything else we have
measures the position a side is in; this measures the pressure it is
applying, which is what produces the position in the first place.

THE ESTIMATE
------------
A deliberately crude stand-in for the real Gen-2 damage formula:

    base_power x accuracy x effectiveness x STAB x (offence / defence)

  accuracy       expected damage, not best case: a 70%-accurate 120-power
                 move is worth less than a certain 100-power one.
  effectiveness  the type chart, same helper as features_matchup.
  STAB           1.5x when the move's type matches one of the attacker's.
  offence        attacker's atk (physical moves) or spa (special moves),
                 level-scaled and boost-adjusted. Halved when burned,
                 which in Gen 2 cuts physical attack.
  defence        defender's def or spd, matching the move's category,
                 likewise boost-adjusted.

It omits the real formula's level term, random spread and critical hits. It
does not need to be right in absolute terms - it only needs to ORDER turns
by how much damage they probably did, and it is far closer to the truth than
base_power on its own. The result is divided by 100 to keep the numbers on a
sane scale.

Status moves score 0: they are covered by features_effects.

Usage:
    from features_damage import build_damage
    dm_train = build_damage(train_data, CHART_PATH, DEX)
"""

import pandas as pd

from pokedex import battle_lookup
from features_matchup import read_chart, effectiveness
from features_speed import boost_multiplier

LATE = 20


def expected_damage(move, attacker_state, defender_state, resolve, chart):
    """Estimated damage of one attack, or None if we cannot resolve it."""
    attacker = resolve(attacker_state["name"])
    defender = resolve(defender_state["name"])
    if attacker is None or defender is None:
        return None

    power = move.get("base_power") or 0
    if power <= 0:
        return 0.0

    multiplier = effectiveness(move.get("type"), defender[0], chart)
    if multiplier is None:
        return None
    stab = 1.5 if move.get("type") in attacker[0] else 1.0

    if move.get("category") == "physical":
        offence = attacker[1]["atk"] * boost_multiplier(attacker_state["boosts"]["atk"])
        if attacker_state["status"] == "brn":
            offence *= 0.5
        defence = defender[1]["def"] * boost_multiplier(defender_state["boosts"]["def"])
    else:
        offence = attacker[1]["spa"] * boost_multiplier(attacker_state["boosts"]["spa"])
        defence = defender[1]["spd"] * boost_multiplier(defender_state["boosts"]["spd"])

    accuracy = move.get("accuracy")
    accuracy = 1.0 if accuracy is None else accuracy

    return power * accuracy * multiplier * stab * offence / max(defence, 1.0) / 100.0


def side_damage(turns, side, other, resolve, chart):
    """Summarise the damage one side was estimated to deal across the battle."""
    values = []
    late_values = []

    for turn in turns:
        move = turn[f"{side}_move_details"]
        if move is None:
            continue
        value = expected_damage(move, turn[f"{side}_pokemon_state"],
                                turn[f"{other}_pokemon_state"], resolve, chart)
        if value is None:
            continue
        values.append(value)
        if turn["turn"] > LATE:
            late_values.append(value)

    if not values:
        nan = float("nan")
        return {"edmg_sum": nan, "edmg_mean": nan, "edmg_max": nan,
                "edmg_late_mean": nan}

    return {
        # Total pressure applied over the 30 turns.
        "edmg_sum": sum(values),
        "edmg_mean": sum(values) / len(values),
        # The single biggest threat this side demonstrated.
        "edmg_max": max(values),
        "edmg_late_mean": (sum(late_values) / len(late_values)
                           if late_values else float("nan")),
    }


def battle_to_row(battle, chart, dex):
    resolve = battle_lookup(battle, dex)
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}

    for side, other in (("p1", "p2"), ("p2", "p1")):
        for key, value in side_damage(turns, side, other, resolve, chart).items():
            row[f"{side}_{key}"] = value

    row["edmg_surplus"] = row["p1_edmg_sum"] - row["p2_edmg_sum"]
    row["edmg_surplus_late"] = row["p1_edmg_late_mean"] - row["p2_edmg_late_mean"]
    total = row["p1_edmg_sum"] + row["p2_edmg_sum"]
    # Scale-free version: 0.5 is an even fight regardless of how violent it was.
    row["edmg_share"] = row["p1_edmg_sum"] / total if total and total > 0 else 0.5

    return row


def build_damage(battles, chart_path, dex):
    """Return one row of estimated-damage features per battle."""
    chart = read_chart(chart_path)
    return pd.DataFrame([battle_to_row(b, chart, dex) for b in battles])
