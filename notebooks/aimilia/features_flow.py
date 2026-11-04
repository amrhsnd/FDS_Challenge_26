"""
Step 7b: HP flow - damage actually dealt, taken and healed.

THE GAP THIS FILLS
------------------
Every HP feature we have so far is a LEVEL: hp_pct now, team HP at turn 30.
None of them is a FLOW. That matters because a fainted Pokemon is replaced
by a fresh one at full health, which resets the visible HP. Two sides can
look identical at turn 30 and have had completely different battles:

  Side A: knocked out three of the opponent's Pokemon, lost none,
          currently has a healthy Pokemon on the field.
  Side B: never landed a serious hit, currently has a healthy Pokemon
          on the field.

`hppct_final` is 1.0 for both. `teamhp_t30` catches part of it, but still
cannot see HOW MUCH healing kept a side alive, or how lopsided the exchange
has been turn by turn.

HOW
---
Walk the 30 turns keeping the last HP we saw for every species on a side.
Each turn, the active Pokemon's change since we last saw it is either
damage taken (negative) or healing (positive). A Pokemon off the field
takes no damage, so attributing the whole change to the turn it reappears
is fine for cumulative sums.

A Pokemon we have never seen is assumed to enter at full health - the same
assumption features_roster makes, so the numbers stay comparable. Any HP it
is missing on first sight therefore counts as damage taken (Spikes, or a
hit landed on the turn it switched in).

Damage DEALT by a side is damage TAKEN by the other, so we do not store it
twice; only the difference is kept, which is the part that carries signal.

Usage:
    from features_flow import build_flow
    fl_train = build_flow(train_data)
"""

import pandas as pd

EARLY = 10        # turns 1-10
LATE = 20         # turns 21-30
BIG_HIT = 0.3     # a single-turn drop this large is a serious hit


def side_flow(turns, side):
    """Summarise one side's HP flow. Returns (features, per_turn_damage_taken)."""
    last_seen = {}
    taken = healed = 0.0
    taken_early = taken_late = 0.0
    worst = 0.0
    big_hits = 0
    per_turn = []

    for turn in turns:
        state = turn[f"{side}_pokemon_state"]
        name, hp = state["name"], state["hp_pct"]
        # Unseen Pokemon are assumed to have been at full health.
        previous = last_seen.get(name, 1.0)
        change = hp - previous
        last_seen[name] = hp

        drop = -change if change < 0 else 0.0
        gain = change if change > 0 else 0.0
        taken += drop
        healed += gain
        if turn["turn"] <= EARLY:
            taken_early += drop
        if turn["turn"] > LATE:
            taken_late += drop
        if drop > worst:
            worst = drop
        if drop > BIG_HIT:
            big_hits += 1
        per_turn.append(drop)

    features = {
        # Total health lost across the battle, in units of "whole Pokemon".
        "dmg_taken": taken,
        # Health regained - Leftovers ticks, Recover, Rest. A side surviving
        # on recovery is in a very different position from one that is simply
        # not being attacked.
        "healed": healed,
        "net_hp": healed - taken,
        "dmg_taken_early": taken_early,
        # The last ten turns are closest to the hidden turns we predict into.
        "dmg_taken_late": taken_late,
        "worst_hit": worst,
        "big_hits": big_hits,
    }
    return features, per_turn


def battle_to_row(battle):
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}

    f1, series1 = side_flow(turns, "p1")
    f2, series2 = side_flow(turns, "p2")
    for key, value in f1.items():
        row[f"p1_{key}"] = value
    for key, value in f2.items():
        row[f"p2_{key}"] = value

    # --- the exchange ----------------------------------------------------
    # p1's damage output is p2's health loss. We keep only the comparison,
    # not a second copy of the same column.
    row["dmg_surplus"] = f2["dmg_taken"] - f1["dmg_taken"]
    row["dmg_surplus_late"] = f2["dmg_taken_late"] - f1["dmg_taken_late"]
    total = f1["dmg_taken"] + f2["dmg_taken"]
    # A ratio rather than a difference: scale-free, so a quiet stall war and
    # a bloodbath are put on the same footing. 0.5 means a dead-even trade.
    row["dmg_share"] = f2["dmg_taken"] / total if total > 0 else 0.5
    row["heal_diff"] = f1["healed"] - f2["healed"]

    # How long p1 has been ahead on the exchange, not just by how much at the
    # end. Running surplus summed over the 30 turns, divided by the number of
    # turns to keep it on the same scale as the other columns.
    cumulative1 = cumulative2 = 0.0
    integral = 0.0
    for drop1, drop2 in zip(series1, series2):
        cumulative1 += drop1
        cumulative2 += drop2
        integral += cumulative2 - cumulative1
    row["dmg_surplus_integral"] = integral / len(turns)

    return row


def build_flow(battles):
    """Return one row of HP-flow features per battle."""
    return pd.DataFrame([battle_to_row(b) for b in battles])
