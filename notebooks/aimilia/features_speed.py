"""
Step 7d: who moves first.

THE GAP THIS FILLS
------------------
Speed decides turn order, and turn order decides who lands the knockout
blow first. It is the most consequential stat in the game. We currently have
`p1_spe_mean` (the whole team's average) and `p2_lead_spe` (one Pokemon),
which tells us almost nothing about the fight actually happening: what
matters is whether the Pokemon ON THE FIELD is faster than the one it is
facing, turn after turn.

Three things change effective speed during a battle and all three are in the
data:

  boosts["spe"]  - stat stages. Each positive stage multiplies by (2+n)/2,
                   each negative one by 2/(2-n). A +2 Pokemon is twice as
                   fast as it started.
  status "par"   - paralysis. In Generation 2 this cuts Speed to a QUARTER
                   (it only became a half in Gen 7). Crippling, and our
                   single `status_rate` column cannot distinguish it from
                   being poisoned.
  level          - handled by the species lookup, which scales base stats.

PRIORITY
--------
`move_details.priority` is the other half of turn order and we have never
read it either. A priority move (Quick Attack, Mach Punch) goes first
regardless of speed, so a slow side leaning on priority is not as helpless
as its speed suggests.

Usage:
    from features_speed import build_speed
    sp_train = build_speed(train_data, DEX)
"""

import pandas as pd

from pokedex import battle_lookup

LATE = 20


def boost_multiplier(stage):
    """Gen-2 stat-stage multiplier. 0 means unchanged."""
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def active_speed(state, resolve):
    """Effective Speed of the Pokemon on the field, or None if unresolvable."""
    entry = resolve(state["name"])
    if entry is None:
        return None
    speed = entry[1]["spe"] * boost_multiplier(state["boosts"]["spe"])
    if state["status"] == "par":
        speed *= 0.25
    return speed


def battle_to_row(battle, dex):
    resolve = battle_lookup(battle, dex)
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}

    speeds1, speeds2 = [], []
    faster, tied, compared = 0, 0, 0
    faster_late, compared_late = 0, 0
    priority = {"p1": 0, "p2": 0}
    acted = {"p1": 0, "p2": 0}

    for turn in turns:
        s1 = active_speed(turn["p1_pokemon_state"], resolve)
        s2 = active_speed(turn["p2_pokemon_state"], resolve)
        if s1 is not None:
            speeds1.append(s1)
        if s2 is not None:
            speeds2.append(s2)
        if s1 is not None and s2 is not None:
            compared += 1
            if s1 > s2:
                faster += 1
            elif s1 == s2:
                tied += 1
            if turn["turn"] > LATE:
                compared_late += 1
                if s1 > s2:
                    faster_late += 1

        for side in ("p1", "p2"):
            move = turn[f"{side}_move_details"]
            if move is None:
                continue
            acted[side] += 1
            if (move.get("priority") or 0) > 0:
                priority[side] += 1

    nan = float("nan")
    row["p1_active_spe_mean"] = sum(speeds1) / len(speeds1) if speeds1 else nan
    row["p2_active_spe_mean"] = sum(speeds2) / len(speeds2) if speeds2 else nan
    row["speed_gap_mean"] = (row["p1_active_spe_mean"] - row["p2_active_spe_mean"]
                             if speeds1 and speeds2 else nan)
    # The headline column: share of turns p1's Pokemon moved first.
    row["p1_speed_adv_rate"] = faster / compared if compared else nan
    row["p1_speed_adv_late"] = faster_late / compared_late if compared_late else nan
    # Speed ties are resolved by a coin flip, so a high tie rate means the
    # battle was genuinely unpredictable rather than controlled by either side.
    row["speed_tie_rate"] = tied / compared if compared else nan
    row["p1_priority_rate"] = priority["p1"] / acted["p1"] if acted["p1"] else 0.0
    row["p2_priority_rate"] = priority["p2"] / acted["p2"] if acted["p2"] else 0.0

    return row


def build_speed(battles, dex):
    """Return one row of turn-order features per battle."""
    return pd.DataFrame([battle_to_row(b, dex) for b in battles])
