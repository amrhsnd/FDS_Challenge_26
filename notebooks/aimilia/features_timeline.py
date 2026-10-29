"""
Step 2: timeline (in-battle) features.

Summarises the 30 recorded turns into one flat row per battle: how the battle
was actually going, rather than who brought what.

Every battle has exactly 30 turns with no padding, and the battle ends AFTER
turn 30 - the outcome is not contained in the data. So these features describe
a snapshot of a battle in progress, not a finished one.

Usage:
    from features_timeline import build_timeline
    tl_train = build_timeline("data/train.jsonl")
    tl_test  = build_timeline("data/test.jsonl")
"""

import json
import pandas as pd

ALIVE_STATUSES = {"nostatus"}   # 'fnt' means fainted; everything else is an ailment
EARLY = 10                       # turns 1-10
LATE = 20                        # turns 21-30


def side_features(turns, side):
    """
    Summarise one player's 30 turns. `side` is 'p1' or 'p2'.

    Note: each turn records only the ACTIVE Pokemon's state, not the whole team.
    So hp_pct jumps around as Pokemon switch in and out - it is the health of
    whoever is on the field, not a team health bar.
    """
    f = {}
    states = [t[f"{side}_pokemon_state"] for t in turns]
    moves = [t[f"{side}_move_details"] for t in turns]
    hp = [s["hp_pct"] for s in states]
    names = [s["name"] for s in states]

    # --- the scoreboard ---------------------------------------------------
    # Count DISTINCT species that fainted, not fainted turns. A species shows
    # as 'fnt' on exactly one turn then gets replaced (verified: 21,922 faint
    # events, all single-turn, and no species ever appears alive afterwards).
    f["faints"] = len({s["name"] for s in states if s["status"] == "fnt"})

    # --- health of the active Pokemon -------------------------------------
    f["hppct_mean"] = sum(hp) / len(hp)
    f["hppct_final"] = hp[-1]
    f["hppct_min"] = min(hp)
    # Momentum: is this side healthier or sicker than it was at the start?
    # A single snapshot cannot distinguish "steady at 60%" from "collapsing
    # from 100% to 60%", and those are very different positions.
    early = [h for h, t in zip(hp, turns) if t["turn"] <= EARLY]
    late = [h for h, t in zip(hp, turns) if t["turn"] > LATE]
    f["hppct_early_mean"] = sum(early) / len(early)
    f["hppct_late_mean"] = sum(late) / len(late)
    f["hppct_trend"] = f["hppct_late_mean"] - f["hppct_early_mean"]

    # --- status conditions ------------------------------------------------
    # Fraction of turns spent under an ailment (poison, sleep, paralysis...).
    # 'fnt' is excluded because being fainted is already counted above and is a
    # different kind of event from being debuffed while still fighting.
    ail = sum(1 for s in states if s["status"] not in ALIVE_STATUSES and s["status"] != "fnt")
    f["status_rate"] = ail / len(states)
    f["slp_rate"] = sum(1 for s in states if s["status"] == "slp") / len(states)

    # --- loss of control --------------------------------------------------
    # Switching: counts turns where the active species changed. Voluntary
    # switches are strategic, but forced ones (after a faint) mean you are
    # reacting rather than attacking.
    f["switches"] = sum(1 for a, b in zip(names, names[1:]) if a != b)
    f["n_species_seen"] = len(set(names))

    # Turns with NO move: move_details is null on ~20% of turn-sides. This is
    # structural, not missing data - it means no move happened (switched out,
    # fainted, asleep, recharging). It is deliberately NOT imputed; the rate
    # itself is the feature, because turns where you cannot act are turns the
    # opponent acts freely.
    f["no_move_rate"] = sum(1 for m in moves if m is None) / len(moves)

    # --- what they did when they could act --------------------------------
    acted = [m for m in moves if m is not None]
    if acted:
        f["mean_base_power"] = sum(m["base_power"] for m in acted) / len(acted)
        f["mean_accuracy"] = sum(m["accuracy"] for m in acted) / len(acted)
        # Status moves have base_power 0 - they buff, debuff or inflict
        # ailments instead of damaging. A high rate means a defensive game.
        f["status_move_rate"] = sum(1 for m in acted if m["category"] == "status") / len(acted)
    else:
        f["mean_base_power"] = 0.0
        f["mean_accuracy"] = 0.0
        f["status_move_rate"] = 0.0

    return f


def battle_to_row(battle):
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}
    for side in ("p1", "p2"):
        for k, v in side_features(turns, side).items():
            row[f"{side}_{k}"] = v

    # Differences, but ONLY where both sides carry signal. Deliberately NOT
    # including a faint difference: p2's faint count is flat across outcomes
    # (0.49 when p1 loses vs 0.52 when p1 wins), so subtracting it would add
    # noise to p1's faint count, which does carry signal (1.98 vs 1.40).
    for k in ("hppct_mean", "hppct_final", "status_rate", "no_move_rate", "switches"):
        row[f"diff_{k}"] = row[f"p1_{k}"] - row[f"p2_{k}"]
    return row


def build_timeline(source):
    """
    Return one row of timeline features per battle.

    `source` is either a path to a .jsonl file, or an already-parsed list of
    battle dicts. Pass the list if you have already loaded the data.
    """
    if isinstance(source, (list, tuple)):
        battles = source
    else:
        with open(source) as f:
            battles = [json.loads(line) for line in f]
    return pd.DataFrame([battle_to_row(b) for b in battles])
