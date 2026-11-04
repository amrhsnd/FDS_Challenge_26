"""
Step 7c: type effectiveness of the moves that were actually used.

THE GAP THIS FILLS
------------------
`move_details` has a `type` field and we have never read it. We use
base_power, accuracy and category, which means a 100-power Earthquake is
treated the same whether it hit a Rock Pokemon (2x damage) or a Flying one
(0x - the move does literally nothing). That is the central mechanic of the
game and it is currently invisible to the model.

The type chart is `data/type_chart_gen2.csv`, supplied by the organisers and
explicitly allowed by the rules. Defending types come from the species
lookup in pokedex.py, which is built from training rosters only.

WHAT IS MEASURED
----------------
For each side, over the turns where it used a DAMAGING move: the average
multiplier it achieved, and how often it hit for 2x or more, was resisted,
or was walled completely. A side that keeps hitting into resistances is
wasting turns even when its own health looks fine.

STAB is counted too: "same type attack bonus", the 1.5x a Pokemon gets for
using a move matching one of its own types. A high STAB rate means a side is
playing to its strengths rather than reaching for filler coverage moves.

Status moves are skipped. They have base_power 0, so a damage multiplier is
not meaningful for them; features_effects already covers what they do.

Usage:
    from features_matchup import build_matchup
    mu_train = build_matchup(train_data, CHART_PATH, DEX)
"""

import csv
import math
import pandas as pd

from pokedex import battle_lookup

LATE = 20


def read_chart(path):
    """
    Read the supplied Gen-2 type chart into {(attacker, defender): multiplier}.

    Handles both shapes a chart CSV comes in: a long form with
    attacking_type / defending_type / multiplier columns, and a matrix with
    attackers down the first column and defenders across the header. Pairs
    absent from the file are neutral (1.0) - charts normally list only the
    non-neutral entries.
    """
    with open(path, newline="") as stream:
        rows = [r for r in csv.reader(stream) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")

    header = [c.strip().lower() for c in rows[0]]
    chart = {}

    if "attacking_type" in header and "defending_type" in header:
        ai, di = header.index("attacking_type"), header.index("defending_type")
        mi = next((i for i, h in enumerate(header)
                   if "mult" in h or "effect" in h or "value" in h), None)
        if mi is None:
            raise ValueError(f"No multiplier column found in {path}: {header}")
        for row in rows[1:]:
            chart[(row[ai].strip().lower(), row[di].strip().lower())] = float(row[mi])
    else:
        defenders = header[1:]
        for row in rows[1:]:
            attacker = row[0].strip().lower()
            for defender, value in zip(defenders, row[1:]):
                if value.strip() == "":
                    continue
                chart[(attacker, defender)] = float(value)

    if not chart:
        raise ValueError(f"Could not parse a type chart from {path}")
    return chart


def effectiveness(move_type, defender_types, chart):
    """
    Damage multiplier of one attack type against one Pokemon.

    Dual types multiply: Rock/Ground takes 4x from Water. Returns None when
    we do not know enough to say, so the caller can skip the turn rather
    than guess 1.0.
    """
    if not move_type or not defender_types:
        return None
    return math.prod(chart.get((move_type, d), 1.0) for d in defender_types)


def side_matchup(turns, side, other, resolve, chart):
    """Summarise the type matchups one side achieved with its damaging moves."""
    values = []
    late_values = []
    stab_flags = []

    for turn in turns:
        move = turn[f"{side}_move_details"]
        if move is None or move.get("category") == "status":
            continue
        defender = resolve(turn[f"{other}_pokemon_state"]["name"])
        if defender is None:
            continue
        value = effectiveness(move.get("type"), defender[0], chart)
        if value is None:
            continue
        values.append(value)
        if turn["turn"] > LATE:
            late_values.append(value)
        attacker = resolve(turn[f"{side}_pokemon_state"]["name"])
        if attacker is not None:
            stab_flags.append(1.0 if move.get("type") in attacker[0] else 0.0)

    n = len(values)
    if n == 0:
        # No resolvable damaging move. NaN rather than 0, so the imputer in
        # the notebook pipeline fills it with the median instead of us
        # claiming this side never got a favourable matchup.
        nan = float("nan")
        return {"eff_mean": nan, "se_rate": nan, "resist_rate": nan,
                "immune_rate": nan, "eff_late_mean": nan, "stab_rate": nan,
                "n_damaging": 0}

    return {
        "eff_mean": sum(values) / n,
        # Hit for double damage or better.
        "se_rate": sum(1 for v in values if v >= 2) / n,
        # Hit into a resistance - half damage or less, but not nothing.
        "resist_rate": sum(1 for v in values if 0 < v <= 0.5) / n,
        # Completely walled. A wasted turn, and the worst outcome here.
        "immune_rate": sum(1 for v in values if v == 0) / n,
        "eff_late_mean": (sum(late_values) / len(late_values)
                          if late_values else float("nan")),
        "stab_rate": (sum(stab_flags) / len(stab_flags)
                      if stab_flags else float("nan")),
        # How many turns this side spent attacking at all - the denominator,
        # and informative in its own right.
        "n_damaging": n,
    }


def battle_to_row(battle, chart, dex):
    resolve = battle_lookup(battle, dex)
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}

    for side, other in (("p1", "p2"), ("p2", "p1")):
        for key, value in side_matchup(turns, side, other, resolve, chart).items():
            row[f"{side}_{key}"] = value

    for key in ("eff_mean", "se_rate", "immune_rate", "stab_rate"):
        row[f"diff_{key}"] = row[f"p1_{key}"] - row[f"p2_{key}"]

    return row


def build_matchup(battles, chart_path, dex):
    """Return one row of type-matchup features per battle."""
    chart = read_chart(chart_path)
    return pd.DataFrame([battle_to_row(b, chart, dex) for b in battles])
