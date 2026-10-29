"""
Step 1: static (pre-battle) features for the Pokemon battle prediction challenge.

Turns the nested JSON into one flat row per battle, using ONLY what is known
before the first turn: p1's six-Pokemon team and p2's lead Pokemon.
The battle_timeline is deliberately ignored here - that is a later step.

Usage:
    from features_static import build_static
    train = build_static("train.jsonl", labelled=True)
    test  = build_static("test.jsonl",  labelled=False)
"""

import json
import pandas as pd

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]

# The 17 real Gen-2 types. "notype" is a placeholder for "this Pokemon has
# only one type", NOT a real type, so it is excluded from the vocabulary.
NOT_A_TYPE = "notype"


def real_types(poke):
    """The Pokemon's actual types, with the 'notype' placeholder stripped out."""
    return [t for t in poke["types"] if t != NOT_A_TYPE]


def effective_stats(poke):
    """
    Base stats scaled by level.

    Base stats are the species' innate values and do NOT include level. 99.3%
    of Pokemon in this data are level 100, but 422 rows are not (abra, pineco,
    elekid, ...). Left unscaled, a level-5 abra looks exactly as strong as a
    level-100 abra, which is wrong.

    level/100 is a crude linear proxy for the real Gen-2 stat formula, which is
    roughly proportional to level. It is an approximation, but it is much closer
    to the truth than ignoring level entirely, and it costs nothing.
    """
    scale = poke["level"] / 100.0
    return {s: poke[f"base_{s}"] * scale for s in STATS}


def battle_to_row(battle, labelled):
    row = {"battle_id": battle["battle_id"]}

    team = battle["p1_team_details"]
    lead = battle["p2_lead_details"]

    # --- p1: six Pokemon collapsed into summary numbers -------------------
    # A team is a SET of six Pokemon with no meaningful order, so we must
    # aggregate. If we instead made columns like p1_poke1_atk ... p1_poke6_atk,
    # the model would learn position-specific patterns that are pure noise.
    team_stats = [effective_stats(p) for p in team]
    for s in STATS:
        vals = [ts[s] for ts in team_stats]
        row[f"p1_{s}_mean"] = sum(vals) / len(vals)   # overall team quality
        row[f"p1_{s}_max"] = max(vals)                # best specialist
    row["p1_level_mean"] = sum(p["level"] for p in team) / len(team)
    row["p1_level_min"] = min(p["level"] for p in team)

    # Type coverage: how many distinct real types the team spans. More coverage
    # means more answers to whatever p2 brings.
    row["p1_n_types"] = len({t for p in team for t in real_types(p)})
    # Dual-typed Pokemon get more resistances but also more weaknesses.
    row["p1_n_dual_type"] = sum(1 for p in team if len(real_types(p)) == 2)
    # 95.5% of items are 'leftovers'. A non-leftovers item is therefore a
    # deliberate, unusual choice and worth counting.
    row["p1_n_special_item"] = sum(1 for p in team if p["item"] != "leftovers")

    # --- p2: only the lead is known --------------------------------------
    # This is the asymmetry in the data. We cannot aggregate a team we have not
    # been shown, so for now we use the lead's raw numbers. Recovering the rest
    # of p2's team from the timeline is a later step.
    lead_stats = effective_stats(lead)
    for s in STATS:
        row[f"p2_lead_{s}"] = lead_stats[s]
    row["p2_lead_level"] = lead["level"]
    row["p2_lead_n_types"] = len(real_types(lead))

    # --- comparisons ------------------------------------------------------
    # Trees can in principle derive these, but handing a model the difference
    # directly makes the signal far easier to find than asking it to discover
    # "subtract column A from column B" by itself.
    for s in STATS:
        row[f"diff_{s}_mean"] = row[f"p1_{s}_mean"] - row[f"p2_lead_{s}"]
        row[f"diff_{s}_max"] = row[f"p1_{s}_max"] - row[f"p2_lead_{s}"]

    if labelled:
        row["player_won"] = battle["player_won"]
    return row


def build_static(source, labelled=True):
    """
    Return one flat DataFrame row per battle.

    `source` is either a path to a .jsonl file, or an already-parsed list of
    battle dicts. Pass the list if you have already loaded the data - parsing
    164 MB of JSON once is enough.
    """
    if isinstance(source, (list, tuple)):
        battles = source
    else:
        with open(source) as f:
            battles = [json.loads(line) for line in f]
    return pd.DataFrame([battle_to_row(b, labelled) for b in battles])


if __name__ == "__main__":
    import sys
    df = build_static(sys.argv[1], labelled="train" in sys.argv[1])
    print(df.shape)
    print(df.head())
