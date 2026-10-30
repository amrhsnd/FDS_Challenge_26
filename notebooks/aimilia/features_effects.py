"""
Step 5: effects and stat-boost features.

These are the last two pieces of the battle timeline we have never read.

WHAT ARE "EFFECTS"?
-------------------
Ongoing conditions on the Pokemon currently fighting - things a move put
there that persist for several turns. The ones that matter here:

  substitute  - the player spent some of their own HP to create a decoy that
                absorbs hits. A deliberate, skilful, defensive play.
  trapped     - the Pokemon cannot switch out. It is stuck in a fight it may
                be losing, which is a bad place to be.
  confusion   - it sometimes hits itself instead of attacking.
  leechseed   - it loses health every turn, and the opponent gains it.
  protect     - it blocked attacks for one turn.
  attract     - it sometimes refuses to attack.
  perish      - a countdown: it faints in three turns unless it switches out.

The placeholder "noeffect" means nothing is active, so we ignore it.

WHAT ARE "BOOSTS"?
------------------
Temporary increases or decreases to a Pokemon's stats during battle, caused
by moves like Swords Dance (raises attack). They are recorded as "stages",
usually between -6 and +6, where 0 means unchanged. A Pokemon that has spent
turns boosting itself is set up to do a lot of damage.

WHY THESE MIGHT MATTER
----------------------
Everything we have measured so far describes the POSITION a player is in -
how much health, how many fainted. Effects and boosts describe the STRATEGY
they are playing. Winners use substitute 65% more often than losers, which
is not something our existing features can see.

Usage:
    from features_effects import build_effects
    ef_train = build_effects(train_data)
    ef_test  = build_effects(test_data)
"""

import pandas as pd

# The effects common enough to be worth a column of their own. Rarer ones
# (bide, mindreader, spite...) appear a handful of times in 300,000 turns
# and would be mostly zeros.
TRACKED_EFFECTS = [
    "substitute",
    "trapped",
    "confusion",
    "leechseed",
    "protect",
    "perish",
    "attract",
    "encore",
    "destinybond",
    "clamp",
]

# The five stats that can be boosted during a battle.
BOOST_STATS = ["atk", "def", "spa", "spd", "spe"]


def side_features(turns, side):
    """Summarise one player's effects and boosts across the 30 turns."""
    features = {}
    states = [turn[f"{side}_pokemon_state"] for turn in turns]
    n_turns = len(states)

    # --- effects -------------------------------------------------------
    # For each tracked effect, what fraction of the 30 turns was it active?
    for effect in TRACKED_EFFECTS:
        count = 0
        for state in states:
            if effect in state["effects"]:
                count += 1
        features[f"eff_{effect}"] = count / n_turns

    # How often ANY effect was active, as a general measure of how eventful
    # this player's side of the battle was.
    any_effect = 0
    for state in states:
        real_effects = [e for e in state["effects"] if e != "noeffect"]
        if real_effects:
            any_effect += 1
    features["eff_any"] = any_effect / n_turns

    # --- boosts --------------------------------------------------------
    # Each turn, add up the five stat stages. Positive means this Pokemon
    # has been buffed overall; negative means it has been weakened.
    totals = []
    offensive = []
    defensive = []
    speed = []

    for state in states:
        boosts = state["boosts"]
        totals.append(sum(boosts[s] for s in BOOST_STATS))
        offensive.append(boosts["atk"] + boosts["spa"])
        defensive.append(boosts["def"] + boosts["spd"])
        speed.append(boosts["spe"])

    features["boost_mean"] = sum(totals) / n_turns
    features["boost_max"] = max(totals)
    features["boost_final"] = totals[-1]
    features["boost_offensive_mean"] = sum(offensive) / n_turns
    features["boost_defensive_mean"] = sum(defensive) / n_turns
    features["boost_speed_mean"] = sum(speed) / n_turns

    return features


def battle_to_row(battle):
    row = {"battle_id": battle["battle_id"]}
    turns = battle["battle_timeline"]

    for side in ("p1", "p2"):
        for name, value in side_features(turns, side).items():
            row[f"{side}_{name}"] = value

    # Differences for the handful with the clearest signal. We keep this
    # short on purpose: adding a difference for every column has repeatedly
    # hurt us, because subtracting a noisy column from a useful one just
    # spreads the noise.
    for name in ("eff_substitute", "eff_trapped", "boost_mean"):
        row[f"diff_{name}"] = row[f"p1_{name}"] - row[f"p2_{name}"]

    return row


def build_effects(battles):
    """Return one row of effect and boost features per battle."""
    return pd.DataFrame([battle_to_row(b) for b in battles])