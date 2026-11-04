"""
Step 9: move semantics - what the moves MEAN, not which moves they are.

WHY A SEPARATE GROUP FROM features_hazards
------------------------------------------
These columns were chosen after reading the actual move frequencies in the
training data, and they do something different from the hazards group: each
one CORRECTS a column the baseline already has but gets wrong. They are kept
separate so the ablation gives them their own verdict.

One-hot encoding all 230 move names has already been tried on this dataset
and only reached parity. These are fifteen semantic flags instead.

WHAT EACH ONE FIXES
-------------------
sleep economics   `slp_rate` treats sleep as an ailment. With 23,664 Rests
                  and 20,236 Sleep Talks in this data, RestTalk is the
                  standard Gen-2 OU set: the Pokemon slept ON PURPOSE,
                  having just healed to full, and keeps attacking through
                  Sleep Talk. `slp_acted_rate` separates that from a
                  Pokemon genuinely locked down by Hypnosis. This also
                  repairs `no_move_rate`, whose docstring claims sleep
                  implies no move.

heal bell         Cures the whole team's status. Cancels exactly the toxic
                  debt features_hazards measures, so `doomed` overstates
                  the damage owed by any side carrying it.

explosion         An intentional faint traded for enormous damage. The
                  baseline's `faints` scores it the same as losing a
                  Pokemon for nothing, which is backwards - it is usually
                  a good trade.

baton pass        Passes stat boosts to a teammate, so boosts can survive a
                  switch. The boost columns assume they die with the active
                  Pokemon.

paralysis moves   Thunder Wave and Stun Spore quarter Speed for the rest of
                  the battle. features_speed measured who was fast without
                  knowing who had been permanently crippled, and a status
                  inflicted at turn 12 is still in force at turn 80.

trapping          Mean Look and Perish Song win games outright regardless of
                  HP: the opponent cannot switch and runs out of turns.

Usage:
    from features_moves import build_moves
    mv_train = build_moves(train_data)
"""

import pandas as pd

# Grouped by what the move DOES. Names verified against the training data's
# move frequencies; unmatched names are harmless.
GROUPS = {
    "sleeptalk":   {"sleeptalk"},
    "rest":        {"rest"},
    "healbell":    {"healbell", "aromatherapy"},
    "explosion":   {"explosion", "selfdestruct"},
    "batonpass":   {"batonpass"},
    "parmove":     {"thunderwave", "stunspore", "glare"},
    "confuse":     {"confuseray", "swagger", "supersonic", "sweetkiss"},
    "trapping":    {"meanlook", "block", "spiderweb"},
    "perishsong":  {"perishsong"},
    "setup":       {"swordsdance", "curse", "growth", "agility", "amnesia",
                    "barrier", "acidarmor", "bellydrum", "irondefense",
                    "meditate", "sharpen", "doubleteam", "minimize"},
    "screens":     {"reflect", "lightscreen"},
    "thief":       {"thief"},
}


def side_moves(turns, side):
    f = {}
    states = [t[f"{side}_pokemon_state"] for t in turns]
    moves = [t[f"{side}_move_details"] for t in turns]
    n = len(states)

    counts = {group: 0 for group in GROUPS}
    for move in moves:
        if move is None:
            continue
        name = move["name"]
        for group, names in GROUPS.items():
            if name in names:
                counts[group] += 1
    for group, count in counts.items():
        f[f"n_{group}"] = count

    # --- sleep economics -------------------------------------------------
    # Of the turns spent asleep, how many did this side still act on? High
    # means RestTalk and the sleep is voluntary; low means it is crippled.
    asleep = [move is not None for state, move in zip(states, moves)
              if state["status"] == "slp"]
    f["slp_turns"] = len(asleep)
    f["slp_acted_rate"] = sum(asleep) / len(asleep) if asleep else 0.0
    # Dead turns as a share of the whole battle - the genuinely costly part
    # that `slp_rate` currently hides inside a larger number.
    f["slp_dead_rate"] = (len(asleep) - sum(asleep)) / n
    # Both halves of the archetype present.
    f["resttalk"] = 1 if counts["rest"] and counts["sleeptalk"] else 0

    # --- faints that were a trade, not a loss ----------------------------
    faints = len({s["name"] for s in states if s["status"] == "fnt"})
    f["explosion_faint_share"] = (counts["explosion"] / faints) if faints else 0.0

    # --- can this side clear the debt it is carrying? --------------------
    f["can_cure_status"] = 1 if counts["healbell"] else 0

    # --- explicit win conditions independent of HP -----------------------
    f["has_trap_wincon"] = 1 if (counts["trapping"] and counts["perishsong"]) else 0

    return f


def battle_to_row(battle):
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}
    for side in ("p1", "p2"):
        for key, value in side_moves(turns, side).items():
            row[f"{side}_{key}"] = value

    # Differences, written so POSITIVE always favours p1.
    row["diff_slp_dead_rate"] = row["p2_slp_dead_rate"] - row["p1_slp_dead_rate"]
    row["diff_n_parmove"] = row["p1_n_parmove"] - row["p2_n_parmove"]
    row["diff_n_setup"] = row["p1_n_setup"] - row["p2_n_setup"]
    row["diff_can_cure_status"] = row["p1_can_cure_status"] - row["p2_can_cure_status"]
    return row


def build_moves(battles):
    """Return one row of move-semantics features per battle."""
    return pd.DataFrame([battle_to_row(b) for b in battles])
