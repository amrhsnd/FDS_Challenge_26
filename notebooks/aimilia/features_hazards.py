"""
Step 8: the hidden turns - damage that is already owed but not yet paid.

WHY THIS GROUP IS DIFFERENT
---------------------------
Tier 1 failed because every one of those features described the PRESENT. Type
effectiveness, speed advantage and damage flow are all causes of the battle
state, and the battle state is already in the baseline as team HP and alive
counts. Measuring the cause after you have measured the effect adds nothing.

These features describe the FUTURE instead: commitments made during the
visible 30 turns whose cost falls entirely in the 50 hidden turns we are
predicting into. None of them leaves a mark on turn-30 HP.

  Spikes        Laid once, it damages every opposing Pokemon that switches
                in, for the rest of the battle. At turn 30 the opponent's HP
                may be untouched; the debt is still real.
  Toxic         Escalating poison - 1/16 of max HP, then 2/16, then 3/16.
                A badly-poisoned Pokemon at 80% health is dead within about
                ten hidden turns. `status_rate` currently averages this in
                with paralysis, which is merely annoying.
  Leech Seed    Drains every turn and heals the opponent. Same logic.
  Recovery      The counterweight: a side with Recover or Rest can service
                that debt, a side without one cannot.

THE KEY COLUMN
--------------
`doomed` - how many of a side's own Pokemon are currently ALIVE but under
escalating damage (toxic or seeded). Those are the walking dead: HP that is
counted as an asset at turn 30 and will be gone before the battle ends. If
anything in this module carries signal, it should be this.

MOVE NAMES
----------
Move names are matched against the sets below. Run
`report_move_coverage(train_data)` once to confirm they match what is
actually in the data - if the dataset spells things differently, every
move-based column here silently reads zero.

Usage:
    from features_hazards import build_hazards, report_move_coverage
    report_move_coverage(train_data)          # check first
    hz_train = build_hazards(train_data)
"""

from collections import Counter
import pandas as pd

LAST_TURN = 30
NEVER = LAST_TURN + 1     # sentinel: "this never happened"

# Gen-2 move names, Showdown style: lowercase, no spaces or punctuation.
MOVE_GROUPS = {
    "spikes":    {"spikes"},
    "toxic":     {"toxic"},
    "leechseed": {"leechseed"},
    "recovery":  {"recover", "softboiled", "milkdrink", "moonlight",
                  "synthesis", "rest", "morningsun", "slackoff"},
    "phazing":   {"roar", "whirlwind"},      # forces a switch, wiping boosts
    "haze":      {"haze"},                   # erases all stat changes
    "rapidspin": {"rapidspin"},              # the only way to clear Spikes
    "sleep":     {"spore", "hypnosis", "sing", "lovelykiss", "sleeppowder"},
}

# Statuses worth separating. `slp` is already in features_timeline; the rest
# are currently collapsed into one `status_rate` column despite having very
# different costs over a long battle.
STATUSES = ("par", "brn", "psn", "tox", "frz")

# Escalating conditions: these keep taking HP for as long as the battle runs.
DEBT_EFFECTS = {"leechseed"}
DEBT_STATUS = {"tox"}


def report_move_coverage(battles):
    """Print which tracked move names actually appear. Run this once."""
    seen = Counter()
    for battle in battles:
        for turn in battle["battle_timeline"]:
            for side in ("p1", "p2"):
                move = turn[f"{side}_move_details"]
                if move is not None:
                    seen[move["name"]] += 1

    print(f"{len(seen)} distinct move names in the data\n")
    for group, names in MOVE_GROUPS.items():
        found = {n: seen[n] for n in sorted(names) if seen[n]}
        missing = sorted(n for n in names if not seen[n])
        print(f"{group:10s} matched {found or '{}'}")
        if missing:
            print(f"{'':10s} NOT FOUND: {missing}")
    tracked = {n for names in MOVE_GROUPS.values() for n in names}
    print("\nMost common moves NOT in any group (check for near-misses):")
    for name, count in seen.most_common():
        if name not in tracked:
            print(f"  {name}: {count:,}")
            if count < 2000:
                break
    return seen


def side_hazards(turns, side, other):
    """Summarise one side's future-facing commitments."""
    f = {}
    states = [t[f"{side}_pokemon_state"] for t in turns]
    moves = [t[f"{side}_move_details"] for t in turns]
    n = len(states)

    # --- move usage ------------------------------------------------------
    used = {group: 0 for group in MOVE_GROUPS}
    first_spikes = NEVER
    for turn, move in zip(turns, moves):
        if move is None:
            continue
        name = move["name"]
        for group, names in MOVE_GROUPS.items():
            if name in names:
                used[group] += 1
                if group == "spikes" and turn["turn"] < first_spikes:
                    first_spikes = turn["turn"]
    for group, count in used.items():
        f[f"n_{group}"] = count

    # --- Spikes pressure -------------------------------------------------
    # Spikes laid by THIS side damages the OTHER side's switch-ins. Count how
    # many times the opponent switched after we laid them: that is damage
    # already dealt but invisible, and it keeps accruing after turn 30.
    f["spikes_up"] = 1 if first_spikes <= LAST_TURN else 0
    f["spikes_turn"] = first_spikes
    opp_names = [t[f"{other}_pokemon_state"]["name"] for t in turns]
    switchins = 0
    for index in range(1, len(turns)):
        if turns[index]["turn"] > first_spikes and opp_names[index] != opp_names[index - 1]:
            switchins += 1
    f["opp_switchins_after_spikes"] = switchins
    # Turns remaining with Spikes up, as a crude measure of how long the
    # opponent has to live with them. Earlier is worth much more than later.
    f["spikes_turns_up"] = max(0, LAST_TURN - first_spikes) if f["spikes_up"] else 0

    # --- status, split by severity ---------------------------------------
    for status in STATUSES:
        f[f"{status}_rate"] = sum(1 for s in states if s["status"] == status) / n

    # --- the debt --------------------------------------------------------
    # Walk the roster: for each species, the last state we saw it in. A
    # species still alive at its last sighting while toxic or seeded is HP
    # that turn-30 features count as an asset and that will not survive.
    last_state = {}
    toxed_species = set()
    for state in states:
        last_state[state["name"]] = state
        if state["status"] == "tox":
            toxed_species.add(state["name"])

    doomed = 0
    for state in last_state.values():
        if state["hp_pct"] <= 0 or state["status"] == "fnt":
            continue
        in_debt = (state["status"] in DEBT_STATUS
                   or any(e in DEBT_EFFECTS for e in state["effects"]))
        if in_debt:
            doomed += 1
    f["doomed"] = doomed

    f["n_species_toxed"] = len(toxed_species)
    final = states[-1]
    f["tox_at_end"] = 1 if final["status"] == "tox" else 0
    # Can this side service the debt? Recovery moves per turn acted.
    acted = sum(1 for m in moves if m is not None)
    f["recovery_rate"] = used["recovery"] / acted if acted else 0.0

    return f


def battle_to_row(battle):
    turns = battle["battle_timeline"]
    row = {"battle_id": battle["battle_id"]}
    for side, other in (("p1", "p2"), ("p2", "p1")):
        for key, value in side_hazards(turns, side, other).items():
            row[f"{side}_{key}"] = value

    # Differences, written so that POSITIVE always favours p1.
    row["diff_spikes_up"] = row["p1_spikes_up"] - row["p2_spikes_up"]
    row["diff_doomed"] = row["p2_doomed"] - row["p1_doomed"]
    row["diff_tox_rate"] = row["p2_tox_rate"] - row["p1_tox_rate"]
    row["diff_recovery_rate"] = row["p1_recovery_rate"] - row["p2_recovery_rate"]
    return row


def build_hazards(battles):
    """Return one row of future-facing features per battle."""
    return pd.DataFrame([battle_to_row(b) for b in battles])
