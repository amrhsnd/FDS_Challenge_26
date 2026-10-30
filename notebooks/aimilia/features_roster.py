"""
Step 6: team material - who has more Pokemon left, and how healthy are they.

THE GAP THIS FILLS
------------------
Every health feature we have so far describes the ACTIVE Pokemon - the one
standing on the field that turn. But a battle is won when all six of a side's
Pokemon have fainted, so what matters is the state of the whole TEAM.

Consider two players at turn 30, both with a healthy Pokemon on the field:
  Player A: five teammates waiting, all untouched.
  Player B: five teammates waiting, all nearly dead.
Our existing features cannot tell these apart. They are completely different
positions.

HOW WE RECOVER IT
-----------------
We keep a "roster" for each side: a running note of the last HP we saw for
every Pokemon. Each turn, we update the entry for whoever is on the field.
By turn 30 the roster holds each Pokemon's most recent known health.

Pokemon we never saw are assumed alive and at full health. For player 1 that
assumption is safe - we are told the whole team, so anyone absent simply never
came out. For player 2 we only know the lead, so an unseen Pokemon might be
one we were never shown. We make the same assumption for both sides so the
two numbers stay comparable.

Usage:
    from features_roster import build_roster
    rs_train = build_roster(train_data)
    rs_test  = build_roster(test_data)
"""

import pandas as pd

TEAM_SIZE = 6

# Points in the battle where we take a snapshot of the scoreboard, so the
# model can see the trend and not just the end state.
CHECKPOINTS = [10, 20, 30]


def track_rosters(battle):
    """
    Replay the 30 turns and record each side's team state at the checkpoints.

    Returns two dictionaries, one per side, of the form
        {turn: (number_alive, total_hp)}
    """
    # Player 1's whole team is known, so start everyone at full health.
    p1_roster = {p["name"]: 1.0 for p in battle["p1_team_details"]}

    # For player 2 we only know the lead. Others get added as they appear.
    p2_roster = {battle["p2_lead_details"]["name"]: 1.0}

    p1_snapshots = {}
    p2_snapshots = {}

    for turn in battle["battle_timeline"]:
        # Update whoever is currently on the field with their latest health.
        s1 = turn["p1_pokemon_state"]
        s2 = turn["p2_pokemon_state"]
        p1_roster[s1["name"]] = s1["hp_pct"]
        p2_roster[s2["name"]] = s2["hp_pct"]

        if turn["turn"] in CHECKPOINTS:
            p1_snapshots[turn["turn"]] = summarise(p1_roster)
            p2_snapshots[turn["turn"]] = summarise(p2_roster)

    return p1_snapshots, p2_snapshots


def summarise(roster):
    """
    Turn a roster into (how many are alive, total health of the team).

    Pokemon we have never seen count as alive and at full health, which is
    what "not yet sent out" means.
    """
    never_seen = max(0, TEAM_SIZE - len(roster))

    alive = never_seen
    total_hp = float(never_seen)
    for hp in roster.values():
        if hp > 0:
            alive += 1
        total_hp += hp

    return alive, total_hp


def battle_to_row(battle):
    row = {"battle_id": battle["battle_id"]}
    p1_snapshots, p2_snapshots = track_rosters(battle)

    for turn in CHECKPOINTS:
        p1_alive, p1_hp = p1_snapshots[turn]
        p2_alive, p2_hp = p2_snapshots[turn]

        row[f"p1_alive_t{turn}"] = p1_alive
        row[f"p2_alive_t{turn}"] = p2_alive
        row[f"alive_diff_t{turn}"] = p1_alive - p2_alive

        row[f"p1_teamhp_t{turn}"] = p1_hp
        row[f"p2_teamhp_t{turn}"] = p2_hp
        row[f"teamhp_diff_t{turn}"] = p1_hp - p2_hp

    # How fast each side is losing ground. A team that went from 6.0 to 3.0
    # total HP is collapsing; one that went 6.0 to 5.5 is holding.
    row["p1_teamhp_decline"] = row["p1_teamhp_t10"] - row["p1_teamhp_t30"]
    row["p2_teamhp_decline"] = row["p2_teamhp_t10"] - row["p2_teamhp_t30"]
    row["teamhp_decline_diff"] = row["p1_teamhp_decline"] - row["p2_teamhp_decline"]

    return row


def build_roster(battles):
    """Return one row of team-material features per battle."""
    return pd.DataFrame([battle_to_row(b) for b in battles])