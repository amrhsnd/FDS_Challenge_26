"""
Step 7a: a species lookup, learned from the competition data only.

WHY THIS EXISTS
---------------
Three of the four new feature families need to know a Pokemon's TYPES and
BASE STATS. For player 1 that is handed to us in `p1_team_details`. For
player 2 we are given only the lead - but the timeline names p2's other
Pokemon as they appear, and those same species appear on p1's teams
somewhere across the 10,000 training battles.

So we can build a species -> (types, base stats) table from the training
rosters and leads alone. That is competition data, not an external
database, so it is allowed under the challenge rules. Do NOT fill gaps from
Pokemon Showdown or Bulbapedia; that is explicitly prohibited.

LEAKAGE
-------
`build_pokedex` is called on TRAIN battles only and never sees a label or a
test row. At transform time each battle's own metadata is overlaid on top,
which is given for train and test alike, so a test battle containing a
species the training set never showed still resolves if it is a p1 member
or the p2 lead.

LEVELS
------
Levels are battle-specific, not a species property, so they are not stored.
Where a battle tells us the level (p1's six, p2's lead) we use it. For p2's
other Pokemon, revealed only by name in the timeline, we assume level 100 -
true for 99.3% of this dataset.

Usage:
    from pokedex import build_pokedex, battle_lookup
    DEX = build_pokedex(train_data)
    resolve = battle_lookup(battle, DEX)
    types, stats = resolve("tyranitar")     # stats are level-scaled
"""

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NOT_A_TYPE = "notype"
DEFAULT_LEVEL = 100


def real_types(poke):
    """The Pokemon's actual types, with the 'notype' placeholder stripped."""
    return tuple(t for t in poke.get("types", ()) if t != NOT_A_TYPE)


def build_pokedex(battles):
    """
    species name -> {"types": (...), "base": {hp:.., atk:.., ...}}

    Call this on the TRAINING battles once, then reuse the result.
    """
    dex = {}
    for battle in battles:
        known = list(battle.get("p1_team_details", []))
        lead = battle.get("p2_lead_details")
        if lead is not None:
            known.append(lead)
        for poke in known:
            name = poke["name"]
            if name in dex:
                continue
            dex[name] = {
                "types": real_types(poke),
                "base": {s: poke[f"base_{s}"] for s in STATS},
            }
    return dex


def battle_lookup(battle, dex):
    """
    Return a resolve(name) function for this battle.

    resolve returns (types_tuple, level_scaled_stats_dict), or None if the
    species cannot be resolved at all. The battle's own metadata wins over
    the shared dex, so a species absent from the dex still works when this
    battle happens to describe it.
    """
    local = {}
    known = list(battle.get("p1_team_details", []))
    lead = battle.get("p2_lead_details")
    if lead is not None:
        known.append(lead)
    for poke in known:
        local[poke["name"]] = (
            real_types(poke),
            poke.get("level", DEFAULT_LEVEL),
            {s: poke[f"base_{s}"] for s in STATS},
        )

    cache = {}

    def resolve(name):
        if name in cache:
            return cache[name]
        if name in local:
            types, level, base = local[name]
        else:
            entry = dex.get(name)
            if entry is None:
                cache[name] = None
                return None
            types, level, base = entry["types"], DEFAULT_LEVEL, entry["base"]
        scale = level / 100.0
        result = (types, {s: base[s] * scale for s in STATS})
        cache[name] = result
        return result

    return resolve
