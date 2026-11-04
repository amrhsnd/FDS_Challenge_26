# Features

274 features per battle, from 11 modules. The reference set 106 describe the position
each side is in at turn 30. The 168 candidate group describe how it got there and
what is owed in the hidden turns after turn 30.

player 1's six Pokemon are fully described, player 2 shows only its
starting Pokemon, and we see the first 30 turns of a battle that runs to somewhere
between 30 and 80. We are never told how long the battle lasts, and the target is
the final result - not who is ahead at turn 30.

---

## Part 1 - reference set (106 features)

### from build_static - before the fight starts (37)

* **p1** (17): six Pokemon 
    - 2 summaries over stat: mean and max (6*2=12)
    - 2 levels (mean, min)
    - shape (types: how many elements the team covers, dual_type: how many creatures have two types instead of one, special_item: how many carry something other than the defaut item) -3

* **p2** (8): only the opening Pokemon is known, so no averaging is possible
  - its 6 base stats (6), 
  - its level, how many types it has (2)

* **comparisons** (12): p1's mean and max minus p2's opener, for each stat (6*2)

### from build_timeline - what happened in the 30 turns (35)

each of the 30 turns stores 4 things per side: active creatures, HP percentage of the active creature, its status condition, and which move it used. Each of the following appears twice,
once per side:

  - **faints**: how many distinct Pokemon died (2)
  - **health**: mean, final, minimum health of the active Pokemon (6)
  - **health over time**: mean in turns 1-10, mean in turns 21-30, and the change
    between them (6)
  - **status**: share of turns under any ailment, share asleep (4)
  - **loss of control**: how often they swapped Pokemon, how many different ones
    appeared, share of turns where no move happened at all (6)
  - **move quality**: mean power, mean accuracy, share of non-damaging moves (6)
  - **differences**: 5 selected comparisons between the sides (5)

Health here is the *active* Pokemon's, not the team's - it jumps around as Pokemon
switch in and out.

### from build_effects - stat boosts (13)

Boosts are temporary stat increases from moves like Swords Dance, recorded as
"stages" from -6 to +6. 
we get each of the following twice (one per team):
- overall boost level: boost_mean (average total stages per turn),
  boost_max (highest total reached), boost_final (total at turn 30) (6)
- boosts by kind: boost_offensive_mean (atk+spa), boost_defensive_mean
  (def+spd), boost_speed_mean (spe) (6)
- difference: diff_boost_mean (1)

### from features_roster - team material (21)

Fixes a real blind spot: every health column above describes the one Pokemon on
the field. Two sides can both have a healthy Pokemon out while one has five
untouched teammates waiting and the other has five nearly dead. We replay the 30
turns keeping a running note of the last health we saw for every Pokemon, then
snapshot at turns 10, 20 and 30:

  - how many are alive per side, and the difference (3 per checkpoint)
  - total health of the whole team per side, and the difference (3 per checkpoint)
  - plus how fast each side is bleeding: team health at turn 10 minus turn 30 (3)

Pokemon never seen are assumed alive at full health - for p1 that is safe, and we
apply it to p2 as well so the two numbers stay comparable.

---

## Part 2 - candidate group (168 features)

### from build_effects - the effect columns (24)

These were being computed and then silently discarded; the merge kept only columns
with "boost" in the name. "Effects" are ongoing conditions a move left behind:
`substitute` (spent own health to make a decoy that absorbs hits), `trapped`
(cannot switch out), `leechseed` (drains health every turn and gives it to the
opponent), `confusion`, `protect`, `perish`, `attract`, `encore`, `destinybond`,
`clamp`. Per side: the share of the 30 turns each was active (10), plus a share
for *any* effect being active (1). Two differences.

### from features_flow - damage actually dealt and healed (19)

**The gap**: every health column so far is a *level*. None is a *flow*. That
matters because a Pokemon that dies is replaced by a fresh one at full health,
which resets the visible number. A side that knocked out three opponents and a
side that never landed a hit can look identical at turn 30.

We walk the turns tracking each Pokemon's last known health. Its change since we
last saw it is either damage taken (down) or healing (up). Per side (7 each):
total damage taken, total healed, the net, damage taken in turns 1-10, damage
taken in turns 21-30, the single worst hit, and how many turns had a drop over 30%.
Plus 5 comparisons, including a *share* (0.5 means a dead-even trade regardless of
how violent the battle was) and an integral that rewards being ahead for longer
rather than just by more at the end.

### from features_matchup - type effectiveness of the moves used (18)

**The gap**: each move has a `type` field and we were never reading it. Types have
advantages over each other: a Water attack does double damage to a Fire Pokemon,
and some combinations do *nothing at all*. So an identical move can be devastating
or completely wasted depending on what it hits, and the reference set could not tell
the difference.

Defending types come from a species lookup built from the training rosters;
multipliers come from the type chart the organisers supplied. Per side, over the
turns it attacked (7 each): average multiplier achieved, share of attacks hitting
for 2x or more, share hitting a resistance, share hitting a total immunity (a
wasted turn), average multiplier in the last 10 turns, share of attacks matching
the attacker's own type (which gets a 1.5x bonus), and the number of attacking
turns. Plus 4 differences.

### from features_speed - who moves first (8)

**The gap**: speed decides turn order, and turn order decides who lands the
killing blow first. We only had the team's *average* speed and the opening
opponent's - nothing about whether the Pokemon on the field was faster than the
one facing it, turn after turn. Three things change effective speed and all three
are in the data: stat-stage boosts, paralysis (quarters speed in this generation),
and level.

Average effective speed of the active Pokemon per side (2), the average gap, the
share of turns p1 was faster, that same share over the last 10 turns, the share of
exact ties (resolved by coin flip, so a high rate means an unpredictable battle),
and each side's rate of using priority moves - which go first regardless of speed.

### from features_damage - estimated damage per turn (11)

The densest feature here: it folds base stats, level, types, stat boosts, burn,
move power and move accuracy into **one number per turn** - roughly how hard that
attack should have hit.

`power x accuracy x type effectiveness x same-type bonus x (attack / defence)`

Accuracy is included because expected damage matters more than best case. It omits
the real formula's random spread and critical hits, and it does not need to be
right in absolute terms - only to order turns correctly by how much damage they
probably did. Per side (4 each): total, mean, max, and mean over the last 10 turns.
Plus total difference, late difference, and a scale-free share.

### from features_hazards - damage already owed but not yet paid (46)

**The gap, and the reasoning that produced this group**: the four groups above all
describe the *present*, and the reference set already measures the present as team health
and alive counts. Measuring the cause after you have already measured its effect
adds nothing - which is what the ablation found. These columns describe the *future*
instead: commitments made in the visible 30 turns whose cost falls entirely in the
50 hidden turns we predict into. None leaves a mark on turn-30 health.

  - **Spikes** - laid once, it damages every opposing Pokemon that switches in, for
    the rest of the battle. At turn 30 the opponent may look untouched; the debt is
    real. Per side: whether it was laid, on which turn, how many turns it has been
    up, and how many times the opponent has switched since (4 each).
  - **Status, split by severity** - the reference set has one `status_rate`, which averages
    together conditions with wildly different costs over a long battle. Separate rates
    for paralysis, burn, poison, bad poison and freeze (5 each). Bad poison escalates:
    a Pokemon at 80% health is dead within about ten hidden turns.
  - **`doomed`** - the key column. How many of a side's own Pokemon are currently
    *alive* but under escalating damage. Those are health the turn-30 features count
    as an asset and that will be gone before the battle ends.
  - **Move counts** (8 each) and **recovery rate** - whether a side has the healing
    moves to service that debt.
  - Plus `n_species_toxed`, `tox_at_end`, and 4 differences.

### from features_moves - what the moves mean (42)

Chosen after reading the actual move frequencies in the training data. Each column
here **corrects a baseline column that the data shows is wrong**.

  - **Sleep economics** - `rest` appears 23,664 times and `sleeptalk` 20,236. Together
    they are the standard set in this format: heal to full and fall asleep on purpose,
    then keep attacking while asleep. So the reference's `slp_rate` was measuring "runs
    the normal set" and "is crippled" as the same number. `slp_acted_rate` (of the turns
    spent asleep, how many did this side still act on), `slp_dead_rate`, `slp_turns` and
    a `resttalk` flag separate the two.
  - **`explosion`** (5,483 uses) - an intentional death traded for enormous damage.
    The reference's `faints` scores it the same as losing a Pokemon for nothing, which
    is backwards.
  - **`healbell`** (2,229) - cures the whole team's status, cancelling exactly the debt
    the hazards group measures. So `doomed` overstates the damage owed by a side
    carrying it.
  - **`batonpass`** (3,280) - passes stat boosts to a teammate, so boosts can survive a
    switch. The boost columns assume they die with the active Pokemon.
  - **Paralysis moves** (8,552 combined) - quarter the opponent's speed for the rest of
    the battle, which is why measuring speed without them was incomplete.
  - **Trapping plus perish song** - wins outright regardless of health: the opponent
    cannot switch and runs out of turns.
  - 12 move-group counts and 7 derived columns per side, plus 4 differences.

Note this is deliberately **not** a one-hot encoding of all 230 move names - that was
tried on this dataset separately and only reached parity. These are fifteen semantic
flags instead.

---

## GRAND TOTAL: 274 features (the starter notebook had 6)




