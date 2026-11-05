"""Numeric battle summaries from supplied rosters and the observed timeline.

These functions compute feature values only: they do not read labels, fit models,
select predictors or write files. HP totals use the latest observed state and
presume unobserved roster slots are full HP. Damage and healing summarize net HP
changes; they do not attribute a change to a particular move.

The registry combines roster, status, active-position, move, type-matchup and
trajectory summaries. Its raw keys are stable identifiers used by the feature
catalog; public model columns receive descriptive names from that catalog.
"""
from collections import Counter
from functools import partial

import numpy as np

from pokemon_challenge.features.first_knockout import first_ko_advantage
from pokemon_challenge.features.retained_position import (
    NEW_FEATURE_COLUMNS, retained_position_features,
)

TEAM_SIZE = 6
MAX_OBSERVED_TURNS = 30

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
TYPES = (
    'bug', 'dark', 'dragon', 'electric', 'fighting', 'fire', 'flying',
    'ghost', 'grass', 'ground', 'ice', 'normal', 'poison', 'psychic',
    'rock', 'steel', 'water',
)
BOOST_STATS = ('atk', 'def', 'spa', 'spd', 'spe')
P2_ROSTER_STAT_COLUMNS = ('p2_mean_hp', 'p2_mean_atk', 'p2_mean_def', 'p2_mean_spe')
# Fixed core schema for roster, HP, moves, boosts and retained resources.
BASE_FEATURE_COLUMNS = (
    'p2_team_hp_pct_at_turn30',
    'p1_dmg',
    'p1_team_hp_pct_at_turn30',
    'p2_dmg',
    'p2_heal',
    'p1_heal',
    'p1_boost_sum_at_turn30',
    'p1_no_damage_moves',
    'p1_mean_hp',
    'alive_diff_at_turn30',
    'p1_alive_at_turn30',
    'p2_no_damage_moves',
    'p2_boost_sum_at_turn30',
    'p2_mean_boosts',
    'p1_mean_boosts',
    'p1_mean_spe',
    'p1_switch',
    'p2_no_of_moves',
    'p1_special_moves',
    'p1_mean_atk',
    *P2_ROSTER_STAT_COLUMNS,
    'p1_mean_def',
    'p1_damage_surplus',
    'p2_damage_surplus',
    'first_ko_advantage',
)
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + NEW_FEATURE_COLUMNS
REQUESTED_FEATURE_SUFFIXES = (
    "positive_boosts_at_turn30", "boost_sum_at_turn30", "no_damage_moves",
    "fainting_turns", "team_hp_pct_at_turn30", "special_moves",
)
assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 32


def extract_requested_features(battle):
    """Summarize final states and observed move outcomes through turn 30."""
    turns = [turn for turn in (battle.get('battle_timeline') or [])
             if 1 <= turn.get('turn', 0) <= MAX_OBSERVED_TURNS]
    final_turn = next((turn for turn in reversed(turns)
                       if turn['turn'] == MAX_OBSERVED_TURNS), {})
    latest_hp = {
        'p1': {pokemon['name']: 1.0 for pokemon in (battle.get('p1_team_details') or [])},
        'p2': {},
    }
    lead_name = (battle.get('p2_lead_details') or {}).get('name')
    if lead_name:
        latest_hp['p2'][lead_name] = 1.0
    previous_state = {'p1': None, 'p2': None}
    fainted = {'p1': set(), 'p2': set()}
    features = {}
    for side in ('p1', 'p2'):
        for suffix in ('no_damage_moves', 'max_observed_move_damage',
                       'fainting_turns', 'special_moves'):
            features[f'{side}_{suffix}'] = 0.0

    for turn in turns:
        for side in ('p1', 'p2'):
            opponent = 'p2' if side == 'p1' else 'p1'
            state = turn.get(f'{side}_pokemon_state') or {}
            name, hp = state.get('name'), state.get('hp_pct')
            is_fainted = hp == 0 or state.get('status') == 'fnt'
            if name and is_fainted and name not in fainted[side]:
                features[f'{side}_fainting_turns'] += 1
                fainted[side].add(name)
            if name and hp is not None:
                latest_hp[side][name] = 0.0 if is_fainted else float(hp)

            move = turn.get(f'{side}_move_details')
            if move is not None:
                features[f'{side}_special_moves'] += move.get('category') == 'special'
                target = turn.get(f'{opponent}_pokemon_state') or {}
                previous = previous_state[opponent]
                # Only adjacent observations of the same target are comparable.
                if (previous and target.get('name')
                        and previous.get('name') == target['name']
                        and previous.get('hp_pct') is not None
                        and target.get('hp_pct') is not None):
                    drop = max(float(previous['hp_pct']) - float(target['hp_pct']), 0.0)
                    features[f'{side}_no_damage_moves'] += drop == 0.0
                    features[f'{side}_max_observed_move_damage'] = max(
                        features[f'{side}_max_observed_move_damage'], drop)
        # Update both sides together so results do not depend on loop order.
        previous_state = {
            side: turn.get(f'{side}_pokemon_state') or {}
            for side in ('p1', 'p2')
        }

    for side in ('p1', 'p2'):
        state = final_turn.get(f'{side}_pokemon_state') or {}
        boosts = state.get('boosts') or {}
        stages = [float(boosts.get(stat, 0)) for stat in BOOST_STATS]
        features[f'{side}_positive_boosts_at_turn30'] = sum(max(value, 0.0) for value in stages)
        features[f'{side}_negative_boosts_at_turn30'] = sum(min(value, 0.0) for value in stages)
        features[f'{side}_boost_sum_at_turn30'] = sum(stages)
        unseen = max(0, TEAM_SIZE - len(latest_hp[side]))
        features[f'{side}_team_hp_pct_at_turn30'] = (
            100.0 * (sum(latest_hp[side].values()) + unseen) / TEAM_SIZE)
    return features


def damage_surplus_curves(battle):
    """End-of-turn cumulative net damage on a fixed, equally weighted 30-turn grid."""
    by_turn = {
        entry.get('turn', index): entry
        for index, entry in enumerate(
            (battle.get('battle_timeline') or [])[:MAX_OBSERVED_TURNS], start=1)
        if 1 <= entry.get('turn', index) <= MAX_OBSERVED_TURNS
    }
    previous_hp = {'p1': {}, 'p2': {}}
    net_received = {'p1': 0.0, 'p2': 0.0}
    curves = {'p1_net_damage_dealt': [], 'p2_net_damage_dealt': []}
    for turn_number in range(1, MAX_OBSERVED_TURNS + 1):
        entry = by_turn.get(turn_number, {})
        for side in ('p1', 'p2'):
            state = entry.get(f'{side}_pokemon_state') or {}
            name, hp = state.get('name'), state.get('hp_pct')
            if name and hp is not None:
                hp = float(hp)
                # Signed drop equals observed damage received minus own healing.
                net_received[side] += previous_hp[side].get(name, 1.0) - hp
                previous_hp[side][name] = hp
        curves['p1_net_damage_dealt'].append(net_received['p2'])
        curves['p2_net_damage_dealt'].append(net_received['p1'])
    curves = {key: np.asarray(values, dtype=float) for key, values in curves.items()}
    curves['turn'] = np.arange(1, MAX_OBSERVED_TURNS + 1)
    return curves


def extract_damage_surplus_features(battle):
    curves = damage_surplus_curves(battle)
    gap = curves['p1_net_damage_dealt'] - curves['p2_net_damage_dealt']
    return {
        'p1_damage_surplus': float(np.maximum(gap, 0).mean()),
        'p2_damage_surplus': float(np.maximum(-gap, 0).mean()),
    }


def extract_selected_features(battle, *, include_candidates=False, include_p2_roster_stats=True):
    """Return core numeric summaries, or all computed core candidates when enabled."""
    roster = battle.get('p1_team_details') or []
    lead = battle.get('p2_lead_details') or {}
    timeline = (battle.get('battle_timeline') or [])[:MAX_OBSERVED_TURNS]
    features = {}
    for stat in STATS:
        features[f'p1_mean_{stat}'] = float(np.mean([
            pokemon.get(f'base_{stat}', 0) for pokemon in roster
        ])) if roster else 0.0
    type_counts = Counter(
        str(kind).lower()
        for pokemon in roster
        for kind in pokemon.get('types', [])
        if str(kind).lower() != 'notype'
    )
    unknown_types = set(type_counts) - set(TYPES)
    if unknown_types:
        raise ValueError(f'Unknown real roster types: {sorted(unknown_types)}')
    features['p1_team_diversity'] = len(type_counts)
    for stat in STATS:
        features[f'p2_lead_{stat}'] = lead.get(f'base_{stat}', 0)

    fainted = {'p1': set(), 'p2': set()}
    previous_hp = {'p1': {}, 'p2': {}}
    previous_name = {'p1': None, 'p2': None}
    boost_sums = {'p1': [], 'p2': []}
    for side in ('p1', 'p2'):
        features[f'{side}_no_of_moves'] = 0
        features[f'{side}_dmg'] = 0.0
        features[f'{side}_heal'] = 0.0
        features[f'{side}_switch'] = 0

    for turn in timeline:
        for side in ('p1', 'p2'):
            opponent = 'p2' if side == 'p1' else 'p1'
            state = turn.get(f'{side}_pokemon_state') or {}
            name = state.get('name')
            boosts = state.get('boosts') or {}
            boost_sums[side].append(sum(boosts.get(stat, 0) for stat in BOOST_STATS))
            features[f'{side}_no_of_moves'] += turn.get(f'{side}_move_details') is not None
            if not name:
                previous_name[side] = None
                continue
            if previous_name[side] is not None and previous_name[side] != name:
                features[f'{side}_switch'] += 1
            previous_name[side] = name
            hp = state.get('hp_pct')
            if hp == 0 or state.get('status') == 'fnt':
                fainted[side].add(name)
            if hp is not None:
                hp = float(hp)
                old_hp = previous_hp[side].get(name, 1.0)
                features[f'{opponent}_dmg'] += max(old_hp - hp, 0.0)
                features[f'{side}_heal'] += max(hp - old_hp, 0.0)
                previous_hp[side][name] = hp

    for side in ('p1', 'p2'):
        features[f'{side}_alive_at_turn30'] = TEAM_SIZE - len(fainted[side])
        features[f'{side}_mean_boosts'] = float(np.mean(boost_sums[side])) if boost_sums[side] else 0.0
    features['alive_diff_at_turn30'] = features['p1_alive_at_turn30'] - features['p2_alive_at_turn30']
    features['first_ko_advantage'] = first_ko_advantage(battle)
    if include_p2_roster_stats:
        from pokemon_challenge.features.observed_roster import p2_observed_roster_stats
        features.update(p2_observed_roster_stats(battle))
    features.update(extract_requested_features(battle))
    features.update(extract_damage_surplus_features(battle))
    features.update(retained_position_features(battle))
    return features if include_candidates else {
        name: features[name] for name in FEATURE_COLUMNS
        if include_p2_roster_stats or name not in P2_ROSTER_STAT_COLUMNS}



def expanded_core_features(battle, *, include_p2_roster_stats=True):
    """Return complementary core summaries outside the fixed core schema."""
    all_features = extract_selected_features(
        battle, include_candidates=True, include_p2_roster_stats=include_p2_roster_stats)
    return {name: value for name, value in all_features.items() if name not in FEATURE_COLUMNS}


def build_feature_functions(chart_path, *, include_p2_roster_stats=True):
    """Build numeric feature groups using the supplied type-effectiveness chart.

    Callers should pass observations restricted to valid turns 1–30 in
    chronological order. Every group returns a fixed dictionary of numeric
    values. Unknown estimates remain NaN for imputation inside training folds.
    Disable P2 roster stats to avoid the static species lookup entirely.
    """
    from battle_features import FEATURE_GROUPS
    from team_features import make_team_feature_groups

    return {
        "selected": partial(extract_selected_features, include_p2_roster_stats=include_p2_roster_stats),
        "expanded_core": partial(expanded_core_features, include_p2_roster_stats=include_p2_roster_stats),
        **FEATURE_GROUPS,
        **make_team_feature_groups(chart_path),
    }
