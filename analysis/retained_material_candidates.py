"""Experimental remaining-resource features adapted from supplied 2025 material.

Source: ``2025/de-lucia-fratello-marchio.ipynb``, cell 7, final board-state
features ``p1_bulk_phys``, ``p1_bulk_spec`` and ``p{1,2}_healthy_mons``.
The low-HP counts and HP-weighted stat means are explicit extensions of that
notebook's remaining-resource idea, not claimed verbatim source features.

Only the first 30 timeline entries whose turn numbers lie in 1--30 are used.
Each species retains its latest known HP; a faint sets HP to zero. Initial P1
roster members and the P2 lead start full and healthy. Other unseen team slots
contribute only to healthy-count proxies, assuming six-member teams. No species
or base stats are invented for P2's hidden slots. An observed missing status is
unknown and does not count as healthy. Observed species with unknown HP do not
count as healthy or severely weakened. Bench states can be stale.

Bulk uses P1's supplied base HP times base defense and latest HP fraction;
it is a heuristic, not level-adjusted effective battle HP. Stat means are
weighted by HP fraction across P1's supplied roster, including unseen members
at full HP and excluding fainted members. They are zero if total weight is zero.
Sleep/freeze do not imply fainting. There is no fitted lookup, label access,
cross-battle state, external source, or inference of P2's complete roster.
"""

from math import isfinite


FEATURE_COLUMNS = (
    'p1_remaining_physical_bulk',
    'p1_remaining_special_bulk',
    'p1_healthy_remaining_count',
    'p2_healthy_remaining_count',
    'p1_severely_weakened_count',
    'p2_severely_weakened_count',
    'p1_hp_weighted_base_speed',
    'p1_hp_weighted_base_attack',
    'p1_hp_weighted_base_special_attack',
)


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def extract_retained_material_candidates(battle):
    """Return nine fixed numeric resource summaries, without using outcomes."""
    roster = {
        member['name']: member
        for member in (battle.get('p1_team_details') or [])
        if member.get('name')
    }
    states = {
        'p1': {name: {'hp': 1.0, 'status': 'nostatus'} for name in roster},
        'p2': {},
    }
    lead_name = (battle.get('p2_lead_details') or {}).get('name')
    if lead_name:
        states['p2'][lead_name] = {'hp': 1.0, 'status': 'nostatus'}

    for entry in (battle.get('battle_timeline') or [])[:30]:
        turn = _number(entry.get('turn'))
        if turn is None or not 1 <= turn <= 30:
            continue
        for side in ('p1', 'p2'):
            observation = entry.get(f'{side}_pokemon_state') or {}
            name = observation.get('name')
            if not name:
                continue
            previous = states[side].setdefault(name, {'hp': None, 'status': None})
            hp = _number(observation.get('hp_pct'))
            if hp is not None:
                previous['hp'] = min(1.0, max(0.0, hp))
            # Missing status on a named observation is unknown, not healthy.
            previous['status'] = observation.get('status')
            if previous['status'] == 'fnt':
                previous['hp'] = 0.0

    result = {}
    for side in ('p1', 'p2'):
        known = list(states[side].values())
        unseen_count = max(0, 6 - len(known))
        result[f'{side}_healthy_remaining_count'] = float(unseen_count + sum(
            state['hp'] is not None and state['hp'] > 0.5
            and state['status'] == 'nostatus'
            for state in known
        ))
        result[f'{side}_severely_weakened_count'] = float(sum(
            state['hp'] is not None and 0.0 < state['hp'] <= 0.25
            and state['status'] != 'fnt'
            for state in known
        ))

    def weight(name):
        state = states['p1'][name]
        return 0.0 if state['status'] == 'fnt' else (state['hp'] or 0.0)

    for defense, feature in (
        ('base_def', 'p1_remaining_physical_bulk'),
        ('base_spd', 'p1_remaining_special_bulk'),
    ):
        result[feature] = float(sum(
            (_number(member.get('base_hp')) or 0.0)
            * (_number(member.get(defense)) or 0.0) * weight(name)
            for name, member in roster.items()
        ))

    for stat, feature in (
        ('base_spe', 'p1_hp_weighted_base_speed'),
        ('base_atk', 'p1_hp_weighted_base_attack'),
        ('base_spa', 'p1_hp_weighted_base_special_attack'),
    ):
        weighted = [(_number(member.get(stat)), weight(name))
                    for name, member in roster.items()]
        eligible = [(value, hp) for value, hp in weighted if value is not None]
        denominator = sum(hp for _, hp in eligible)
        result[feature] = (sum(value * hp for value, hp in eligible) / denominator
                           if denominator > 0 else 0.0)
    return {name: result[name] for name in FEATURE_COLUMNS}
