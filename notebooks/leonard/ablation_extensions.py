"""Stateless, observation-only extensions for notebook 05 (no fitted statistics)."""


def battle_state_features(battle):
    """Track last-seen roster status and recent changes without switch HP jumps.

    Unobserved opponents are not assigned a status. Last-seen HP/status can be
    stale; exposure and observation-age features describe that uncertainty.
    """
    turns = [t for t in battle['battle_timeline'] if t['turn'] <= 30]
    if not turns:
        raise ValueError('At least one observed turn is required')
    row = {}
    for side in ('p1', 'p2'):
        seen = {}
        for t in turns:
            state = t[f'{side}_pokemon_state']
            seen[state['name']] = (state, t['turn'])
        states = [s for s, _ in seen.values()]
        alive = [s for s in states if s['hp_pct'] > 0 and s['status'] != 'fnt']
        values = {
            'observed_alive': len(alive),
            'observed_injured': sum(0 < s['hp_pct'] < .5 for s in states),
            'observed_hp_sq': sum(s['hp_pct'] ** 2 for s in states),
            'last_seen_age_mean': sum(turns[-1]['turn'] - t for _, t in seen.values()) / len(seen),
        }
        for status in ('slp', 'par', 'brn', 'psn', 'tox', 'frz'):
            values[f'roster_{status}'] = sum(s['status'] == status for s in alive)
            values[f'final_{status}'] = int(turns[-1][f'{side}_pokemon_state']['status'] == status)
        for window in (5, 10):
            recent = turns[-window:]
            recent_states = [t[f'{side}_pokemon_state'] for t in recent]
            values[f'recent_{window}_faints'] = len({s['name'] for s in recent_states if s['status'] == 'fnt'})
            values[f'recent_{window}_no_move'] = sum(t[f'{side}_move_details'] is None for t in recent) / len(recent)
            # Compare consecutive observations only for the same active species.
            changes = [b['hp_pct'] - a['hp_pct'] for a, b in zip(recent_states, recent_states[1:]) if a['name'] == b['name']]
            values[f'recent_{window}_hp_loss'] = sum(max(-v, 0) for v in changes)
            values[f'recent_{window}_hp_gain'] = sum(max(v, 0) for v in changes)
        for stat in ('atk', 'def', 'spa', 'spd', 'spe'):
            values[f'final_boost_{stat}'] = turns[-1][f'{side}_pokemon_state']['boosts'][stat]
        row.update({f'{side}_{key}': value for key, value in values.items()})
    for key in values:
        row[f'diff_{key}'] = row[f'p1_{key}'] - row[f'p2_{key}']
    return row
