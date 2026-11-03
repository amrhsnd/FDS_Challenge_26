"""Signed timing of the first observed faint within the first 30 turns."""


def first_ko_advantage(battle):
    """Positive when P2 faints first; zero for a same-turn tie or no faint."""
    first = {'p1': 31, 'p2': 31}
    for entry in (battle.get('battle_timeline') or [])[:30]:
        turn = entry.get('turn', 0)
        if not 1 <= turn <= 30:
            continue
        for side in first:
            state = entry.get(f'{side}_pokemon_state') or {}
            if state.get('hp_pct') == 0 or state.get('status') == 'fnt':
                first[side] = min(first[side], turn)
    if first['p1'] == first['p2']:
        return 0.0
    side = 1.0 if first['p2'] < first['p1'] else -1.0
    return side * (31 - min(first.values())) / 30
