"""Three independent feature hypotheses using observed turns 1–30 only."""

STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


def observed_turns(battle):
    turns = sorted((t for t in battle['battle_timeline'] if 1 <= t['turn'] <= 30), key=lambda t: t['turn'])
    if not turns:
        raise ValueError('No observed turns in 1–30')
    return turns


def last_seen(turns, side):
    seen = {}
    for t in turns:
        seen[t[f'{side}_pokemon_state']['name']] = (t[f'{side}_pokemon_state'], t['turn'])
    return seen


def remaining_strength(battle):
    """HP-weighted own-team stats; opponent stats only for its known lead.

    Own unseen members start at full HP. Unknown opponent species never receive
    invented stats. Status=fnt overrides positive HP in inconsistent inputs.
    """
    turns = observed_turns(battle)
    own = last_seen(turns, 'p1')
    opponent = last_seen(turns, 'p2')
    team = battle['p1_team_details']
    row = {}
    for stat in STATS:
        strength = []
        for poke in team:
            state = own.get(poke['name'], ({'hp_pct': 1., 'status': 'nostatus'}, 0))[0]
            hp = 0. if state['status'] == 'fnt' else state['hp_pct']
            strength.append(hp * poke[f'base_{stat}'] * poke['level'] / 100.)
        row[f'p1_remaining_{stat}_sum'] = sum(strength)
        row[f'p1_remaining_{stat}_max'] = max(strength)
        lead = battle['p2_lead_details']
        state = opponent.get(lead['name'], ({'hp_pct': 1., 'status': 'nostatus'}, 0))[0]
        hp = 0. if state['status'] == 'fnt' else state['hp_pct']
        row[f'p2_known_lead_remaining_{stat}'] = hp * lead[f'base_{stat}'] * lead['level'] / 100.
    row['p1_unseen_members'] = sum(p['name'] not in own for p in team)
    row['p2_observed_species'] = len(opponent)
    row['p2_species_without_known_stats'] = sum(name != battle['p2_lead_details']['name'] for name in opponent)
    for side, seen in [('p1', own), ('p2', opponent)]:
        row[f'{side}_observation_age_mean'] = sum(turns[-1]['turn'] - t for _, t in seen.values()) / len(seen)
        row[f'{side}_observation_age_max'] = max(turns[-1]['turn'] - t for _, t in seen.values())
    return row


def advantage_trajectory(battle):
    """More frequent roster snapshots and recent changes in team advantage.

    Both sides' unseen members are provisionally full HP, as in the reference.
    Observation counts expose uncertainty in the opponent estimate.
    """
    turns = observed_turns(battle)
    rosters = {'p1': {p['name']: 1. for p in battle['p1_team_details']},
               'p2': {battle['p2_lead_details']['name']: 1.}}
    seen = {'p1': set(), 'p2': set()}
    row = {}
    for checkpoint in (5, 10, 15, 20, 25, 30):
        for t in turns:
            if t['turn'] > checkpoint:
                break
            for side in rosters:
                s = t[f'{side}_pokemon_state']
                rosters[side][s['name']] = 0. if s['status'] == 'fnt' else s['hp_pct']
                seen[side].add(s['name'])
        for side, roster in rosters.items():
            unseen = max(0, 6-len(roster))
            row[f'{side}_teamhp_t{checkpoint}'] = unseen + sum(roster.values())
            row[f'{side}_alive_t{checkpoint}'] = unseen + sum(v > 0 for v in roster.values())
            row[f'{side}_observed_t{checkpoint}'] = len(seen[side])
        for metric in ('teamhp', 'alive'):
            row[f'{metric}_diff_t{checkpoint}'] = row[f'p1_{metric}_t{checkpoint}'] - row[f'p2_{metric}_t{checkpoint}']
    for window in (5, 10):
        for prefix in ('p1_teamhp', 'p2_teamhp', 'teamhp_diff', 'p1_alive', 'p2_alive', 'alive_diff'):
            row[f'{prefix}_change_last{window}'] = row[f'{prefix}_t30'] - row[f'{prefix}_t{30-window}']
        for side in rosters:
            # Newly observed fainted identities in the window, excluding earlier faints.
            before = {t[f'{side}_pokemon_state']['name'] for t in turns if t['turn'] <= 30-window and t[f'{side}_pokemon_state']['status'] == 'fnt'}
            after = {t[f'{side}_pokemon_state']['name'] for t in turns if t['turn'] > 30-window and t[f'{side}_pokemon_state']['status'] == 'fnt'}
            row[f'{side}_new_faints_last{window}'] = len(after - before)
    # Existing roster group already contains these 18 values. Avoid repeating them.
    for checkpoint in (10, 20, 30):
        for prefix in ('p1_teamhp', 'p2_teamhp', 'teamhp_diff', 'p1_alive', 'p2_alive', 'alive_diff'):
            row.pop(f'{prefix}_t{checkpoint}')
    return row


def boost_threat(battle):
    """Separate attack types, HP-weighted stages and status interactions.

    Last-seen boosts may reset on switches; age and active flags distinguish
    stale bench observations. Values are observations, not exact damage estimates.
    """
    turns = observed_turns(battle)
    row = {}
    for side in ('p1', 'p2'):
        seen = last_seen(turns, side)
        active = turns[-1][f'{side}_pokemon_state']
        alive = [(s, t) for s, t in seen.values() if s['hp_pct'] > 0 and s['status'] != 'fnt']
        for stat in ('atk', 'spa', 'spe'):
            stage = active['boosts'][stat]
            hp = 0. if active['status'] == 'fnt' else active['hp_pct']
            row[f'{side}_active_{stat}_stage'] = stage
            row[f'{side}_active_{stat}_hp_stage'] = hp * stage
            for status in ('slp', 'par', 'brn'):
                row[f'{side}_active_{stat}_{status}_interaction'] = hp * stage * (active['status'] == status)
            threats = [max(s['boosts'][stat], 0) * s['hp_pct'] for s, _ in alive]
            row[f'{side}_observed_{stat}_threat_max'] = max(threats, default=0.)
            row[f'{side}_observed_{stat}_threat_sum'] = sum(threats)
            row[f'{side}_recent_{stat}_threat_max'] = max((max(s['boosts'][stat], 0)*s['hp_pct'] for s, t in alive if turns[-1]['turn']-t <= 5), default=0.)
    for key in list(row):
        if key.startswith('p1_'):
            row['diff_' + key[3:]] = row[key] - row['p2_' + key[3:]]
    return row
