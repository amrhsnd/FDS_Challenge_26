"""Base-stat averages for the distinct observed members of P2's roster."""
import json
from pathlib import Path

# Static species attributes from supplied team/lead records; no outcome information.
SPECIES_BASE_STATS = json.loads(Path(__file__).with_name('species_base_stats.json').read_text())
ROSTER_STATS = ('hp', 'atk', 'def', 'spe')


def p2_observed_roster_stats(battle):
    """Count each species once, including the initial lead and first 30 entries."""
    names = set()
    lead_name = (battle.get('p2_lead_details') or {}).get('name')
    if lead_name:
        names.add(lead_name)
    for turn in (battle.get('battle_timeline') or [])[:30]:
        if not 1 <= turn.get('turn', 0) <= 30:
            continue
        name = (turn.get('p2_pokemon_state') or {}).get('name')
        if name:
            names.add(name)
    unknown = names - SPECIES_BASE_STATS.keys()
    if unknown:
        raise ValueError(f'Missing base stats for observed P2 species: {sorted(unknown)}')
    return {
        f'p2_mean_{stat}': (sum(SPECIES_BASE_STATS[name][stat] for name in sorted(names)) / len(names)
                            if names else 0.0)
        for stat in ROSTER_STATS
    }
