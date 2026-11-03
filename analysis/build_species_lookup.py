"""Build static lookup from permitted records, with five verified Showdown fallbacks."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
fields=('hp','atk','def','spe')
stats={};seen=set()
for fn in ('train.jsonl','test.jsonl'):
 for line in (ROOT/'data/raw'/fn).open():
  b=json.loads(line)
  for p in [*b['p1_team_details'],b['p2_lead_details']]:
   row={s:p['base_'+s] for s in fields}
   assert p['name'] not in stats or stats[p['name']]==row
   stats[p['name']]=row
  seen.update(t['p2_pokemon_state']['name'] for t in b['battle_timeline'][:30])
fallback={'charizardmegay':(78,104,78,100),'weedle':(40,35,30,50),'ledyba':(40,20,30,55),'nidorina':(70,62,67,56),'remoraid':(35,65,35,65)}
for name,values in fallback.items():
 assert name not in stats
 stats[name]=dict(zip(fields,values))
assert not seen-set(stats),sorted(seen-set(stats))
p=ROOT/'src/pokemon_challenge/features/species_base_stats.json'
p.write_text(json.dumps(dict(sorted(stats.items())),indent=2)+'\n')
p.with_name('species_base_stats_sources.json').write_text(json.dumps({'primary':'supplied train/test p1_team_details and p2_lead_details; no outcome labels','fallback_species':list(fallback),'fallback_url':'https://github.com/smogon/pokemon-showdown/blob/master/data/pokedex.ts'},indent=2)+'\n')
print('Verified full coverage:',len(seen),'observed P2 species;',len(stats),'lookup entries')
