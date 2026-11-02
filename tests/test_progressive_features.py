"""Synthetic checks for Leonard's three feature hypotheses."""
import copy
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'notebooks/leonard'))
from progressive_features import remaining_strength, advantage_trajectory, boost_threat


def fixture():
    team=[{'name':f'own{i}','level':100,**{f'base_{s}':100 for s in ('hp','atk','def','spa','spd','spe')}} for i in range(6)]
    def state(name):
        return {'name':name,'hp_pct':1.,'status':'nostatus','boosts':dict.fromkeys(('atk','def','spa','spd','spe'),0)}
    return {'p1_team_details':team,'p2_lead_details':{**team[0],'name':'opponent'},
            'battle_timeline':[{'turn':t,'p1_pokemon_state':state('own0'),'p2_pokemon_state':state('opponent')} for t in range(1,31)]}


class ProgressiveFeatureTests(unittest.TestCase):
    def test_strength_tracks_faints_and_preserves_unseen_team(self):
        b=fixture();b['battle_timeline'][-1]['p1_pokemon_state'].update(hp_pct=0,status='fnt')
        r=remaining_strength(b)
        self.assertEqual(r['p1_remaining_atk_sum'],500.)
        self.assertEqual(r['p1_remaining_atk_max'],100.)
        self.assertEqual(r['p1_unseen_members'],5)
        self.assertEqual(r['p2_species_without_known_stats'],0)
        # Only the lead has known opponent stats; no invented stats for others.
        b['battle_timeline'][-1]['p2_pokemon_state']['name']='unknown_species'
        self.assertEqual(remaining_strength(b)['p2_species_without_known_stats'],1)

    def test_team_momentum_and_new_faint(self):
        b=fixture();b['battle_timeline'][-1]['p1_pokemon_state'].update(hp_pct=0,status='fnt')
        r=advantage_trajectory(b)
        self.assertEqual(r['teamhp_diff_t25'],0)
        self.assertNotIn('teamhp_diff_t30',r)  # Already supplied by baseline.
        self.assertEqual(r['teamhp_diff_change_last5'],-1)
        self.assertEqual(r['p1_new_faints_last5'],1)
        self.assertEqual(r['p2_new_faints_last5'],0)

    def test_boost_hp_status_and_faint_interactions(self):
        b=fixture();s=b['battle_timeline'][-1]['p1_pokemon_state']
        s.update(hp_pct=.5,status='par');s['boosts']['atk']=2
        r=boost_threat(b)
        self.assertEqual(r['p1_active_atk_hp_stage'],1.)
        self.assertEqual(r['p1_active_atk_par_interaction'],1.)
        self.assertEqual(r['p1_observed_atk_threat_max'],1.)
        self.assertEqual(r['p1_active_spa_hp_stage'],0.)
        s.update(hp_pct=0,status='fnt')
        self.assertEqual(boost_threat(b)['p1_observed_atk_threat_max'],0.)

    def test_horizon_schema_and_target_independence(self):
        b=fixture();c=copy.deepcopy(b)
        c.update(battle_id='ignored',player_won=1)
        future=copy.deepcopy(c['battle_timeline'][-1]);future['turn']=31
        future['p1_pokemon_state'].update(hp_pct=0,status='fnt')
        c['battle_timeline'].append(future)
        for f in (remaining_strength,advantage_trajectory,boost_threat):
            self.assertEqual(f(b),f(c))
            self.assertTrue(all(math.isfinite(v) for v in f(c).values()))
            changed=copy.deepcopy(b)
            changed['battle_timeline'][-1]['p1_pokemon_state']['status']='slp'
            self.assertEqual(set(f(b)),set(f(changed)))

if __name__=='__main__': unittest.main()
