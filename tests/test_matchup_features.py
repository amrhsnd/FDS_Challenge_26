"""Gen-2 directions, dual types, uncertainty, horizon and fold-local metadata."""
import copy
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks/leonard'))
from matchup_features import read_chart,effectiveness,extract_matchups,MatchupTransformer,ACTIVE_KEYS,TEAM_KEYS


def fixture():
    def state(name):return {'name':name,'hp_pct':1.,'status':'nostatus'}
    def move(name,t):return {'name':name,'type':t,'category':'special','base_power':40}
    return {'p1_team_details':[{'name':'own','types':['water','notype']}],
            'p2_lead_details':{'name':'enemy','types':['fire','notype']},
            'battle_timeline':[{'turn':i,'p1_pokemon_state':state('own'),'p2_pokemon_state':state('enemy'),
                                'p1_move_details':move('watergun','water'),'p2_move_details':move('ember','fire')} for i in range(1,31)]}


class MatchupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.chart_path=Path(cls.temp.name)/'type_chart.csv'
        cls.chart_path.write_text('attacking_type,defending_type,multiplier\n'
            'water,fire,2\nfire,water,0.5\nfire,grass,2\nfire,steel,2\n'
            'electric,ground,0\nelectric,flying,2\nnormal,ghost,0\ngrass,fire,0.5\n')
        cls.chart=read_chart(cls.chart_path)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_direction_dual_types_immunity_and_omitted_neutral_entries(self):
        self.assertEqual(effectiveness('water',('fire',),self.chart),2.)
        self.assertEqual(effectiveness('fire',('water',),self.chart),.5)
        self.assertEqual(effectiveness('fire',('grass','steel'),self.chart),4.)
        self.assertEqual(effectiveness('electric',('ground','flying'),self.chart),0.)
        self.assertEqual(effectiveness('normal',('ghost',),self.chart),0.)
        self.assertEqual(effectiveness('fire',('fighting',),self.chart),1.)
        self.assertTrue(np.isnan(effectiveness('water',None,self.chart)))

    def test_actual_moves_and_hp_weighted_coverage(self):
        b=fixture();b['battle_timeline'][-1]['p1_pokemon_state']['hp_pct']=.5
        r=extract_matchups(b,{},self.chart)
        self.assertEqual(r['p1_active_stab_best'],2.)
        self.assertEqual(r['p2_active_stab_best'],.5)
        self.assertEqual(r['active_observed_move_advantage'],1.5)
        self.assertEqual(r['hp_weighted_advantage'],1.5)
        self.assertEqual(r['favorable_pair_fraction'],1.)
        self.assertEqual(r['resistant_pair_fraction'],1.)
        self.assertEqual(r['safe_counter_coverage_fraction'],1.)
        self.assertEqual(set(r),set(ACTIVE_KEYS+TEAM_KEYS))

    def test_unseen_own_member_uses_full_hp_and_team_order_is_irrelevant(self):
        b=fixture();b['p1_team_details'].append({'name':'bench','types':['grass']})
        b['battle_timeline'][-1]['p1_pokemon_state']['hp_pct']=.5
        r=extract_matchups(b,{},self.chart)
        # Water: advantage +1.5 at HP .5; unseen grass: -1.5 at HP 1.
        self.assertAlmostEqual(r['hp_weighted_advantage'],-.5)
        self.assertEqual(r['favorable_pair_fraction'],.5)
        self.assertEqual(r['p1_known_move_coverage'],.5)
        b['p1_team_details'].reverse();z=extract_matchups(b,{},self.chart)
        np.testing.assert_allclose([r[k] for k in ACTIVE_KEYS+TEAM_KEYS],[z[k] for k in ACTIVE_KEYS+TEAM_KEYS],equal_nan=True)

    def test_unknown_enemy_is_not_assumed_neutral_but_its_observed_move_is_usable(self):
        b=fixture()
        for t in b['battle_timeline']:t['p2_pokemon_state']['name']='unknown_enemy'
        r=extract_matchups(b,{},self.chart)
        self.assertEqual(r['p2_active_types_known'],0.)
        self.assertTrue(np.isnan(r['p1_active_stab_best']))
        self.assertEqual(r['p2_active_observed_move_best'],.5)
        self.assertEqual(r['p2_revealed_type_coverage'],0.)

    def test_hidden_power_and_status_moves_do_not_invent_attack_types(self):
        b=fixture()
        for t in b['battle_timeline']:t['p1_move_details'].update(name='hiddenpower',type='normal',base_power=60)
        r=extract_matchups(b,{},self.chart)
        self.assertEqual(r['p1_active_moves_known'],0.)
        self.assertTrue(np.isnan(r['p1_active_observed_move_best']))
        self.assertEqual(r['p1_active_stab_best'],2.)

    def test_horizon_and_target_independence(self):
        b=fixture();c=copy.deepcopy(b);c.update(battle_id=123,player_won=1)
        future=copy.deepcopy(c['battle_timeline'][-1]);future['turn']=31
        future['p1_pokemon_state']['name']='future_species';c['battle_timeline'].append(future)
        a=extract_matchups(b,{},self.chart);z=extract_matchups(c,{},self.chart)
        np.testing.assert_allclose([a[k] for k in ACTIVE_KEYS+TEAM_KEYS],[z[k] for k in ACTIVE_KEYS+TEAM_KEYS],equal_nan=True)

    def test_validation_metadata_never_mutates_training_lookup(self):
        tr=MatchupTransformer(str(self.chart_path)).fit(pd.DataFrame({'battle':[fixture()]}))
        val=fixture();val['p1_team_details'].append({'name':'validation_only','types':['grass']})
        tr.transform(pd.DataFrame({'battle':[val]}))
        self.assertNotIn('validation_only',tr.species_types_)
        self.assertEqual(tr.transform(pd.DataFrame({'battle':[val]})).shape,(1,26))

    def test_all_fainted_rosters_do_not_divide_by_zero(self):
        b=fixture()
        for side in ('p1','p2'):b['battle_timeline'][-1][f'{side}_pokemon_state'].update(hp_pct=0.,status='fnt')
        r=extract_matchups(b,{},self.chart)
        self.assertFalse(any(np.isinf(v) for v in r.values()))
        self.assertTrue(np.isnan(r['hp_weighted_advantage']))

if __name__=='__main__':unittest.main()
