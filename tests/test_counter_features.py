"""Counter count, fragility, HP weighting, uncertainty and training-only metadata."""
from pathlib import Path
import sys,copy,tempfile,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks/leonard'))
from matchup_features import read_chart
from counter_features import extract_counters,CounterTransformer,REDUNDANCY_KEYS,QUALITY_KEYS


def fixture(own_hps=(.9,.8),own_type='water'):
    own=[{'name':f'own{i}','types':[own_type,'notype']} for i in range(len(own_hps))]
    def state(name,hp):return {'name':name,'hp_pct':hp,'status':'nostatus'}
    turns=[]
    for i in range(1,31):
        j=min(i-1,len(own_hps)-1)
        turns.append({'turn':i,'p1_pokemon_state':state(f'own{j}',own_hps[j]),
                      'p2_pokemon_state':state('enemy',1.),'p1_move_details':None,'p2_move_details':None})
    return {'p1_team_details':own,'p2_lead_details':{'name':'enemy','types':['fire','notype']},'battle_timeline':turns}


class CounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/'chart.csv'
        cls.path.write_text('attacking_type,defending_type,multiplier\nwater,fire,2\nfire,water,0.5\ngrass,fire,0.5\nfire,grass,2\nwater,grass,0.5\ngrass,water,2\n')
        cls.chart=read_chart(cls.path)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_two_healthy_counters_have_redundancy_and_hp_reserves(self):
        r=extract_counters(fixture(),{},self.chart)
        self.assertEqual(r['counter_count_mean'],2.)
        self.assertEqual(r['safe_counter_count_min'],2.)
        self.assertEqual(r['two_safe_counter_coverage_fraction'],1.)
        self.assertEqual(r['single_safe_counter_enemy_fraction'],0.)
        self.assertAlmostEqual(r['best_safe_counter_hp_mean'],.9)
        self.assertAlmostEqual(r['safe_counter_hp_reserve_mean'],1.7)
        self.assertEqual(r['healthy_safe_counter_coverage_fraction'],1.)
        self.assertEqual(r['hp_weighted_uncovered_threat_fraction'],0.)

    def test_one_fragile_counter_is_not_a_healthy_counter(self):
        r=extract_counters(fixture((.1,)),{},self.chart)
        self.assertEqual(r['counter_count_mean'],1.)
        self.assertEqual(r['two_counter_coverage_fraction'],0.)
        self.assertEqual(r['single_safe_counter_enemy_fraction'],1.)
        self.assertEqual(r['healthy_safe_counter_coverage_fraction'],0.)
        self.assertAlmostEqual(r['weakest_best_safe_counter_hp'],.1)

    def test_fainted_counter_is_removed(self):
        b=fixture((.9,.4))
        b['battle_timeline'][0]['p1_pokemon_state'].update(hp_pct=0.,status='fnt')
        r=extract_counters(b,{},self.chart)
        self.assertEqual(r['safe_counter_count_mean'],1.)
        self.assertAlmostEqual(r['safe_counter_hp_reserve_mean'],.4)

    def test_uncovered_enemy_returns_zero_counters_not_missing(self):
        r=extract_counters(fixture((1.,),own_type='grass'),{},self.chart)
        self.assertEqual(r['counter_count_min'],0.)
        self.assertEqual(r['no_safe_counter_enemy_count'],1.)
        self.assertEqual(r['best_counter_hp_mean'],0.)
        self.assertEqual(r['hp_weighted_uncovered_threat_fraction'],1.)

    def test_unknown_enemy_is_missing_instead_of_counted_as_uncovered(self):
        b=fixture()
        for t in b['battle_timeline']:t['p2_pokemon_state']['name']='unknown'
        r=extract_counters(b,{},self.chart)
        self.assertTrue(all(np.isnan(v) for v in r.values()))

    def test_uncovered_threat_uses_enemy_hp_weight(self):
        b=fixture((1.,))
        b['battle_timeline'][0]['p2_pokemon_state']['hp_pct']=.8
        for t in b['battle_timeline'][1:]:t['p2_pokemon_state'].update(name='grass_enemy',hp_pct=.2)
        r=extract_counters(b,{'grass_enemy':('grass',)},self.chart)
        self.assertEqual(r['safe_counter_count_min'],0.)
        self.assertAlmostEqual(r['hp_weighted_uncovered_threat_fraction'],.2)

    def test_order_horizon_and_label_independence(self):
        b=fixture();c=copy.deepcopy(b);c['p1_team_details'].reverse();c.update(battle_id=4,player_won=1)
        t=copy.deepcopy(c['battle_timeline'][-1]);t['turn']=31;t['p2_pokemon_state']['name']='future'
        c['battle_timeline'].append(t)
        a=extract_counters(b,{},self.chart);z=extract_counters(c,{},self.chart)
        np.testing.assert_allclose([a[k] for k in REDUNDANCY_KEYS+QUALITY_KEYS],[z[k] for k in REDUNDANCY_KEYS+QUALITY_KEYS],equal_nan=True)

    def test_transformer_metadata_is_fold_local_and_schema_is_fixed(self):
        b=fixture();X=pd.DataFrame({'battle':[b]});c=copy.deepcopy(b)
        c['p1_team_details'].append({'name':'validation_only','types':['water']})
        for group,n in [('redundancy',8),('quality',8),('all',16)]:
            tr=CounterTransformer(str(self.path),group).fit(X)
            self.assertEqual(tr.transform(pd.DataFrame({'battle':[c]})).shape,(1,n))
            self.assertNotIn('validation_only',tr.lookup_.species_types_)

if __name__=='__main__':unittest.main()
