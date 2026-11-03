"""Species/move encoding invariants and fold-vocabulary isolation."""
import copy
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks/leonard'))
from identity_features import identity_features,IdentityEncoder,CAT_KEYS


def fixture():
    def state(name):return {'name':name,'hp_pct':1.,'status':'nostatus'}
    move={'name':'testmove','type':'normal'}
    return {'p1_team_details':[{'name':'a'},{'name':'b'}], 'p2_lead_details':{'name':'c'},
            'battle_timeline':[{'turn':t,'p1_pokemon_state':state('a'), 'p2_pokemon_state':state('c'),
                                'p1_move_details':dict(move),'p2_move_details':None} for t in range(1,31)]}


class IdentityTests(unittest.TestCase):
    def test_team_order_labels_and_future_turns_do_not_change_features(self):
        b=fixture();c=copy.deepcopy(b);c['p1_team_details'].reverse();c.update(battle_id=5,player_won=1)
        t=copy.deepcopy(c['battle_timeline'][-1]);t['turn']=31;t['p1_move_details']['name']='future_move'
        c['battle_timeline'].append(t)
        self.assertEqual(identity_features(b),identity_features(c))

    def test_move_counts_recency_and_unseen_own_hp(self):
        r=identity_features(fixture());n=r['identity_numeric']
        self.assertEqual(n['p1_move_rate::testmove'],1.)
        self.assertEqual(n['p1_recent_move_rate::testmove'],1.)
        self.assertEqual(n['p1_move_recency::testmove'],1.)
        self.assertEqual(n['p1_team_species_hp::b'],1.)
        self.assertEqual(r['p2_last_move'],'<none>')
        self.assertNotIn('p2_team_species_hp::unknown',n)

    def test_validation_only_species_do_not_enter_training_vocabulary(self):
        train=pd.DataFrame([{'base':1.,**identity_features(fixture())}])
        b=fixture();b['p1_team_details'].append({'name':'validation_only'})
        b['battle_timeline'][-1]['p2_pokemon_state']['name']='new_opponent'
        val=pd.DataFrame([{'base':2.,**identity_features(b)}])
        e=IdentityEncoder(['base']).fit(train)
        self.assertFalse(any('validation_only' in k for k in e.vectorizer_.feature_names_))
        self.assertFalse(any('new_opponent' in k for k in e.vectorizer_.feature_names_))
        self.assertEqual(e.transform(train).shape,e.transform(val).shape)

if __name__=='__main__':unittest.main()
