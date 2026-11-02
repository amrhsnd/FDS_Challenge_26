"""Observation-only species/move features and fold-fitted identity encoding."""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

CAT_KEYS=('p1_active','p2_active','p2_lead','p1_last_move','p2_last_move',
          'p1_most_used_move','p2_most_used_move')


def identity_features(battle):
    """No IDs/targets. Unseen opponents stay unknown; own unseen HP defaults to 1.

    Per-species last HP can be stale. Move names are observations, not assumed
    complete movesets. Dynamic dictionaries are encoded using training-fold-only
    vocabularies; new species/moves at inference are safely ignored by that encoder.
    """
    turns=sorted((t for t in battle['battle_timeline'] if 1 <= t['turn'] <= 30),key=lambda t:t['turn'])
    if not turns: raise ValueError('No observed turns')
    numeric={};categorical={'p2_lead':battle['p2_lead_details']['name']}
    for side in ('p1','p2'):
        seen={};species_count=Counter();moves=Counter();recent=Counter();last_used={};types=Counter()
        for t in turns:
            state=t[f'{side}_pokemon_state'];name=state['name']
            seen[name]=state;species_count[name]+=1
            move=t[f'{side}_move_details']
            if move is not None:
                moves[move['name']]+=1;last_used[move['name']]=t['turn'];types[move['type']]+=1
                if t['turn'] > turns[-1]['turn']-5:recent[move['name']]+=1
        for name,state in seen.items():
            numeric[f'{side}_seen_species::{name}']=1.
            numeric[f'{side}_species_exposure::{name}']=species_count[name]/len(turns)
            numeric[f'{side}_last_hp::{name}']=0. if state['status']=='fnt' else state['hp_pct']
        for name,count in moves.items():
            numeric[f'{side}_move_rate::{name}']=count/len(turns)
            numeric[f'{side}_recent_move_rate::{name}']=recent[name]/min(5,len(turns))
            numeric[f'{side}_move_recency::{name}']=1./(1+turns[-1]['turn']-last_used[name])
        for name,count in types.items():numeric[f'{side}_move_type_rate::{name}']=count/len(turns)
        categorical[f'{side}_active']=turns[-1][f'{side}_pokemon_state']['name']
        categorical[f'{side}_last_move']=max(last_used,key=last_used.get) if last_used else '<none>'
        categorical[f'{side}_most_used_move']=min(moves,key=lambda n:(-moves[n],n)) if moves else '<none>'
        if side=='p1':
            for poke in battle['p1_team_details']:
                name=poke['name'];state=seen.get(name)
                numeric[f'p1_team_species::{name}']=1.
                numeric[f'p1_team_species_hp::{name}']=1. if state is None else (0. if state['status']=='fnt' else state['hp_pct'])
    assert set(categorical)==set(CAT_KEYS)
    return {'identity_numeric':numeric,**categorical}


class IdentityEncoder(TransformerMixin,BaseEstimator):
    """Fold-fitted one-hot species/move vocabulary plus fixed numeric features."""
    def __init__(self,base_columns):
        self.base_columns=base_columns

    def _dictionaries(self,X):
        rows=[]
        for row in X[['identity_numeric',*CAT_KEYS]].to_dict('records'):
            values=dict(row['identity_numeric'])
            values.update({k:row[k] for k in CAT_KEYS})
            rows.append(values)
        return rows

    def fit(self,X,y=None):
        self.vectorizer_=DictVectorizer(sparse=False,dtype=np.float64)
        self.vectorizer_.fit(self._dictionaries(X))
        self.numeric_columns_=[f'n{i}' for i in range(len(self.base_columns)+len(self.vectorizer_.feature_names_))]
        return self

    def transform(self,X):
        identity=self.vectorizer_.transform(self._dictionaries(X))
        numbers=np.column_stack([X[self.base_columns].to_numpy(dtype=float),identity])
        if not np.isfinite(numbers).all():raise ValueError('Nonfinite model input')
        return pd.DataFrame(numbers,columns=self.numeric_columns_,index=X.index)

    def get_feature_names_out(self,input_features=None):
        return np.asarray(self.numeric_columns_,dtype=object)
