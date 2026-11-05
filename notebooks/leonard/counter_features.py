"""Counter redundancy and HP quality beyond binary type coverage.

Counters are type-based proxies: own best same-type potential >1; a safe counter
also resists every enemy own-type option (best enemy effectiveness <1). No exact
moveset, damage or switching simulation is claimed. Unknown enemy types stay NaN.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from matchup_features import (summarize_battle,battle_metadata,best_effect,
                              MatchupTransformer,_mean,_weighted)

REDUNDANCY_KEYS=('counter_count_mean','counter_count_min','safe_counter_count_mean',
                'safe_counter_count_min','two_counter_coverage_fraction','two_safe_counter_coverage_fraction',
                'no_safe_counter_enemy_count','single_safe_counter_enemy_fraction')
QUALITY_KEYS=('best_counter_hp_mean','weakest_best_counter_hp','best_safe_counter_hp_mean',
              'weakest_best_safe_counter_hp','healthy_safe_counter_coverage_fraction',
              'counter_hp_reserve_mean','safe_counter_hp_reserve_mean','hp_weighted_uncovered_threat_fraction')


def extract_counters(battle,species_types,chart):
    battle=battle if battle.get('_matchup_summary') is True else summarize_battle(battle)
    types={**species_types,**battle_metadata(battle)};seen=battle['seen']
    own=[]
    for p in battle['p1_team_details']:
        s=seen['p1'].get(p['name']);hp=1. if s is None else (0. if s['status']=='fnt' else s['hp_pct'])
        if hp>0 and types.get(p['name']):own.append((p['name'],hp))
    own_alive=sum(1 for p in battle['p1_team_details'] if
                  p['name'] not in seen['p1'] or
                  (seen['p1'][p['name']]['hp_pct']>0 and seen['p1'][p['name']]['status']!='fnt'))
    enemies=[(name,s['hp_pct']) for name,s in seen['p2'].items()
             if s['hp_pct']>0 and s['status']!='fnt' and types.get(name)]
    # No known matchup does not mean no counter. Return missing values when all
    # revealed enemies have unknown types (or all alive own types are unknown).
    if not enemies or (own_alive and not own):
        return dict.fromkeys(REDUNDANCY_KEYS+QUALITY_KEYS,np.nan)
    counters=[];safe_counts=[];best_hp=[];best_safe_hp=[];reserves=[];safe_reserves=[];uncovered=[]
    for enemy,enemy_hp in enemies:
        good=[];safe=[]
        for name,hp in own:
            a=best_effect(types.get(name) or (),types.get(enemy),chart)
            b=best_effect(types.get(enemy) or (),types.get(name),chart)
            if np.isfinite(a) and a>1:
                good.append(hp)
                if np.isfinite(b) and b<1:safe.append(hp)
        counters.append(len(good));safe_counts.append(len(safe))
        best_hp.append(max(good,default=0.));best_safe_hp.append(max(safe,default=0.))
        reserves.append(sum(good));safe_reserves.append(sum(safe))
        uncovered.append((float(not safe),enemy_hp))
    row={'counter_count_mean':_mean(counters),'counter_count_min':float(min(counters)),
         'safe_counter_count_mean':_mean(safe_counts),'safe_counter_count_min':float(min(safe_counts)),
         'two_counter_coverage_fraction':_mean([float(n>=2) for n in counters]),
         'two_safe_counter_coverage_fraction':_mean([float(n>=2) for n in safe_counts]),
         'no_safe_counter_enemy_count':float(sum(n==0 for n in safe_counts)),
         'single_safe_counter_enemy_fraction':_mean([float(n==1) for n in safe_counts]),
         'best_counter_hp_mean':_mean(best_hp),'weakest_best_counter_hp':min(best_hp),
         'best_safe_counter_hp_mean':_mean(best_safe_hp),'weakest_best_safe_counter_hp':min(best_safe_hp),
         'healthy_safe_counter_coverage_fraction':_mean([float(hp>=.5) for hp in best_safe_hp]),
         'counter_hp_reserve_mean':_mean(reserves),'safe_counter_hp_reserve_mean':_mean(safe_reserves),
         'hp_weighted_uncovered_threat_fraction':_weighted(uncovered)}
    assert set(row)==set(REDUNDANCY_KEYS+QUALITY_KEYS)
    return row


class CounterTransformer(TransformerMixin,BaseEstimator):
    def __init__(self,chart_path,group='all'):
        self.chart_path=chart_path;self.group=group

    def fit(self,X,y=None):
        if self.group not in ('redundancy','quality','all'):raise ValueError('Unknown counter group')
        self.lookup_=MatchupTransformer(self.chart_path,'team').fit(X,y)
        self.feature_names_=list(REDUNDANCY_KEYS if self.group=='redundancy' else
                                 QUALITY_KEYS if self.group=='quality' else REDUNDANCY_KEYS+QUALITY_KEYS)
        return self

    def transform(self,X):
        rows=[extract_counters(b,self.lookup_.species_types_,self.lookup_.chart_) for b in X['battle']]
        return pd.DataFrame(rows,index=X.index)[self.feature_names_]

    def get_feature_names_out(self,input_features=None):
        return np.asarray(self.feature_names_,dtype=object)
