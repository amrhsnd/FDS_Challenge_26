"""Gen-2 type matchup proxies using only observed turns and known metadata.

Own unrevealed members start at full HP. Unrevealed enemies are never invented.
Species types are learned from training-fold metadata, with each input battle's
own known team/lead metadata overlaid at transform time. No labels are consulted.
"""
import csv
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin

TYPES=frozenset(('normal','fire','water','electric','grass','ice','fighting','poison',
                 'ground','flying','psychic','bug','rock','ghost','dragon','dark','steel'))
ACTIVE_KEYS=('p1_active_stab_best','p2_active_stab_best','active_stab_advantage',
             'p1_active_observed_move_best','p2_active_observed_move_best','active_observed_move_advantage',
             'p1_active_types_known','p2_active_types_known','p1_active_moves_known','p2_active_moves_known')
TEAM_KEYS=('favorable_pair_fraction','resistant_pair_fraction','immune_pair_fraction',
           'hp_weighted_offense','hp_weighted_enemy_offense','hp_weighted_advantage',
           'counter_coverage_fraction','safe_counter_coverage_fraction',
           'observed_move_favorable_pair_fraction','observed_move_resistance_pair_fraction',
           'observed_move_immunity_pair_fraction','hp_weighted_observed_move_advantage',
           'p2_revealed_type_coverage','p1_known_move_coverage','p2_known_move_coverage','p2_unseen_count')


def read_chart(path):
    chart={}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            key=(row['attacking_type'],row['defending_type'])
            if key in chart:raise ValueError('Duplicate chart entry')
            value=float(row['multiplier'])
            if not set(key)<=TYPES or value not in (0.,.5,1.,2.):raise ValueError('Invalid Gen-2 chart entry')
            chart[key]=value
    return chart


def real_types(poke):
    types=tuple(sorted(t for t in poke.get('types',()) if t!='notype'))
    return types if types and set(types)<=TYPES else None


def effectiveness(attack_type,defending_types,chart):
    if attack_type not in TYPES or not defending_types or not set(defending_types)<=TYPES:return np.nan
    # The supplied chart lists non-neutral entries only; missing valid pairs = 1.
    return math.prod(chart.get((attack_type,t),1.) for t in defending_types)


def best_effect(attack_types,defending_types,chart):
    values=[effectiveness(t,defending_types,chart) for t in attack_types]
    values=[v for v in values if np.isfinite(v)]
    return max(values) if values else np.nan


def battle_metadata(battle):
    return {p['name']:real_types(p) for p in [*battle['p1_team_details'],battle['p2_lead_details']]}


def _mean(values):
    return float(np.mean(values)) if values else np.nan


def _weighted(values):
    # [(value, weight)]; NaNs express unavailable knowledge, not neutral matchups.
    valid=[(v,w) for v,w in values if np.isfinite(v) and w>0]
    return sum(v*w for v,w in valid)/sum(w for _,w in valid) if valid else np.nan


def summarize_battle(battle):
    turns=sorted((t for t in battle['battle_timeline'] if 1<=t['turn']<=30),key=lambda t:t['turn'])
    if not turns:raise ValueError('No observed turns')
    seen={'p1':{},'p2':{}};moves={'p1':{},'p2':{}}
    for t in turns:
        for side in ('p1','p2'):
            s=t[f'{side}_pokemon_state'];seen[side][s['name']]=s
            move=t[f'{side}_move_details']
            # Unknown actual Hidden Power type and zero-power/fixed-damage moves
            # are omitted. These are type-coverage proxies, not damage estimates.
            if (move and move['category'] in ('physical','special') and move['base_power']>0
                    and move['name']!='hiddenpower' and move['type'] in TYPES):
                moves[side].setdefault(s['name'],set()).add(move['type'])
    active={side:turns[-1][f'{side}_pokemon_state']['name'] for side in ('p1','p2')}
    return {'_matchup_summary':True,
            'p1_team_details':[{'name':p['name'],'types':p['types']} for p in battle['p1_team_details']],
            'p2_lead_details':{'name':battle['p2_lead_details']['name'],'types':battle['p2_lead_details']['types']},
            'seen':{side:{name:{k:s[k] for k in ('name','hp_pct','status')} for name,s in states.items()} for side,states in seen.items()},
            'moves':{side:{name:sorted(values) for name,values in by_species.items()} for side,by_species in moves.items()},
            'active':active}


def extract_matchups(battle,species_types,chart):
    battle=battle if battle.get('_matchup_summary') is True else summarize_battle(battle)
    types={**species_types,**battle_metadata(battle)}
    seen=battle['seen'];moves=battle['moves'];active=battle['active']
    row={}
    for side,other in [('p1','p2'),('p2','p1')]:
        own=active[side];enemy=active[other]
        row[f'{side}_active_stab_best']=best_effect(types.get(own) or (),types.get(enemy),chart)
        row[f'{side}_active_observed_move_best']=best_effect(moves[side].get(own,set()),types.get(enemy),chart)
        row[f'{side}_active_types_known']=float(bool(types.get(own)))
        row[f'{side}_active_moves_known']=float(bool(moves[side].get(own)))
    row['active_stab_advantage']=row['p1_active_stab_best']-row['p2_active_stab_best']
    row['active_observed_move_advantage']=row['p1_active_observed_move_best']-row['p2_active_observed_move_best']
    own=[]
    for p in battle['p1_team_details']:
        s=seen['p1'].get(p['name']);hp=1. if s is None else (0. if s['status']=='fnt' else s['hp_pct'])
        if hp>0:own.append((p['name'],hp))
    enemies=[(name,s['hp_pct']) for name,s in seen['p2'].items() if s['hp_pct']>0 and s['status']!='fnt']
    favorable=[];resistant=[];immune=[];offense=[];enemy_offense=[];advantage=[]
    move_favorable=[];move_resistant=[];move_immune=[];move_advantage=[]
    coverage=[];safe=[]
    for enemy,ehp in enemies:
        favorable_counters=[];safe_counters=[]
        for name,hp in own:
            a=best_effect(types.get(name) or (),types.get(enemy),chart)
            b=best_effect(types.get(enemy) or (),types.get(name),chart)
            weight=hp*ehp
            if np.isfinite(a) and np.isfinite(b):
                favorable.append(float(a>1));resistant.append(float(b<1));immune.append(float(b==0))
                offense.append((a,weight));enemy_offense.append((b,weight));advantage.append((a-b,weight))
                favorable_counters.append(a>1);safe_counters.append(a>1 and b<1)
            ma=best_effect(moves['p1'].get(name,set()),types.get(enemy),chart)
            mb=best_effect(moves['p2'].get(enemy,set()),types.get(name),chart)
            if np.isfinite(ma):move_favorable.append(float(ma>1))
            if np.isfinite(mb):move_resistant.append(float(mb<1));move_immune.append(float(mb==0))
            if np.isfinite(ma) and np.isfinite(mb):move_advantage.append((ma-mb,weight))
        if favorable_counters:coverage.append(float(any(favorable_counters)));safe.append(float(any(safe_counters)))
    row.update(favorable_pair_fraction=_mean(favorable),resistant_pair_fraction=_mean(resistant),
               immune_pair_fraction=_mean(immune),hp_weighted_offense=_weighted(offense),
               hp_weighted_enemy_offense=_weighted(enemy_offense),hp_weighted_advantage=_weighted(advantage),
               counter_coverage_fraction=_mean(coverage),safe_counter_coverage_fraction=_mean(safe),
               observed_move_favorable_pair_fraction=_mean(move_favorable),
               observed_move_resistance_pair_fraction=_mean(move_resistant),
               observed_move_immunity_pair_fraction=_mean(move_immune),
               hp_weighted_observed_move_advantage=_weighted(move_advantage),
               p2_revealed_type_coverage=sum(bool(types.get(n)) for n,_ in enemies)/max(1,len(enemies)),
               p1_known_move_coverage=sum(bool(moves['p1'].get(n)) for n,_ in own)/max(1,len(own)),
               p2_known_move_coverage=sum(bool(moves['p2'].get(n)) for n,_ in enemies)/max(1,len(enemies)),
               p2_unseen_count=float(max(0,6-len(seen['p2']))))
    assert set(row)==set(ACTIVE_KEYS+TEAM_KEYS)
    return row


class MatchupTransformer(TransformerMixin,BaseEstimator):
    def __init__(self,chart_path,group='all'):
        self.chart_path=chart_path;self.group=group

    def fit(self,X,y=None):
        if self.group not in ('active','team','all'):raise ValueError('Unknown matchup group')
        self.chart_=read_chart(self.chart_path);self.species_types_={}
        for battle in X['battle']:
            for name,types in battle_metadata(battle).items():
                if types is None:continue
                if name in self.species_types_ and self.species_types_[name]!=types:raise ValueError('Conflicting species types')
                self.species_types_[name]=types
        self.feature_names_=list(ACTIVE_KEYS if self.group=='active' else TEAM_KEYS if self.group=='team' else ACTIVE_KEYS+TEAM_KEYS)
        return self

    def transform(self,X):
        rows=[extract_matchups(b,self.species_types_,self.chart_) for b in X['battle']]
        return pd.DataFrame(rows,index=X.index)[self.feature_names_]

    def get_feature_names_out(self,input_features=None):
        return np.asarray(self.feature_names_,dtype=object)
