"""Fixed 2x2 comparison: full/compact features and original/simpler boosting.

Run from repository root:
.venv/bin/python notebooks/leonard/run_simplification_experiment.py
Only the previously permitted development folds are used.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'), str(Path(__file__).parent), str(ROOT/'notebooks/aimilia')]
import json
import shutil
from datetime import datetime
import platform
import subprocess
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from pokemon_challenge.models.classifiers import make_pipeline
from pokemon_challenge.paths import data_dir
from run_progressive_experiments import digest

PRIOR=ROOT/'outputs/leonard/20261102_121822_824155_progressive_features'


def compact_columns():
    # Prespecified from earlier findings, not selected using validation labels.
    roster=[]
    for turn in (10,20,30):
        roster += [f'original__{name}_t{turn}' for name in
                   ('p1_alive','p2_alive','alive_diff','p1_teamhp','p2_teamhp','teamhp_diff')]
    roster += ['original__'+name for name in ('p1_teamhp_decline','p2_teamhp_decline','teamhp_decline_diff')]
    health=[f'original__{side}_{name}' for side in ('p1','p2')
            for name in ('faints','hppct_final','hppct_late_mean')]
    # Drop all-time last-seen bench boosts; keep current HP-weighted boosts
    # and maximum threat observed within five turns. These remain estimates.
    boosts=[f'boost_threat__{side}_{name}' for side in ('p1','p2','diff')
            for stat in ('atk','spa','spe')
            for name in (f'active_{stat}_hp_stage',f'recent_{stat}_threat_max')]
    return roster+health+boosts


def run():
    prior=json.loads((PRIOR/'experiment.json').read_text())
    assert digest(PRIOR/'train_features.csv')==prior['feature_sha256'], 'Changed feature cache'
    assert digest(PRIOR/'splits.csv')==prior['splits_sha256'], 'Changed folds'
    assert digest(data_dir()/'train.jsonl')==prior['train_sha256'], 'Changed training data'
    assert sklearn.__version__==prior['sklearn'], 'Use the prior sklearn version'
    X=pd.read_csv(PRIOR/'train_features.csv')
    assignments=pd.read_csv(PRIOR/'splits.csv')
    dev=np.flatnonzero(assignments.partition.to_numpy()=='development')
    previous=pd.read_csv(PRIOR/'oof_hgb_boost_threat.csv')
    assert len(X)==len(assignments) and len(dev)==7996
    assert previous.battle_id.tolist()==assignments.iloc[dev].battle_id.tolist()
    assert previous.fold.tolist()==assignments.iloc[dev].validation_fold.tolist()
    assert previous.battle_id.is_unique
    is_development = assignments.partition.eq('development')
    assert is_development.groupby(assignments.observation_sha256).nunique().max()==1
    assert assignments.iloc[dev].groupby('observation_sha256').validation_fold.nunique().max()==1
    y=previous.actual.to_numpy();X=X.iloc[dev].reset_index(drop=True)
    assert np.isfinite(X.to_numpy()).all()
    fold_ids=previous.fold.to_numpy()
    assert set(fold_ids)==set(range(5))
    configs=json.loads((PRIOR/'candidates.json').read_text())
    full=next(c for c in configs if c['candidate']=='hgb_boost_threat')
    compact=compact_columns()
    assert len(compact)==len(set(compact))==45 and set(compact)<=set(full['columns'])
    feature_sets={'full':full['columns'],'compact':compact}
    assert len(feature_sets['full'])==289
    original=full['params']
    assert original=={'max_iter':200,'learning_rate':.06,'max_leaf_nodes':15,'min_samples_leaf':40,'l2_regularization':5.,'early_stopping':False}
    models={'original':original,'simpler':{**original,'max_leaf_nodes':7,'min_samples_leaf':80,'l2_regularization':10.}}
    folder=ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_simplification')
    folder.mkdir(parents=True)
    shutil.copy2(PRIOR/'splits.csv',folder/'splits.csv')
    manifest={'prior_run':str(PRIOR.relative_to(ROOT)), 'seed':42,'development_rows':len(dev),
              'cv':'same five duplicate-grouped folds as prior run',
              'splits_sha256':prior['splits_sha256'],'train_sha256':prior['train_sha256'],
              'feature_cache_sha256':prior['feature_sha256'],
              'design':'Prespecified 2x2 factorial: 289/45 features x original/simpler HGB. No adaptive search.',
              'feature_sets':feature_sets,'model_params':models,
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'holdout_used':False,'test_used':False,
              'limitations':'Previously explored development set. Repeated comparisons can overfit CV. Exact duplicates controlled, nonidentical related battles not audited. Recent bench boosts may still reset when switching.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for path in [Path(__file__), ROOT/'src/pokemon_challenge/models/classifiers.py']:
        shutil.copy2(path,folder/path.name)
    # Preserve the original feature code/configuration alongside this experiment.
    for name in ('progressive_features.py','ablation_extensions.py','run_progressive_experiments.py',
                 'features_static.py','features_timeline.py','features_effects.py','features_roster.py'):
        shutil.copy2(PRIOR/name,folder/name)
    print('Artifacts:',folder,flush=True)
    results=[];predictions={}
    with threadpool_limits(limits=2):
        for feature_name,cols in feature_sets.items():
            for model_name,params in models.items():
                name=f'{feature_name}_{model_name}'
                pred=np.full(len(dev),-1,dtype=int);prob=np.full(len(dev),np.nan)
                scores=[]
                import time
                started=time.perf_counter()
                for fold in range(5):
                    val=np.flatnonzero(fold_ids==fold);train=np.flatnonzero(fold_ids!=fold)
                    assert set(train).isdisjoint(val) and len(train)+len(val)==len(dev)
                    pipe=make_pipeline('hist_gradient_boosting',seed=42,params=params)
                    pipe.fit(X.iloc[train][cols],y[train])
                    pred[val]=pipe.predict(X.iloc[val][cols])
                    prob[val]=pipe.predict_proba(X.iloc[val][cols])[:,1]
                    score=float((pred[val]==y[val]).mean());scores.append(score)
                    print(f'{name} fold {fold+1}: {score:.6f}',flush=True)
                assert set(pred)<={0,1} and np.isfinite(prob).all()
                if name=='full_original':
                    np.testing.assert_array_equal(pred,previous.predicted.to_numpy())
                    np.testing.assert_allclose(prob,previous.probability.to_numpy(),rtol=1e-10,atol=1e-12)
                predictions[name]=pred
                oof=pd.DataFrame({'battle_id':previous.battle_id,'actual':y,'fold':fold_ids,'predicted':pred,'probability':prob})
                oof.to_csv(folder/f'oof_{name}.csv',index=False)
                results.append({'candidate':name,'features':len(cols),'max_leaf_nodes':params['max_leaf_nodes'],
                                'min_samples_leaf':params['min_samples_leaf'],'l2_regularization':params['l2_regularization'],
                                'cv_accuracy':float(np.mean(scores)),'oof_accuracy':float((pred==y).mean()),
                                'fold_std':float(np.std(scores)),'fit_predict_seconds':time.perf_counter()-started,
                                **{f'fold_{i+1}':v for i,v in enumerate(scores)}})
                pd.DataFrame(results).to_csv(folder/'cv_results.csv',index=False)
    table=pd.DataFrame(results).set_index('candidate');pairs=[]
    # Main changes and interactions, all measured on exactly the same folds.
    for name,baseline in [('full_simpler','full_original'),('compact_original','full_original'),
                          ('compact_simpler','full_original'),('compact_simpler','compact_original'),
                          ('compact_simpler','full_simpler')]:
        delta=np.array([table.loc[name,f'fold_{i}']-table.loc[baseline,f'fold_{i}'] for i in range(1,6)])
        correct=predictions[name]==y;base_correct=predictions[baseline]==y
        pairs.append({'candidate':name,'baseline':baseline,'delta_pp':100*delta.mean(),
                      'folds_better':int((delta>1e-12).sum()),'folds_worse':int((delta < -1e-12).sum()),
                      'corrected_errors':int((correct & ~base_correct).sum()),'introduced_errors':int((~correct & base_correct).sum()),
                      **{f'delta_fold_{i+1}_pp':100*v for i,v in enumerate(delta)}})
    paired=pd.DataFrame(pairs);paired.to_csv(folder/'paired_comparison.csv',index=False)
    best=table.cv_accuracy.idxmax()
    validation={'checks_passed':True,'development_rows':len(dev),'candidates':4,'model_fits':20,
                'prior_baseline_labels_and_probabilities_reproduced':True,'group_overlap':0,
                'complete_oof_coverage':True,'holdout_used':False,'test_used':False,
                'best_candidate':best,'best_cv_accuracy':float(table.loc[best,'cv_accuracy']),
                'baseline_cv_accuracy':float(table.loc['full_original','cv_accuracy'])}
    (folder/'validation_report.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print(paired.to_string(index=False),flush=True)
    print('Finished:',folder,flush=True)
    return folder

if __name__=='__main__':
    run()
