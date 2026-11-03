"""Validate three feature groups independently; never evaluate old holdout/test.

Run: .venv/bin/python notebooks/leonard/run_progressive_experiments.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'notebooks/aimilia'), str(Path(__file__).parent)]
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from pokemon_challenge.data import iter_battles
from pokemon_challenge.paths import data_dir
from pokemon_challenge.models.classifiers import make_pipeline
import features_static as static
import features_timeline as timeline
import features_effects as effects
import features_roster as roster
from ablation_extensions import battle_state_features
from progressive_features import remaining_strength, advantage_trajectory, boost_threat

OLD_RUN = ROOT/'outputs/leonard/20261101_230041_643033_aimilia_feature_ablation'
HYPOTHESES = {'remaining_strength': remaining_strength, 'advantage_trajectory': advantage_trajectory, 'boost_threat': boost_threat}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def feature_row(b):
    # Whitelist observed inputs, exclude labels/IDs, enforce the 30-turn horizon.
    observed = {k: b[k] for k in ('p1_team_details', 'p2_lead_details', 'battle_timeline')}
    observed = {**observed, 'battle_timeline': [t for t in observed['battle_timeline'] if 1 <= t['turn'] <= 30], 'battle_id': 0}
    clean = lambda d: {k:v for k,v in d.items() if k not in ('battle_id', 'player_won')}
    ef = clean(effects.battle_to_row(observed))
    original = {**clean(static.battle_to_row(observed, labelled=False)),
                **clean(timeline.battle_to_row(observed)),
                **{k:v for k,v in ef.items() if 'boost' in k},
                **clean(roster.battle_to_row(observed))}
    assert len(original) == 106
    groups = {'original': original, 'effects': {k:v for k,v in ef.items() if 'eff_' in k},
              'battle_state': battle_state_features(observed),
              **{name: f(observed) for name,f in HYPOTHESES.items()}}
    return {f'{group}__{k}': v for group, values in groups.items() for k,v in values.items()}


def run():
    folder = ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_progressive_features')
    folder.mkdir(parents=True)
    print('Artifacts:', folder, flush=True)
    old = pd.read_csv(OLD_RUN/'splits.csv').set_index('battle_id')
    ids, labels, hashes, rows = [], [], [], []
    for b in iter_battles(data_dir()/'train.jsonl'):
        ids.append(b['battle_id']); labels.append(int(b['player_won']))
        observation = {k:v for k,v in b.items() if k not in ('battle_id','player_won')}
        hashes.append(hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        rows.append(feature_row(b))
    X = pd.DataFrame(rows, dtype=float)
    y = np.asarray(labels); ids = np.asarray(ids); hashes = np.asarray(hashes)
    assert len(set(ids)) == len(ids) and set(ids) == set(old.index)
    assert not np.isinf(X.to_numpy()).any()
    # Quarantine every duplicate of the previously opened holdout as well.
    old_holdout = np.array([old.loc[i, 'partition'] == 'holdout' for i in ids])
    quarantined_groups = set(hashes[old_holdout])
    allowed = np.array([h not in quarantined_groups for h in hashes])
    dev = np.flatnonzero(allowed)
    assignments = pd.DataFrame({'battle_id': ids, 'observation_sha256': hashes,
                                'partition': np.where(old_holdout, 'previous_holdout', np.where(allowed, 'development', 'holdout_duplicate_quarantine')),
                                'validation_fold': -1})
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X.iloc[dev], y[dev], hashes[dev]))
    for fold, (train, val) in enumerate(folds):
        assert not set(hashes[dev[train]]) & set(hashes[dev[val]])
        assert not set(hashes[dev[train]]) & quarantined_groups
        assignments.loc[dev[val], 'validation_fold'] = fold
    assignments.to_csv(folder/'splits.csv', index=False)
    X.to_csv(folder/'train_features.csv', index=False)
    settings = {'lr': ('logistic_regression', {'C':1., 'max_iter':4000, 'solver':'lbfgs'}),
                'hgb': ('hist_gradient_boosting', {'max_iter':200, 'learning_rate':.06, 'max_leaf_nodes':15,
                       'min_samples_leaf':40, 'l2_regularization':5., 'early_stopping':False})}
    manifest = {'seed':42, 'metric':'accuracy', 'cv':'StratifiedGroupKFold', 'folds':5,
                'development_rows':len(dev), 'previous_holdout_rows':int(old_holdout.sum()),
                'quarantined_duplicate_rows':int((~allowed & ~old_holdout).sum()),
                'old_split_sha256':digest(OLD_RUN/'splits.csv'), 'splits_sha256':digest(folder/'splits.csv'),
                'train_sha256':digest(data_dir()/'train.jsonl'), 'feature_sha256':digest(folder/'train_features.csv'),
                'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__, 'sklearn':sklearn.__version__,
                'models':settings, 'reference_features':{'lr':'original 106', 'hgb':'previous extended 217'},
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'working_tree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                'holdout_used':False, 'test_used':False,
                'limitations':'Development data was explored previously. Exact duplicate groups only; nonidentical related battles may remain. No new independent generalization estimate.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for path in [Path(__file__), Path(__file__).with_name('progressive_features.py'), Path(__file__).with_name('ablation_extensions.py'),
                 Path(static.__file__), Path(timeline.__file__), Path(effects.__file__), Path(roster.__file__),
                 ROOT/'src/pokemon_challenge/models/classifiers.py']:
        shutil.copy2(path, folder/path.name)
    print(f'{len(dev)} development rows; {old_holdout.sum()} previous holdout; {manifest["quarantined_duplicate_rows"]} quarantined duplicates',flush=True)
    results, oof, configs = [], {}, []
    with threadpool_limits(limits=2):
        for family, (model, params) in settings.items():
            base_groups = ['original'] if family == 'lr' else ['original','effects','battle_state']
            for addition in [None, *HYPOTHESES]:
                name = family+'_'+(addition or 'baseline')
                groups = base_groups + ([addition] if addition else [])
                cols = [c for c in X if c.split('__')[0] in groups]
                if addition is None: assert len(cols) == (106 if family=='lr' else 217)
                prediction = np.full(len(dev), -1, dtype=int)
                probs = np.full(len(dev), np.nan)
                scores = []
                for fold, (train, val) in enumerate(folds,1):
                    pipeline = make_pipeline(model, seed=42, params=params)
                    pipeline.fit(X.iloc[dev[train]][cols], y[dev[train]])
                    prediction[val] = pipeline.predict(X.iloc[dev[val]][cols])
                    probs[val] = pipeline.predict_proba(X.iloc[dev[val]][cols])[:,1]
                    score = accuracy_score(y[dev[val]], prediction[val]);scores.append(score)
                    print(f'{name} fold {fold}: {score:.5f}',flush=True)
                assert not np.isnan(probs).any() and set(prediction) <= {0,1}
                oof[name] = prediction
                configs.append({'candidate':name,'model':model,'params':params,'groups':groups,'columns':cols})
                results.append({'candidate':name,'family':family,'added_group':addition or 'baseline','input_features':len(cols),
                                'cv_accuracy':float(np.mean(scores)), 'oof_accuracy':float(accuracy_score(y[dev],prediction)),
                                'fold_std':float(np.std(scores)), **{f'fold_{i+1}':v for i,v in enumerate(scores)}})
                pd.DataFrame(results).to_csv(folder/'cv_results.csv',index=False)
                pd.DataFrame({'battle_id':ids[dev],'actual':y[dev],'predicted':prediction,'probability':probs,
                              'fold':assignments.loc[dev,'validation_fold'].to_numpy()}).to_csv(folder/f'oof_{name}.csv',index=False)
    table = pd.DataFrame(results).set_index('candidate')
    comparisons=[]
    for family in settings:
        baseline = table.loc[family+'_baseline']
        for addition in HYPOTHESES:
            name=family+'_'+addition;row=table.loc[name]
            delta=np.array([row[f'fold_{i}']-baseline[f'fold_{i}'] for i in range(1,6)])
            bc=oof[family+'_baseline']==y[dev];nc=oof[name]==y[dev]
            comparisons.append({'candidate':name,'baseline':family+'_baseline','cv_accuracy':row.cv_accuracy,
                                'delta_pp':100*delta.mean(),'folds_better':int((delta>1e-12).sum()),
                                'folds_worse':int((delta < -1e-12).sum()),'corrected_errors':int((~bc & nc).sum()),
                                'introduced_errors':int((bc & ~nc).sum()),
                                **{f'delta_fold_{i+1}_pp':100*v for i,v in enumerate(delta)}})
    paired=pd.DataFrame(comparisons)
    paired.to_csv(folder/'paired_comparison.csv',index=False)
    (folder/'candidates.json').write_text(json.dumps(configs,indent=2)+'\n')
    print('\n'+pd.DataFrame(results).to_string(index=False),flush=True)
    print('\n'+paired.to_string(index=False),flush=True)
    print('\nFinished:',folder,flush=True)
    return folder

if __name__ == '__main__':
    run()
