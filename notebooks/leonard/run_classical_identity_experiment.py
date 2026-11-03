"""Logistic regression / random forest with and without species/move identity.

.venv/bin/python notebooks/leonard/run_classical_identity_experiment.py
Uses fixed previous grouped folds, then fits the CV-selected simple model on all
labels and generates a validated submission. No holdout evaluation or upload.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent),str(ROOT/'notebooks/aimilia')]
import json
import shutil
from datetime import datetime
import platform
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from identity_features import identity_features,IdentityEncoder
from run_progressive_experiments import digest,feature_row
from pokemon_challenge.data import iter_battles
from pokemon_challenge.paths import data_dir

PRIOR=ROOT/'outputs/leonard/20261102_121822_824155_progressive_features'
SETTINGS={'lr':{'C':0.1,'max_iter':4000,'solver':'lbfgs','random_state':42},
          'rf':{'n_estimators':300,'min_samples_leaf':2,'max_features':'sqrt',
                'max_depth':None,'n_jobs':2,'random_state':42}}


def make_model(family,identity,columns):
    encoding=IdentityEncoder(columns) if identity else ColumnTransformer([('numeric','passthrough',columns)],remainder='drop')
    model=LogisticRegression(**SETTINGS['lr']) if family=='lr' else RandomForestClassifier(**SETTINGS['rf'])
    return Pipeline([('encoding',encoding),('constant_filter',VarianceThreshold()),
                     ('scaler',StandardScaler() if family=='lr' else 'passthrough'),('model',model)])


def run():
    prior=json.loads((PRIOR/'experiment.json').read_text())
    assert digest(PRIOR/'train_features.csv')==prior['feature_sha256']
    assert digest(PRIOR/'splits.csv')==prior['splits_sha256']
    assert digest(data_dir()/'train.jsonl')==prior['train_sha256']
    configurations=json.loads((PRIOR/'candidates.json').read_text())
    columns=next(c['columns'] for c in configurations if c['candidate']=='hgb_boost_threat')
    assert len(columns)==289
    assignments=pd.read_csv(PRIOR/'splits.csv')
    numeric=pd.read_csv(PRIOR/'train_features.csv')[columns]
    assert len(numeric)==len(assignments) and np.isfinite(numeric.to_numpy()).all()
    # Predeclare the four comparisons and parameters before exposing CV results.
    candidates=[{'name':f'{family}_{"identity" if identity else "numeric"}', 'family':family,'identity':identity}
                for family in ('lr','rf') for identity in (False,True)]
    folder=ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_classical_identity')
    folder.mkdir(parents=True)
    print('Artifacts:',folder,flush=True)
    shutil.copy2(PRIOR/'splits.csv',folder/'splits.csv')
    identity_rows=[];labels=[];ids=[]
    with (folder/'identity_features.jsonl').open('w') as out:
        for b in iter_battles(data_dir()/'train.jsonl'):
            ids.append(b['battle_id']);labels.append(int(b['player_won']))
            row=identity_features(b);identity_rows.append(row)
            out.write(json.dumps(row,sort_keys=True)+'\n')
    assert ids==assignments.battle_id.tolist() and len(set(ids))==len(ids)
    y_all=np.asarray(labels)
    X_all=pd.concat([numeric,pd.DataFrame(identity_rows)],axis=1)
    dev=np.flatnonzero(assignments.partition.eq('development').to_numpy())
    excluded=np.flatnonzero(~assignments.partition.eq('development').to_numpy())
    assert len(dev)==7996 and len(excluded)==2004
    assert set(assignments.iloc[dev].observation_sha256).isdisjoint(set(assignments.iloc[excluded].observation_sha256))
    assert assignments.iloc[dev].groupby('observation_sha256').validation_fold.nunique().max()==1
    fold_ids=assignments.iloc[dev].validation_fold.to_numpy();X=X_all.iloc[dev];y=y_all[dev]
    previous=pd.read_csv(PRIOR/'oof_hgb_boost_threat.csv')
    np.testing.assert_array_equal(previous.actual.to_numpy(),y)
    manifest={'design':'Four fixed simple-model comparisons: logistic regression/random forest x numeric/identity. No boosting, ensemble or parameter search.',
              'prior_run':str(PRIOR.relative_to(ROOT)),'settings':SETTINGS,'candidates':candidates,'numeric_columns':columns,
              'seed':42,'cv':'same five StratifiedGroupKFold partitions as prior run',
              'development_rows':len(dev),'excluded_prior_holdout_and_duplicates':len(excluded),
              'splits_sha256':digest(folder/'splits.csv'),'train_sha256':prior['train_sha256'],
              'numeric_feature_cache_sha256':prior['feature_sha256'],'identity_feature_sha256':digest(folder/'identity_features.jsonl'),
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'holdout_evaluated':False,'limitations':'Repeated development CV is exploratory. Exact duplicates grouped; related nonidentical battles may remain. Species HP may be stale. Unknown validation/test identities are ignored by fold vocabulary.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for path in [Path(__file__),Path(__file__).with_name('identity_features.py')]:shutil.copy2(path,folder/path.name)
    for name in ('progressive_features.py','ablation_extensions.py','run_progressive_experiments.py',
                 'features_static.py','features_timeline.py','features_effects.py','features_roster.py'):
        shutil.copy2(PRIOR/name,folder/name)
    results=[];predictions={}
    with threadpool_limits(limits=2):
        for c in candidates:
            name=c['name'];pred=np.full(len(dev),-1,dtype=int);probs=np.full(len(dev),np.nan)
            scores=[];counts=[];started=time.perf_counter()
            for fold in range(5):
                val=np.flatnonzero(fold_ids==fold);train=np.flatnonzero(fold_ids!=fold)
                pipeline=make_model(c['family'],c['identity'],columns)
                pipeline.fit(X.iloc[train],y[train])
                pred[val]=pipeline.predict(X.iloc[val]);probs[val]=pipeline.predict_proba(X.iloc[val])[:,1]
                scores.append(float((pred[val]==y[val]).mean()))
                counts.append(int(pipeline.named_steps['constant_filter'].get_support().sum()))
                print(f'{name} fold {fold+1}: {scores[-1]:.6f}; {counts[-1]} features',flush=True)
            assert set(pred)<={0,1} and np.isfinite(probs).all()
            predictions[name]=pred
            pd.DataFrame({'battle_id':np.asarray(ids)[dev],'actual':y,'fold':fold_ids,'predicted':pred,'probability':probs}).to_csv(folder/f'oof_{name}.csv',index=False)
            results.append({'candidate':name,'model':c['family'],'identity':c['identity'],'cv_accuracy':float(np.mean(scores)),
                            'oof_accuracy':float((pred==y).mean()),'fold_std':float(np.std(scores)),
                            'mean_selected_features':float(np.mean(counts)),'fit_predict_seconds':time.perf_counter()-started,
                            **{f'fold_{i+1}':v for i,v in enumerate(scores)}})
            pd.DataFrame(results).to_csv(folder/'cv_results.csv',index=False)
        table=pd.DataFrame(results).set_index('candidate');paired=[]
        for family in ('lr','rf'):
            baseline=family+'_numeric';name=family+'_identity'
            delta=np.array([table.loc[name,f'fold_{i}']-table.loc[baseline,f'fold_{i}'] for i in range(1,6)])
            bc=predictions[baseline]==y;nc=predictions[name]==y
            paired.append({'candidate':name,'baseline':baseline,'delta_pp':float(100*delta.mean()),
                           'folds_better':int((delta>1e-12).sum()),'folds_worse':int((delta < -1e-12).sum()),
                           'corrected_errors':int((nc & ~bc).sum()),'introduced_errors':int((~nc & bc).sum()),
                           **{f'delta_fold_{i+1}_pp':100*v for i,v in enumerate(delta)}})
        pd.DataFrame(paired).to_csv(folder/'paired_comparison.csv',index=False)
        best_name=table.cv_accuracy.idxmax();chosen=next(c for c in candidates if c['name']==best_name)
        # Freeze selection before extracting official test observations or fitting all labels.
        (folder/'selection.json').write_text(json.dumps({'candidate':chosen,'cv_accuracy':float(table.loc[best_name,'cv_accuracy']),
                                                      'rule':'highest mean CV accuracy among four prespecified simple models; stable order on ties'},indent=2)+'\n')
        print('Frozen CV winner:',best_name,flush=True)
        test_rows=[];test_ids=[]
        for b in iter_battles(data_dir()/'test.jsonl'):
            test_ids.append(b['battle_id'])
            numeric_row=feature_row(b)
            test_rows.append({**{k:numeric_row[k] for k in columns},**identity_features(b)})
        assert len(test_ids)==len(set(test_ids))
        test=pd.DataFrame(test_rows)
        final=make_model(chosen['family'],chosen['identity'],columns)
        final.fit(X_all,y_all)
        test_probability=final.predict_proba(test)[:,1];test_prediction=final.predict(test)
        sample=pd.read_csv(data_dir()/'sample_submission.csv')
        assert list(sample)==['battle_id','player_won'] and sample.battle_id.is_unique and sample.battle_id.notna().all()
        assert len(sample)==len(test_ids) and set(sample.battle_id)==set(test_ids)
        mapping=pd.Series(test_prediction,index=test_ids)
        submission=sample.copy();submission['player_won']=submission.battle_id.map(mapping).astype(int)
        assert submission.battle_id.equals(sample.battle_id) and submission.player_won.isin([0,1]).all()
        assert submission.notna().all().all()
        submission.to_csv(folder/'submission.csv',index=False)
        pd.testing.assert_frame_equal(submission,pd.read_csv(folder/'submission.csv'))
        pd.DataFrame({'battle_id':test_ids,'probability':test_probability}).to_csv(folder/'test_probabilities.csv',index=False)
        joblib.dump(final,folder/'model.joblib')
    report={'candidate':best_name,'cv_accuracy':float(table.loc[best_name,'cv_accuracy']),
            'historical_boosting_cv_accuracy':float((previous.actual==previous.predicted).groupby(previous.fold).mean().mean()),
            'development_rows':len(dev),'fit_rows':len(y_all),'test_rows':len(test_ids),
            'prediction_counts':submission.player_won.value_counts().to_dict(),
            'holdout_evaluated':False,'test_labels_used':False,'cv_fits':20,
            'submission_checks_passed':True,'submission_sha256':digest(folder/'submission.csv'),
            'test_sha256':digest(data_dir()/'test.jsonl'),'sample_sha256':digest(data_dir()/'sample_submission.csv'),
            'model_sha256':digest(folder/'model.joblib'),
            'submission':str(folder/'submission.csv')}
    (folder/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print('Finished:',folder,flush=True)
    return folder

if __name__=='__main__':run()
