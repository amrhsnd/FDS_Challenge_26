"""Fold-fitted SelectKBest and RFE on the best simple logistic regression.

Run: .venv/bin/python notebooks/leonard/run_feature_selection_experiment.py
Optional local runtime: PYTHONPATH=/tmp/pokemon_identity_runtime
No holdout re-evaluation, boosting models, global feature ranking or Kaggle upload.
"""
from run_identity_scaling_followup import *
from sklearn.feature_selection import SelectKBest, RFE, f_classif

BASELINE_RUN=ROOT/'outputs/leonard/20261102_124639_898979_classical_identity_followup'
# Fixed small comparison; model C/scaling stay unchanged to isolate selection.
CANDIDATES=[('all_features','none',None),('kbest_100','kbest',100),
            ('kbest_300','kbest',300),('kbest_1000','kbest',1000),
            ('rfe_300','rfe',300),('rfe_1000','rfe',1000)]


def make_selection_model(columns,method,k=None):
    base=make_followup(columns,.01)
    if method=='none':selector='passthrough'
    elif method=='kbest':selector=SelectKBest(score_func=f_classif,k=k)
    elif method=='rfe':
        selector=RFE(estimator=LogisticRegression(C=.01,max_iter=4000,solver='lbfgs',random_state=42),
                     n_features_to_select=k,step=.25)
    else:raise ValueError('Unknown feature selection method')
    return Pipeline([*base.steps[:-1],('feature_selection',selector),base.steps[-1]])


def semantic_feature_names(model,columns):
    """Recover actual feature names after encoding/scaling/constant removal."""
    names=np.asarray([*columns,*model.named_steps['encoding'].vectorizer_.feature_names_],dtype=object)
    names=names[model.named_steps['constant_filter'].get_support()]
    selector=model.named_steps['feature_selection']
    if selector!='passthrough':names=names[selector.get_support()]
    return list(names)


def run_selection():
    baseline_manifest=json.loads((BASELINE_RUN/'experiment.json').read_text())
    initial=baseline_manifest['data_and_splits'];columns=initial['numeric_columns']
    assert digest(data_dir()/'train.jsonl')==initial['train_sha256']
    assert digest(PRIOR/'train_features.csv')==initial['numeric_feature_cache_sha256']
    assert digest(INITIAL/'identity_features.jsonl')==initial['identity_feature_sha256']
    assert digest(BASELINE_RUN/'splits.csv')==initial['splits_sha256']
    assignments=pd.read_csv(BASELINE_RUN/'splits.csv')
    identities=[json.loads(s) for s in (INITIAL/'identity_features.jsonl').open()]
    X_all=pd.concat([pd.read_csv(PRIOR/'train_features.csv')[columns],pd.DataFrame(identities)],axis=1)
    ids=[];labels=[]
    for battle in iter_battles(data_dir()/'train.jsonl'):ids.append(battle['battle_id']);labels.append(int(battle['player_won']))
    assert ids==assignments.battle_id.tolist()
    dev=np.flatnonzero(assignments.partition.eq('development').to_numpy())
    excluded=np.flatnonzero(~assignments.partition.eq('development').to_numpy())
    assert len(dev)==7996 and len(excluded)==2004
    assert set(assignments.iloc[dev].observation_sha256).isdisjoint(set(assignments.iloc[excluded].observation_sha256))
    assert assignments.iloc[dev].groupby('observation_sha256').validation_fold.nunique().max()==1
    X=X_all.iloc[dev];y_all=np.asarray(labels);y=y_all[dev]
    fold_ids=assignments.iloc[dev].validation_fold.to_numpy()
    old=pd.read_csv(BASELINE_RUN/'oof_lr_identity_base_scaled_C001.csv')
    assert old.battle_id.tolist()==np.asarray(ids)[dev].tolist()
    np.testing.assert_array_equal(y,old.actual.to_numpy())
    folder=ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_feature_selection')
    folder.mkdir(parents=True);print('Artifacts:',folder,flush=True)
    shutil.copy2(BASELINE_RUN/'splits.csv',folder/'splits.csv')
    from importlib.metadata import version
    versions={p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','python-dateutil')}
    manifest={'design':'6 prespecified candidates: no selection, F-test KBest 100/300/1000, LR RFE 300/1000. CV selection is fold-fitted.',
              'candidates':CANDIDATES,'model':{'class':'LogisticRegression','C':.01,'max_iter':4000,'solver':'lbfgs','random_state':42},
              'rfe_step':.25,'rfe_estimator':'same logistic regression; coefficients depend on the retained mixed scaling',
              'baseline_columns':columns,'baseline_run':str(BASELINE_RUN.relative_to(ROOT)),
              'initial_data_run':str(INITIAL.relative_to(ROOT)),'numeric_feature_run':str(PRIOR.relative_to(ROOT)),
              'development_rows':len(dev),'excluded_rows':len(excluded),'versions':versions,
              'splits_sha256':digest(folder/'splits.csv'),'train_sha256':initial['train_sha256'],
              'numeric_feature_sha256':initial['numeric_feature_cache_sha256'],'identity_feature_sha256':initial['identity_feature_sha256'],
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'holdout_evaluated':False,'test_labels_used':False,
              'limitations':'Repeated exploratory CV, not independent performance confirmation. KBest ignores joint effects; RFE depends on coefficient scale and regularization. Correlated features may substitute for each other. Only exact duplicate groups are controlled.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'requirements_classical.txt').write_text(''.join(f'{n}=={v}\n' for n,v in versions.items()))
    for name in ('run_feature_selection_experiment.py','identity_features.py','run_identity_scaling_followup.py','run_classical_identity_experiment.py'):
        shutil.copy2(Path(__file__).with_name(name),folder/name)
    results=[];oofs={};raw_selection=[]
    with threadpool_limits(limits=2):
        for name,method,k in CANDIDATES:
            pred=np.full(len(dev),-1,dtype=int);prob=np.full(len(dev),np.nan)
            scores=[];counts=[];started=time.perf_counter()
            for fold in range(5):
                train=np.flatnonzero(fold_ids!=fold);val=np.flatnonzero(fold_ids==fold)
                model=make_selection_model(columns,method,k)
                model.fit(X.iloc[train],y[train])
                pred[val]=model.predict(X.iloc[val]);prob[val]=model.predict_proba(X.iloc[val])[:,1]
                chosen=semantic_feature_names(model,columns)
                if k is not None:assert len(chosen)==k
                raw_selection.extend({'candidate':name,'fold':fold,'feature':f} for f in chosen)
                counts.append(len(chosen));scores.append(float((pred[val]==y[val]).mean()))
                print(f'{name} fold {fold+1}: {scores[-1]:.6f}; {len(chosen)} features',flush=True)
            assert np.isfinite(prob).all() and set(pred)<={0,1}
            if method=='none':
                np.testing.assert_array_equal(pred,old.predicted.to_numpy())
                np.testing.assert_allclose(prob,old.probability.to_numpy(),rtol=1e-10,atol=1e-12)
            oofs[name]=pred
            pd.DataFrame({'battle_id':np.asarray(ids)[dev],'actual':y,'fold':fold_ids,'predicted':pred,'probability':prob}).to_csv(folder/f'oof_{name}.csv',index=False)
            results.append({'candidate':name,'method':method,'requested_features':k,'mean_selected_features':float(np.mean(counts)),
                            'cv_accuracy':float(np.mean(scores)),'oof_accuracy':float((pred==y).mean()),
                            'fold_std':float(np.std(scores)),'fit_predict_seconds':time.perf_counter()-started,
                            **{f'fold_{i+1}':v for i,v in enumerate(scores)}})
            pd.DataFrame(results).to_csv(folder/'cv_results.csv',index=False)
            pd.DataFrame(raw_selection).to_csv(folder/'selected_features_by_fold.csv',index=False)
        table=pd.DataFrame(results).set_index('candidate');paired=[]
        for name,_,_ in CANDIDATES[1:]:
            delta=np.array([table.loc[name,f'fold_{i}']-table.loc['all_features',f'fold_{i}'] for i in range(1,6)])
            bc=oofs['all_features']==y;nc=oofs[name]==y
            paired.append({'candidate':name,'delta_pp':100*delta.mean(),'folds_better':int((delta>1e-12).sum()),
                           'folds_worse':int((delta < -1e-12).sum()),'corrected_errors':int((nc & ~bc).sum()),
                           'introduced_errors':int((~nc & bc).sum()),**{f'delta_fold_{i+1}_pp':100*v for i,v in enumerate(delta)}})
        pd.DataFrame(paired).to_csv(folder/'paired_comparison.csv',index=False)
        pd.DataFrame(raw_selection).groupby(['candidate','feature']).size().rename('selected_folds').reset_index().to_csv(folder/'selection_stability.csv',index=False)
        best=table.cv_accuracy.idxmax();choice=next(c for c in CANDIDATES if c[0]==best)
        (folder/'selection.json').write_text(json.dumps({'candidate':choice,'rule':'highest mean CV; baseline first on ties'},indent=2)+'\n')
        print('Frozen choice:',best,flush=True)
        if best=='all_features':
            # Preserve the exact already validated full-data baseline fit/CSV.
            for name in ('model.joblib','submission.csv','test_probabilities.csv'):shutil.copy2(BASELINE_RUN/name,folder/name)
            model=joblib.load(folder/'model.joblib')
            chosen_names=np.asarray([*columns,*model.named_steps['encoding'].vectorizer_.feature_names_],dtype=object)
            chosen_names=list(chosen_names[model.named_steps['constant_filter'].get_support()])
        else:
            model=make_selection_model(columns,choice[1],choice[2]);model.fit(X_all,y_all)
            chosen_names=semantic_feature_names(model,columns)
            test_ids=[];test_rows=[]
            for battle in iter_battles(data_dir()/'test.jsonl'):
                test_ids.append(battle['battle_id']);numbers=feature_row(battle)
                test_rows.append({**{k:numbers[k] for k in columns},**identity_features(battle)})
            assert len(test_ids)==len(set(test_ids))
            test=pd.DataFrame(test_rows);pred=model.predict(test);prob=model.predict_proba(test)[:,1]
            sample=pd.read_csv(data_dir()/'sample_submission.csv')
            assert len(sample)==len(test_ids) and set(sample.battle_id)==set(test_ids)
            sub=sample.copy();sub['player_won']=sub.battle_id.map(pd.Series(pred,index=test_ids)).astype(int)
            sub.to_csv(folder/'submission.csv',index=False)
            pd.DataFrame({'battle_id':test_ids,'probability':prob}).to_csv(folder/'test_probabilities.csv',index=False)
            joblib.dump(model,folder/'model.joblib')
        pd.DataFrame({'feature':chosen_names,'coefficient':model.named_steps['model'].coef_[0]}).to_csv(folder/'final_selected_features.csv',index=False)
    sub=pd.read_csv(folder/'submission.csv');sample=pd.read_csv(data_dir()/'sample_submission.csv')
    assert list(sub)==['battle_id','player_won'] and sub.battle_id.equals(sample.battle_id) and sub.battle_id.is_unique
    assert sub.notna().all().all() and sub.player_won.isin([0,1]).all()
    report={'candidate':best,'cv_accuracy':float(table.loc[best,'cv_accuracy']),'baseline_cv_accuracy':float(table.loc['all_features','cv_accuracy']),
            'final_features':len(chosen_names),'fit_rows':len(y_all),'test_rows':len(sub),'development_rows':len(dev),
            'submission_checks_passed':True,'baseline_predictions_reproduced':True,'holdout_evaluated':False,
            'test_labels_used':False,'submission_sha256':digest(folder/'submission.csv'),'model_sha256':digest(folder/'model.joblib'),
            'test_sha256':digest(data_dir()/'test.jsonl'),'sample_sha256':digest(data_dir()/'sample_submission.csv'),
            'submission':str(folder/'submission.csv')}
    (folder/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True);print(json.dumps(report,indent=2),flush=True)
    print('Finished:',folder,flush=True);return folder

if __name__=='__main__':run_selection()
