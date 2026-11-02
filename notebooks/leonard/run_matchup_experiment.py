"""Controlled matchup additions to the fold-fitted RFE300 logistic baseline.

Same five grouped folds, C=.01, no extra tuning or holdout scoring. Train the
CV-preferred candidate on all labels and generate a separate checked submission.
"""
from run_feature_selection_experiment import *
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from matchup_features import MatchupTransformer,summarize_battle,ACTIVE_KEYS,TEAM_KEYS

SELECTION_RUN=ROOT/'outputs/leonard/20261102_144359_081808_feature_selection'
COMPACT_RUN=ROOT/'outputs/leonard/20261102_145002_205922_rfe300_compact'
CANDIDATES=[('rfe300_baseline',None),('rfe300_active_matchups','active'),
            ('rfe300_team_coverage','team'),('rfe300_all_matchups','all')]


def make_matchup_model(columns,group,chart_path):
    baseline=make_selection_model(columns,'rfe',300)
    if group is None:return baseline
    baseline_transform=Pipeline(baseline.steps[:-1])
    matchup_transform=Pipeline([('raw',MatchupTransformer(chart_path,group)),
                               ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                               ('constant_filter',VarianceThreshold()),('scaler',StandardScaler())])
    return Pipeline([('features',FeatureUnion([('baseline',baseline_transform),('matchups',matchup_transform)])),
                     ('model',LogisticRegression(C=.01,max_iter=4000,solver='lbfgs',random_state=42))])


def feature_names(model,columns,group):
    if group is None:return semantic_feature_names(model,columns)
    branches=dict(model.named_steps['features'].transformer_list)
    old=semantic_feature_names(branches['baseline'],columns)
    branch=branches['matchups']
    new=branch.named_steps['raw'].get_feature_names_out()[branch.named_steps['constant_filter'].get_support()]
    return [*old,*['matchup__'+key for key in new]]


def run_matchups():
    source=json.loads((SELECTION_RUN/'experiment.json').read_text());columns=source['baseline_columns']
    assert digest(data_dir()/'train.jsonl')==source['train_sha256']
    assert digest(PRIOR/'train_features.csv')==source['numeric_feature_sha256']
    assert digest(INITIAL/'identity_features.jsonl')==source['identity_feature_sha256']
    assert digest(SELECTION_RUN/'splits.csv')==source['splits_sha256']
    chart_path=str(data_dir()/'type_chart_gen2.csv')
    assignments=pd.read_csv(SELECTION_RUN/'splits.csv')
    identities=[json.loads(line) for line in (INITIAL/'identity_features.jsonl').open()]
    numeric=pd.read_csv(PRIOR/'train_features.csv')[columns]
    ids=[];labels=[];observations=[]
    for b in iter_battles(data_dir()/'train.jsonl'):
        ids.append(b['battle_id']);labels.append(int(b['player_won']));observations.append(summarize_battle(b))
    assert ids==assignments.battle_id.tolist()
    X_all=pd.concat([numeric,pd.DataFrame(identities)],axis=1);X_all['battle']=observations
    y_all=np.asarray(labels);dev=np.flatnonzero(assignments.partition.eq('development').to_numpy())
    excluded=np.flatnonzero(~assignments.partition.eq('development').to_numpy())
    assert len(dev)==7996 and len(excluded)==2004
    assert set(assignments.iloc[dev].observation_sha256).isdisjoint(set(assignments.iloc[excluded].observation_sha256))
    assert assignments.iloc[dev].groupby('observation_sha256').validation_fold.nunique().max()==1
    X=X_all.iloc[dev];y=y_all[dev];fold_ids=assignments.iloc[dev].validation_fold.to_numpy()
    reference=pd.read_csv(SELECTION_RUN/'oof_rfe_300.csv')
    assert reference.battle_id.tolist()==np.asarray(ids)[dev].tolist()
    np.testing.assert_array_equal(reference.actual.to_numpy(),y)
    previous_selection=pd.read_csv(SELECTION_RUN/'selected_features_by_fold.csv')
    previous_selection=previous_selection[previous_selection.candidate.eq('rfe_300')]
    folder=ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_matchups')
    folder.mkdir(parents=True);print('Artifacts:',folder,flush=True)
    shutil.copy2(SELECTION_RUN/'splits.csv',folder/'splits.csv');shutil.copy2(chart_path,folder/'type_chart_gen2.csv')
    from importlib.metadata import version
    versions={p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl','python-dateutil')}
    manifest={'design':'Four fixed LR comparisons: fold-fitted RFE300, +10 active features, +16 team features, +all26. No model/parameter search.',
              'candidates':CANDIDATES,'classifier':'LogisticRegression','C':.01,'max_iter':4000,'solver':'lbfgs','seed':42,
              'baseline_selection':'Same LR RFE300 fitted within each training fold, before appending matchup features.',
              'matchup_preprocessing':'Training-fold median imputation, constant removal, standard scaling.',
              'species_types':'Training-fold own/lead metadata only; per-battle known metadata overlay at transform time.',
              'active_features':ACTIVE_KEYS,'team_features':TEAM_KEYS,'columns':columns,'versions':versions,
              'development_rows':len(dev),'excluded_previous_holdout_and_duplicates':len(excluded),
              'source_selection_run':str(SELECTION_RUN.relative_to(ROOT)),
              'train_sha256':source['train_sha256'],'splits_sha256':source['splits_sha256'],
              'numeric_feature_sha256':source['numeric_feature_sha256'],'identity_feature_sha256':source['identity_feature_sha256'],
              'type_chart_sha256':digest(chart_path),
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'holdout_evaluated':False,'test_labels_used':False,
              'limitations':'Exploratory repeated CV, exact duplicate grouping only. Own-type potential is not actual moveset. Only revealed enemies included; last HP/moves can be stale. Hidden Power and zero-power damaging moves excluded. No exact damage, abilities or item calculations.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (folder/'requirements_classical.txt').write_text(''.join(f'{n}=={v}\n' for n,v in versions.items()))
    for name in ('run_matchup_experiment.py','matchup_features.py','run_feature_selection_experiment.py',
                 'identity_features.py','run_identity_scaling_followup.py','run_classical_identity_experiment.py'):
        shutil.copy2(Path(__file__).with_name(name),folder/name)
    results=[];oofs={};selected_rows=[];coverage=[]
    with threadpool_limits(limits=2):
        for name,group in CANDIDATES:
            pred=np.full(len(dev),-1,dtype=int);prob=np.full(len(dev),np.nan)
            scores=[];counts=[];started=time.perf_counter()
            for fold in range(5):
                train=np.flatnonzero(fold_ids!=fold);val=np.flatnonzero(fold_ids==fold)
                model=make_matchup_model(columns,group,chart_path)
                model.fit(X.iloc[train],y[train])
                pred[val]=model.predict(X.iloc[val]);prob[val]=model.predict_proba(X.iloc[val])[:,1]
                names=feature_names(model,columns,group);baseline_names=[n for n in names if not n.startswith('matchup__')]
                assert len(baseline_names)==300
                expected=previous_selection[previous_selection.fold.eq(fold)].feature.tolist()
                assert set(baseline_names)==set(expected),'Baseline selection changed'
                selected_rows.extend({'candidate':name,'fold':fold,'feature':f} for f in names)
                if group is not None:
                    raw=dict(model.named_steps['features'].transformer_list)['matchups'].named_steps['raw']
                    values=raw.transform(X.iloc[val])
                    coverage.append({'candidate':name,'fold':fold,'training_species_types':len(raw.species_types_),
                                     'validation_rows':len(val),'missing_feature_cells':int(values.isna().sum().sum()),
                                     'missing_by_feature':values.isna().sum().to_dict()})
                scores.append(float((pred[val]==y[val]).mean()));counts.append(len(names))
                print(f'{name} fold {fold+1}: {scores[-1]:.6f}; {len(names)} features',flush=True)
            assert np.isfinite(prob).all() and set(pred)<={0,1}
            if group is None:
                np.testing.assert_array_equal(pred,reference.predicted.to_numpy())
                np.testing.assert_allclose(prob,reference.probability.to_numpy(),rtol=1e-10,atol=1e-12)
            oofs[name]=pred
            pd.DataFrame({'battle_id':np.asarray(ids)[dev],'actual':y,'fold':fold_ids,'predicted':pred,'probability':prob}).to_csv(folder/f'oof_{name}.csv',index=False)
            results.append({'candidate':name,'group':group or 'baseline','mean_selected_features':float(np.mean(counts)),
                            'cv_accuracy':float(np.mean(scores)),'oof_accuracy':float((pred==y).mean()),
                            'fold_std':float(np.std(scores)),'fit_predict_seconds':time.perf_counter()-started,
                            **{f'fold_{i+1}':v for i,v in enumerate(scores)}})
            pd.DataFrame(results).to_csv(folder/'cv_results.csv',index=False)
            pd.DataFrame(selected_rows).to_csv(folder/'selected_features_by_fold.csv',index=False)
        table=pd.DataFrame(results).set_index('candidate');paired=[]
        for name,_ in CANDIDATES[1:]:
            delta=np.array([table.loc[name,f'fold_{i}']-table.loc['rfe300_baseline',f'fold_{i}'] for i in range(1,6)])
            bc=oofs['rfe300_baseline']==y;nc=oofs[name]==y
            paired.append({'candidate':name,'delta_pp':100*delta.mean(),'folds_better':int((delta>1e-12).sum()),
                           'folds_worse':int((delta < -1e-12).sum()),'corrected_errors':int((nc & ~bc).sum()),
                           'introduced_errors':int((~nc & bc).sum()),**{f'delta_fold_{i+1}_pp':100*v for i,v in enumerate(delta)}})
        pd.DataFrame(paired).to_csv(folder/'paired_comparison.csv',index=False)
        (folder/'coverage_audit.json').write_text(json.dumps(coverage,indent=2)+'\n')
        best=table.cv_accuracy.idxmax();choice=next(c for c in CANDIDATES if c[0]==best)
        (folder/'selection.json').write_text(json.dumps({'candidate':choice,'rule':'highest mean CV accuracy, baseline first on ties'},indent=2)+'\n')
        print('Frozen selection:',best,flush=True)
        if choice[1] is None:
            for name in ('model.joblib','submission.csv','test_probabilities.csv'):shutil.copy2(COMPACT_RUN/name,folder/name)
            shutil.copy2(COMPACT_RUN/'selected_features.csv',folder/'final_selected_features.csv')
            final_count=300
        else:
            model=make_matchup_model(columns,choice[1],chart_path);model.fit(X_all,y_all)
            names=feature_names(model,columns,choice[1]);final_count=len(names)
            pd.DataFrame({'feature':names,'coefficient':model.named_steps['model'].coef_[0]}).to_csv(folder/'final_selected_features.csv',index=False)
            test_ids=[];test_rows=[]
            for b in iter_battles(data_dir()/'test.jsonl'):
                numbers=feature_row(b);test_ids.append(b['battle_id'])
                test_rows.append({**{k:numbers[k] for k in columns},**identity_features(b),'battle':summarize_battle(b)})
            assert len(test_ids)==len(set(test_ids))
            test=pd.DataFrame(test_rows);pred=model.predict(test);prob=model.predict_proba(test)[:,1]
            sample=pd.read_csv(data_dir()/'sample_submission.csv')
            assert len(sample)==len(test_ids) and set(sample.battle_id)==set(test_ids)
            sub=sample.copy();sub['player_won']=sub.battle_id.map(pd.Series(pred,index=test_ids)).astype(int)
            sub.to_csv(folder/'submission.csv',index=False)
            pd.DataFrame({'battle_id':test_ids,'probability':prob}).to_csv(folder/'test_probabilities.csv',index=False)
            joblib.dump(model,folder/'model.joblib')
            loaded=joblib.load(folder/'model.joblib')
            np.testing.assert_array_equal(loaded.predict(test),pred)
            np.testing.assert_allclose(loaded.predict_proba(test)[:,1],prob,rtol=1e-12,atol=1e-12)
    sub=pd.read_csv(folder/'submission.csv');sample=pd.read_csv(data_dir()/'sample_submission.csv')
    assert list(sub)==['battle_id','player_won'] and sub.battle_id.equals(sample.battle_id) and sub.battle_id.is_unique
    assert sub.notna().all().all() and sub.player_won.isin([0,1]).all()
    report={'candidate':best,'cv_accuracy':float(table.loc[best,'cv_accuracy']),
            'baseline_cv_accuracy':float(table.loc['rfe300_baseline','cv_accuracy']),
            'delta_pp':100*float(table.loc[best,'cv_accuracy']-table.loc['rfe300_baseline','cv_accuracy']),
            'features':final_count,'fit_rows':len(y_all),'test_rows':len(sub),'outer_cv_fits':20,
            'baseline_reproduced':True,'all_baseline_subsets_unchanged':True,'submission_checks_passed':True,
            'serialized_winner_checks_passed':True,'holdout_evaluated':False,'test_labels_used':False,
            'submission_sha256':digest(folder/'submission.csv'),'model_sha256':digest(folder/'model.joblib'),
            'test_sha256':digest(data_dir()/'test.jsonl'),'sample_sha256':digest(data_dir()/'sample_submission.csv'),
            'submission':str(folder/'submission.csv')}
    (folder/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True);print(json.dumps(report,indent=2),flush=True)
    print('Finished:',folder,flush=True);return folder

if __name__=='__main__':run_matchups()
