"""Bounded follow-up: scale only baseline numeric features, keep identities raw.

Motivated by overfitting in the first four classical-model comparisons. Three fixed
LR regularization strengths; select among these and the four previous candidates.
No holdout scoring; report exploratory CV and reproduce the final Kaggle CSV.
"""
from run_classical_identity_experiment import *
INITIAL=ROOT/'outputs/leonard/20261102_123710_282289_classical_identity'


def make_followup(columns,C):
    # Standardize only baseline continuous features. Do not inflate rare identity
    # indicators or move rates by inverse standard deviation.
    return Pipeline([('encoding',IdentityEncoder(columns)),
                     ('scaling',ColumnTransformer([('base',StandardScaler(),list(range(len(columns))))],remainder='passthrough')),
                     ('constant_filter',VarianceThreshold()),
                     ('model',LogisticRegression(C=C,max_iter=4000,solver='lbfgs',random_state=42))])


def run_followup():
    initial=json.loads((INITIAL/'experiment.json').read_text());columns=initial['numeric_columns']
    assert digest(data_dir()/'train.jsonl')==initial['train_sha256']
    assert digest(PRIOR/'train_features.csv')==initial['numeric_feature_cache_sha256']
    assert digest(INITIAL/'identity_features.jsonl')==initial['identity_feature_sha256']
    assert digest(INITIAL/'splits.csv')==initial['splits_sha256']
    assignments=pd.read_csv(INITIAL/'splits.csv');dev=np.flatnonzero(assignments.partition.eq('development').to_numpy())
    identity_rows=[json.loads(line) for line in (INITIAL/'identity_features.jsonl').open()]
    X_all=pd.concat([pd.read_csv(PRIOR/'train_features.csv')[columns],pd.DataFrame(identity_rows)],axis=1)
    ids=[];labels=[]
    for b in iter_battles(data_dir()/'train.jsonl'):ids.append(b['battle_id']);labels.append(int(b['player_won']))
    assert ids==assignments.battle_id.tolist()
    y_all=np.asarray(labels);X=X_all.iloc[dev];y=y_all[dev]
    fold_ids=assignments.iloc[dev].validation_fold.to_numpy()
    folder=ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_classical_identity_followup')
    folder.mkdir(parents=True);print('Artifacts:',folder,flush=True)
    shutil.copy2(INITIAL/'splits.csv',folder/'splits.csv')
    for path in [Path(__file__),Path(__file__).with_name('identity_features.py'),Path(__file__).with_name('run_classical_identity_experiment.py')]:shutil.copy2(path,folder/path.name)
    settings=[('lr_identity_base_scaled_C001',.01),('lr_identity_base_scaled_C01',.1),('lr_identity_base_scaled_C1',1.)]
    manifest={'design':'Adaptive but bounded follow-up to the four classical comparisons: three prespecified C values with only baseline numeric scaling.',
              'initial_run':str(INITIAL.relative_to(ROOT)),'data_and_splits':initial,
              'settings':settings,'holdout_evaluated':False,
              'limitations':'Repeated exploratory CV, including a follow-up after poor identity results. This is not independent confirmation. No claim of a higher Kaggle score.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results=pd.read_csv(INITIAL/'cv_results.csv').to_dict('records')
    for c in results:shutil.copy2(INITIAL/f'oof_{c["candidate"]}.csv',folder/f'oof_{c["candidate"]}.csv')
    with threadpool_limits(limits=2):
        for name,C in settings:
            pred=np.full(len(dev),-1,dtype=int);prob=np.full(len(dev),np.nan);scores=[];counts=[];started=time.perf_counter()
            for fold in range(5):
                train=np.flatnonzero(fold_ids!=fold);val=np.flatnonzero(fold_ids==fold)
                model=make_followup(columns,C)
                model.fit(X.iloc[train],y[train]);pred[val]=model.predict(X.iloc[val]);prob[val]=model.predict_proba(X.iloc[val])[:,1]
                scores.append(float((pred[val]==y[val]).mean()));counts.append(int(model.named_steps['constant_filter'].get_support().sum()))
                print(f'{name} fold {fold+1}: {scores[-1]:.6f}',flush=True)
            assert np.isfinite(prob).all() and set(pred)<={0,1}
            pd.DataFrame({'battle_id':np.asarray(ids)[dev],'actual':y,'fold':fold_ids,'predicted':pred,'probability':prob}).to_csv(folder/f'oof_{name}.csv',index=False)
            results.append({'candidate':name,'model':'lr','identity':True,'C':C,'cv_accuracy':float(np.mean(scores)),
                            'oof_accuracy':float((pred==y).mean()),'fold_std':float(np.std(scores)),
                            'mean_selected_features':float(np.mean(counts)), 'fit_predict_seconds':time.perf_counter()-started,
                            **{f'fold_{i+1}':v for i,v in enumerate(scores)}})
            pd.DataFrame(results).to_csv(folder/'cv_results.csv',index=False)
        table=pd.DataFrame(results).set_index('candidate');best=table.cv_accuracy.idxmax()
        (folder/'selection.json').write_text(json.dumps({'candidate':best,'rule':'maximum mean CV accuracy among 4 initial and 3 follow-up simple models; stable order on ties'},indent=2)+'\n')
        print('Frozen selection:',best,flush=True)
        if best in dict(settings):
            final=make_followup(columns,dict(settings)[best]);final.fit(X_all,y_all)
            test_ids=[];test_rows=[]
            for b in iter_battles(data_dir()/'test.jsonl'):
                test_ids.append(b['battle_id']);numbers=feature_row(b)
                test_rows.append({**{k:numbers[k] for k in columns},**identity_features(b)})
            assert len(test_ids)==len(set(test_ids))
            test=pd.DataFrame(test_rows);pred=final.predict(test);prob=final.predict_proba(test)[:,1]
            sample=pd.read_csv(data_dir()/'sample_submission.csv')
            assert len(sample)==len(test_ids) and set(sample.battle_id)==set(test_ids)
            submission=sample.copy();submission['player_won']=submission.battle_id.map(pd.Series(pred,index=test_ids)).astype(int)
            submission.to_csv(folder/'submission.csv',index=False)
            pd.DataFrame({'battle_id':test_ids,'probability':prob}).to_csv(folder/'test_probabilities.csv',index=False)
            joblib.dump(final,folder/'model.joblib')
        else:
            assert best==json.loads((INITIAL/'selection.json').read_text())['candidate']['name']
            for name in ('submission.csv','model.joblib','test_probabilities.csv'):shutil.copy2(INITIAL/name,folder/name)
        sample=pd.read_csv(data_dir()/'sample_submission.csv');submission=pd.read_csv(folder/'submission.csv')
        assert list(submission)==['battle_id','player_won'] and submission.battle_id.equals(sample.battle_id)
        assert submission.battle_id.is_unique and submission.notna().all().all() and submission.player_won.isin([0,1]).all()
        report={'candidate':best,'cv_accuracy':float(table.loc[best,'cv_accuracy']),
                'historical_boosting_cv_accuracy':json.loads((INITIAL/'run_summary.json').read_text())['historical_boosting_cv_accuracy'],
                'candidates':7,'cv_fits_total_including_initial':35,'development_rows':len(dev),'fit_rows':len(y_all),
                'test_rows':len(submission),'holdout_evaluated':False,'test_labels_used':False,
                'prediction_counts':submission.player_won.value_counts().to_dict(),'submission_checks_passed':True,
                'submission_sha256':digest(folder/'submission.csv'),'model_sha256':digest(folder/'model.joblib'),
                'test_sha256':digest(data_dir()/'test.jsonl'),'sample_sha256':digest(data_dir()/'sample_submission.csv'),
                'submission':str(folder/'submission.csv')}
        (folder/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(results).to_string(index=False),flush=True);print(json.dumps(report,indent=2),flush=True)
    print('Finished:',folder,flush=True);return folder

if __name__=='__main__':run_followup()
