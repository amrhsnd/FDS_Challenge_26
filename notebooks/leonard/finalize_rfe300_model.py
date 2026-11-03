"""Finalize the user-preferred RFE300 logistic model and explain its 300 inputs.

The selection procedure's CV score comes from the previous fold-local experiment;
this all-data feature list is for the final model, never for preprocessing CV.
"""
from run_feature_selection_experiment import *

SELECTION_RUN=ROOT/'outputs/leonard/20261102_144359_081808_feature_selection'


def feature_category(name):
    if name.startswith('original__'):
        s=name.split('__',1)[1]
        if any(k in s for k in ('teamhp','alive','faints','hppct')):return 'HP, remaining team and faint counts'
        if 'boost' in s:return 'Boost summaries'
        if any(k in s for k in ('status_rate','slp_rate')):return 'Status conditions'
        if any(k in s for k in ('switches','n_species_seen','no_move')):return 'Switching, exposure and inactivity'
        if any(k in s for k in ('power','accuracy','move_rate')):return 'Attack summaries'
        return 'Starting stats, levels, types and items'
    if name.startswith('battle_state__'):return 'Recent battle state and roster status'
    if name.startswith('boost_threat__'):return 'Boost threat and HP/status interactions'
    if name.startswith('effects__'):return 'Effects such as trapped or perish'
    if '_team_species_hp::' in name or '_last_hp::' in name:return 'HP of particular species'
    if '_team_species::' in name:return 'Own team species membership'
    if '_seen_species::' in name or '_species_exposure::' in name:return 'Observed species and exposure'
    if '_active=' in name or name.startswith('p2_lead='):return 'Active or lead species identity'
    if '_last_move=' in name or '_most_used_move=' in name:return 'Last or most-used move identity'
    if '_move_recency::' in name or '_recent_move_rate::' in name:return 'Recency and recent frequency of particular moves'
    if '_move_type_rate::' in name:return 'Attack-type frequencies'
    if '_move_rate::' in name:return 'Frequency of particular moves'
    raise ValueError('Uncategorized feature: '+name)


def run_finalize():
    from collections import Counter
    source=json.loads((SELECTION_RUN/'experiment.json').read_text());columns=source['baseline_columns']
    assert digest(data_dir()/'train.jsonl')==source['train_sha256']
    assert digest(PRIOR/'train_features.csv')==source['numeric_feature_sha256']
    assert digest(INITIAL/'identity_features.jsonl')==source['identity_feature_sha256']
    identities=[json.loads(line) for line in (INITIAL/'identity_features.jsonl').open()]
    X=pd.concat([pd.read_csv(PRIOR/'train_features.csv')[columns],pd.DataFrame(identities)],axis=1)
    ids=[];labels=[]
    for b in iter_battles(data_dir()/'train.jsonl'):ids.append(b['battle_id']);labels.append(int(b['player_won']))
    assignments=pd.read_csv(SELECTION_RUN/'splits.csv');assert ids==assignments.battle_id.tolist()
    cv=pd.read_csv(SELECTION_RUN/'cv_results.csv').set_index('candidate')
    score=float(cv.loc['rfe_300','cv_accuracy'])
    folder=ROOT/'outputs/leonard'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_rfe300_compact')
    folder.mkdir(parents=True);print('Artifacts:',folder,flush=True)
    manifest={'decision':'User prefers RFE300 for simplicity despite 0.10 pp lower mean CV accuracy.',
              'procedure_cv_accuracy':score,'score_source':str(SELECTION_RUN.relative_to(ROOT)),
              'fit_rows':len(labels),'method':'RFE','selected_features':300,'step':.25,
              'classifier':'LogisticRegression','C':.01,'max_iter':4000,'solver':'lbfgs','seed':42,
              'scaling':'baseline numeric standardized; identities unscaled',
              'train_sha256':source['train_sha256'],'splits_sha256':source['splits_sha256'],
              'versions':source['versions'],'holdout_evaluated':False,
              'caution':'This final feature set is fitted using all 10,000 labels. Future CV must refit RFE within folds, not reuse this all-data selection.'}
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with threadpool_limits(limits=2):
        model=make_selection_model(columns,'rfe',300)
        model.fit(X,np.asarray(labels))
        features=semantic_feature_names(model,columns);assert len(features)==len(set(features))==300
        table=pd.DataFrame({'feature':features,'category':[feature_category(f) for f in features],
                            'coefficient':model.named_steps['model'].coef_[0]})
        stability=pd.read_csv(SELECTION_RUN/'selection_stability.csv')
        stability=stability[stability.candidate.eq('rfe_300')].set_index('feature').selected_folds
        table['selected_in_cv_folds']=table.feature.map(stability).fillna(0).astype(int)
        table.sort_values(['category','feature']).to_csv(folder/'selected_features.csv',index=False)
        table.groupby('category').size().rename('features').reset_index().to_csv(folder/'feature_groups.csv',index=False)
        (folder/'selected_features.txt').write_text('\n'.join(features)+'\n')
        test_rows=[];test_ids=[]
        for b in iter_battles(data_dir()/'test.jsonl'):
            numbers=feature_row(b);test_ids.append(b['battle_id'])
            test_rows.append({**{k:numbers[k] for k in columns},**identity_features(b)})
        assert len(set(test_ids))==len(test_ids)
        test=pd.DataFrame(test_rows);pred=model.predict(test);prob=model.predict_proba(test)[:,1]
        sample=pd.read_csv(data_dir()/'sample_submission.csv')
        assert list(sample)==['battle_id','player_won'] and sample.battle_id.is_unique and sample.battle_id.notna().all()
        assert len(sample)==len(test_ids) and set(sample.battle_id)==set(test_ids)
        sub=sample.copy();sub['player_won']=sub.battle_id.map(pd.Series(pred,index=test_ids)).astype(int)
        assert sub.battle_id.equals(sample.battle_id) and sub.player_won.isin([0,1]).all() and sub.notna().all().all()
        sub.to_csv(folder/'submission.csv',index=False)
        pd.DataFrame({'battle_id':test_ids,'probability':prob}).to_csv(folder/'test_probabilities.csv',index=False)
        joblib.dump(model,folder/'model.joblib')
        reloaded=joblib.load(folder/'model.joblib')
        np.testing.assert_array_equal(reloaded.predict(test),pred)
        np.testing.assert_allclose(reloaded.predict_proba(test)[:,1],prob,rtol=1e-12,atol=1e-12)
    pd.testing.assert_frame_equal(sub,pd.read_csv(folder/'submission.csv'))
    for name in ('finalize_rfe300_model.py','run_feature_selection_experiment.py','identity_features.py','run_identity_scaling_followup.py','run_classical_identity_experiment.py'):
        shutil.copy2(Path(__file__).with_name(name),folder/name)
    for name in ('progressive_features.py','ablation_extensions.py','run_progressive_experiments.py','features_static.py','features_timeline.py','features_effects.py','features_roster.py'):
        shutil.copy2(BASELINE_RUN/name,folder/name)
    report={'candidate':'rfe_300','cv_accuracy':score,'features':300,'fit_rows':len(labels),'test_rows':len(sub),
            'categories':Counter(table.category),'submission_sha256':digest(folder/'submission.csv'),
            'model_sha256':digest(folder/'model.joblib'),'test_sha256':digest(data_dir()/'test.jsonl'),
            'sample_sha256':digest(data_dir()/'sample_submission.csv'),
            'prediction_counts':sub.player_won.value_counts().to_dict(),'serialized_model_checks_passed':True,
            'submission_checks_passed':True,'holdout_evaluated':False,'test_labels_used':False,
            'submission':str(folder/'submission.csv'),'feature_list':str(folder/'selected_features.csv')}
    (folder/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True);print('Finished:',folder,flush=True)
    return folder

if __name__=='__main__':run_finalize()
