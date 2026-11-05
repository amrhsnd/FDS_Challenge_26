"""Train 300-feature battle models with fold-local selection and grouped validation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from pokemon_challenge.experiment import Candidate, _write_json
from pokemon_challenge.models.classifiers import make_pipeline
from feature_catalog import FEATURE_NAMES, feature_dictionary
from feature_selection import CorrelationUtilitySelector
from features import P2_ROSTER_STAT_COLUMNS, build_feature_functions
from study import AmirStudy


class BattleFeatureLibrary:
    """Compute fixed numeric battle measurements with descriptive, unique names."""
    def __init__(self, chart_path, *, include_p2_roster_stats=True):
        self.functions = build_feature_functions(
            chart_path, include_p2_roster_stats=include_p2_roster_stats)
        self.feature_names = dict(FEATURE_NAMES)
        if not include_p2_roster_stats:
            for key in P2_ROSTER_STAT_COLUMNS:
                del self.feature_names[f'selected__{key}']

    def __call__(self, battle):
        row = {}
        for group, function in self.functions.items():
            for key, value in function(battle).items():
                raw = f'{group}__{key}'
                row[self.feature_names[raw]] = value
        if len(row) != len(self.feature_names):
            raise ValueError('The feature schema does not match its definitions.')
        return row


class ReducedBattleStudy(AmirStudy):
    """Fit feature selection inside each fold; expose a documented 300-column view."""
    def __init__(self, *, data_path, owner, run, n_features=300,
                 correlation_threshold=.90, include_p2_roster_stats=True, **kwargs):
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        library = BattleFeatureLibrary(Path(data_path)/'type_chart_gen2.csv',
                                       include_p2_roster_stats=include_p2_roster_stats)
        super().__init__(owner=owner, run=run, data_path=data_path,
                         feature_functions={'battle': library}, **kwargs)
        self.manifest.update(model_features=n_features,
            include_p2_roster_stats=include_p2_roster_stats,
            feature_selection='Fold-fitted utility ranking and absolute Pearson correlation pruning',
            requested_correlation_threshold=correlation_threshold)
        _write_json(self.output_path/'experiment.json', self.manifest)

    def _feature_row(self, battle, groups):
        row = super()._feature_row(battle, groups)
        return {name.removeprefix('battle__'): value for name, value in row.items()}

    def _columns(self, candidate):
        if tuple(candidate.groups) != ('battle',):
            raise ValueError('Use the battle feature group for 300-feature models.')
        return list(self.X.columns)

    def _selector(self):
        return CorrelationUtilitySelector(n_features=self.n_features,
            correlation_threshold=self.correlation_threshold,
            random_state=self.validation['seed'])

    def _pipeline(self, candidate):
        if candidate.percentile != 100:
            raise ValueError('The 300-feature pipeline does not apply an additional percentile filter.')
        memory = joblib.Memory(self.output_path/'pipeline_cache', verbose=0)
        if candidate.model == 'soft_voting':
            definitions = candidate.params.get('members', [])
            if len(definitions) < 2 or any(x['model'] == 'soft_voting' for x in definitions):
                raise ValueError('Provide at least two ordinary models for soft voting.')
            members = [Candidate(**definition) for definition in definitions]
            if any(tuple(member.groups) != ('battle',) or member.percentile != 100 for member in members):
                raise ValueError('Each voting member must use the complete battle group without another feature filter.')
            model = VotingClassifier(estimators=[(f'model_{i}', make_pipeline(
                member.model, seed=self.validation['seed'], params=member.params))
                for i, member in enumerate(members)], voting='soft', n_jobs=1,
                weights=candidate.params.get('weights'))
            steps = [('select_features', self._selector()),
                     ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('model', model)]
        else:
            numeric = make_pipeline(candidate.model, seed=self.validation['seed'],
                                    percentile=100, params=candidate.params)
            steps = [('select_features', self._selector()), *numeric.steps]
        return Pipeline(steps, memory=memory)

    def compare(self, candidates):
        super().compare(candidates)
        for result in self._results:
            result['candidate_features'] = len(self.X.columns)
            result['input_features'] = self.n_features
        self.results.to_csv(self.output_path/'cv_results.csv', index=False)
        return self.results

    def select_features(self):
        """Fit the readable feature set on development data, without opening holdout."""
        with threadpool_limits(limits=2):
            self.feature_selector = self._selector().fit(
                self.X.iloc[self.development], self.y.iloc[self.development])
        columns = list(self.feature_selector.get_feature_names_out())
        self.selected_matrix = self.X.loc[:, columns]
        self.feature_definitions = feature_dictionary(columns)
        self.selected_matrix.to_csv(self.output_path/'selected_features.csv', index=False)
        self.feature_definitions.to_csv(self.output_path/'feature_definitions.csv', index=False)
        self.feature_selector.audit_.to_csv(self.output_path/'feature_selection_audit.csv', index=False)
        _write_json(self.output_path/'feature_set.json', {
            'features': columns, 'count': len(columns),
            'selection_fit_rows': len(self.development),
            'requested_correlation_threshold': self.correlation_threshold,
            'effective_correlation_threshold': self.feature_selector.effective_threshold_,
            'selection_scope': 'development; validation refits selection separately inside each fold'})
        assert self.selected_matrix.shape[1] == self.n_features
        return self.selected_matrix

    def save_run_record(self, root):
        """Save execution settings and source files needed to repeat this run."""
        root = Path(root)
        destination = self.output_path/'sources'
        personal = root/'notebooks/amirhosein'
        paths = list((root/'notebooks/_amir_support').glob('*.py')) + [personal/'amirs-notebook.ipynb']
        paths += list((root/'src').rglob('*.py')) + list((root/'src').rglob('*.json'))
        paths += list((root/'notebooks/aimilia').glob('*.py'))
        paths += [root/'notebooks/leonard'/name for name in
                  ('ablation_extensions.py', 'identity_features.py', 'progressive_features.py')]
        paths += [root/'configs/validation.json', root/'requirements.txt', root/'pyproject.toml']
        manifest = []
        for path in paths:
            relative = path.relative_to(root)
            copied = destination/relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copied)
            manifest.append({'path': str(relative),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        _write_json(self.output_path/'source_manifest.json', manifest)
        _write_json(self.output_path/'settings.json', {
            'n_features': self.n_features, 'correlation_threshold': self.correlation_threshold,
            'include_p2_roster_stats': self.manifest['include_p2_roster_stats'],
            'selection_scope': 'inside training folds',
            'validation': self.validation, 'holdout_evaluated': self.manifest['holdout_evaluated'],
            'test_read': self.manifest['test_read']})

    def make_submission(self):
        result = super().make_submission()
        model = joblib.load(self.output_path/'model.joblib')
        selector = model.named_steps['select_features']
        columns = list(selector.get_feature_names_out())
        feature_dictionary(columns).to_csv(self.output_path/'final_feature_definitions.csv', index=False)
        selector.audit_.to_csv(self.output_path/'final_feature_selection_audit.csv', index=False)
        path = self.output_path/'final_fit.json'
        record = json.loads(path.read_text())
        record.update(selected_features=columns, selected_feature_count=len(columns))
        _write_json(path, record)
        return result


def model_candidates():
    """A small, fixed comparison of regularized linear and tree models."""
    logistic = Candidate('logistic', groups=('battle',), model='logistic_regression',
                         params={'C': .01, 'max_iter': 3000})
    histogram = Candidate('histogram_boosting', groups=('battle',), model='hist_gradient_boosting',
        params={'max_iter': 350, 'learning_rate': .04, 'max_leaf_nodes': 10,
                'min_samples_leaf': 70, 'l2_regularization': 10., 'early_stopping': False})
    xgboost = Candidate('xgboost', groups=('battle',), model='xgboost',
        params={'n_estimators': 400, 'max_depth': 3, 'learning_rate': .035,
                'subsample': .8, 'colsample_bytree': .8, 'min_child_weight': 8, 'reg_lambda': 8.})
    ensemble = Candidate('soft_voting', groups=('battle',), model='soft_voting',
                         params={'members': [asdict(c) for c in (logistic, histogram, xgboost)]})
    return [logistic, histogram, xgboost, ensemble]
