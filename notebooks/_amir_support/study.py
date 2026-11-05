"""Amir's duplicate-aware, development-only experiment runner.

Keeps the shared holdout, quarantines its exact observation duplicates, and
fits every transform inside grouped folds. Shared project APIs remain unchanged.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pokemon_challenge.data import iter_battles
from pokemon_challenge.experiment import BattleExperiment, Candidate, _safe_name, _write_json


def observed_battle(battle):
    """Copy only permitted inputs, sorted by actual turn number within 1–30."""
    from copy import deepcopy
    result = {key: deepcopy(battle.get(key)) for key in
              ('p1_team_details', 'p2_lead_details')}
    turns = [deepcopy(t) for t in (battle.get('battle_timeline') or [])
             if isinstance(t.get('turn'), (int, float)) and 1 <= t['turn'] <= 30]
    turns.sort(key=lambda t: t['turn'])
    if len({t['turn'] for t in turns}) != len(turns):
        raise ValueError('Duplicate turn numbers within a battle.')
    for turn in turns:
        for side in ('p1', 'p2'):
            state = turn.get(f'{side}_pokemon_state') or {}
            if state.get('status') == 'fnt':
                state['hp_pct'] = 0.0
    result['battle_timeline'] = turns
    return result


def observation_digest(battle):
    observed = observed_battle(battle)
    # Roster order does not change the observed team. Canonicalize this copy
    # for duplicate grouping without changing the feature extractor's input.
    if observed['p1_team_details'] is not None:
        observed['p1_team_details'].sort(key=lambda pokemon: json.dumps(
            pokemon, sort_keys=True, separators=(',', ':'), allow_nan=False))
    data = json.dumps(observed, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


class AmirStudy(BattleExperiment):
    """Grouped CV with OOF probabilities and paired feature-family comparisons."""
    def __init__(self, *args, rfe_features=80, pca_variance=.95, **kwargs):
        self.rfe_features, self.pca_variance = rfe_features, pca_variance
        super().__init__(*args, **kwargs)
        source_validation = dict(self.validation)
        self.validation['splitter'] = 'StratifiedGroupKFold'
        hashes = np.asarray([observation_digest(b) for b in
                             iter_battles(self.data_path / 'train.jsonl')])
        previous_holdout = set(hashes[self._holdout])
        allowed = np.asarray([h not in previous_holdout for h in hashes])
        self.development = np.flatnonzero(allowed)
        splitter = StratifiedGroupKFold(n_splits=self.validation['n_splits'],
                                       shuffle=True, random_state=self.validation['seed'])
        self.folds = list(splitter.split(self.X.iloc[self.development],
                         self.y.iloc[self.development], hashes[self.development]))
        self.assignments['observation_sha256'] = hashes
        self.assignments['partition'] = np.where(allowed, 'development', 'holdout_duplicate_quarantine')
        self.assignments.loc[self._holdout, 'partition'] = 'holdout'
        self.assignments['validation_fold'] = -1
        for number, (train, valid) in enumerate(self.folds):
            train_groups, valid_groups = (set(hashes[self.development[x]]) for x in (train, valid))
            assert not train_groups & valid_groups
            assert not (train_groups | valid_groups) & previous_holdout
            self.assignments.loc[self.development[valid], 'validation_fold'] = number
        self.assignments.to_csv(self.output_path / 'splits.csv', index=False)
        self.observation_hashes = hashes
        self.manifest.update(development_rows=len(self.development),
            source_validation=source_validation, validation=dict(self.validation),
            validation_protocol='StratifiedGroupKFold on exact permitted-observation hashes',
            duplicate_rows=int(pd.Series(hashes).duplicated(keep=False).sum()),
            quarantined_rows=int((self.assignments.partition == 'holdout_duplicate_quarantine').sum()),
            development_unique_observations=len(set(hashes[self.development])),
            splits_sha256=hashlib.sha256((self.output_path/'splits.csv').read_bytes()).hexdigest(),
            holdout_evaluated=False, test_read=False,
            limitations='Repeated development selection is exploratory; nonidentical related battles may remain.')
        _write_json(self.output_path / 'experiment.json', self.manifest)
        self.oof_probabilities = {}
        self.oof_predictions = {}
        print(f'Grouped validation: {len(self.development):,} development rows; '
              f'{self.manifest["quarantined_rows"]} holdout duplicates quarantined; '
              f'{self.manifest["duplicate_rows"]} rows belong to duplicate groups.')

    def _feature_row(self, battle, groups):
        return super()._feature_row(observed_battle(battle), groups)

    def _pipeline(self, candidate):
        if candidate.model == 'soft_voting':
            definitions = candidate.params.get('members')
            if not isinstance(definitions, list) or len(definitions) < 2:
                raise ValueError('Soft voting requires at least two member candidate definitions.')
            members = [Candidate(**definition) for definition in definitions]
            if any(member.model == 'soft_voting' for member in members):
                raise ValueError('Nested soft-voting members are not supported.')
            member_groups = set().union(*(set(member.groups) for member in members))
            if member_groups != set(candidate.groups):
                raise ValueError('Soft-voting member groups must together match the ensemble groups.')
            estimators = []
            for index, member in enumerate(members):
                columns = self._columns(member)
                selector = ColumnTransformer([('subset', 'passthrough', columns)],
                    remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')
                estimators.append((f'member_{index}', Pipeline([
                    ('columns', selector), ('pipeline', self._pipeline(member))])))
            return Pipeline([('inputs', 'passthrough'), ('model', VotingClassifier(
                estimators=estimators, voting='soft', n_jobs=1,
                weights=candidate.params.get('weights')))])
        if candidate.model not in {'pca_logistic', 'rfe_logistic', 'rfe_pca_logistic'}:
            return super()._pipeline(candidate)
        steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                 ('constant_filter', VarianceThreshold()), ('scaler', StandardScaler())]
        if 'rfe' in candidate.model:
            steps.append(('rfe', RFE(LogisticRegression(C=.1, max_iter=3000,
                         random_state=self.validation['seed']),
                         n_features_to_select=self.rfe_features, step=.2)))
        if 'pca' in candidate.model:
            steps.append(('pca', PCA(n_components=self.pca_variance, svd_solver='full')))
        steps.append(('model', LogisticRegression(random_state=self.validation['seed'],
                      **{'C':.1, 'max_iter':3000, **candidate.params})))
        return Pipeline(steps)

    def compare(self, candidates):
        if self._locked_candidate is not None:
            raise RuntimeError('Holdout opened: further comparisons are locked.')
        candidates = [Candidate(**asdict(c)) for c in candidates]
        names = [_safe_name(c.name) for c in candidates]
        if not names or len(names) != len(set(names)) or set(names) & set(self._candidates):
            raise ValueError('Candidate names must be new, unique, and nonempty.')
        X, y = self.X.iloc[self.development], self.y.iloc[self.development].to_numpy()
        majority_scores = [accuracy_score(y[v], np.full(len(v), np.bincount(y[t]).argmax()))
                           for t, v in self.folds]
        self._dummy_accuracy = float(np.mean(majority_scores))
        for candidate in candidates:
            columns = self._columns(candidate)
            probability = np.full(len(y), np.nan)
            predicted = np.full(len(y), -1, dtype=int)
            scores, train_scores, fit_times, selected = [], [], [], []
            with threadpool_limits(limits=2):
                for train, valid in self.folds:
                    model = clone(self._pipeline(candidate))
                    started = time.monotonic()
                    model.fit(X.iloc[train][columns], y[train])
                    fit_times.append(time.monotonic() - started)
                    probability[valid] = model.predict_proba(X.iloc[valid][columns])[:, 1]
                    # Preserve the estimator's class decision, including its
                    # tie rule when the positive probability is exactly .5.
                    predicted[valid] = model.predict(X.iloc[valid][columns])
                    scores.append(accuracy_score(y[valid], predicted[valid]))
                    train_scores.append(accuracy_score(y[train], model.predict(X.iloc[train][columns])))
                    selected.append(list(model[:-1].get_feature_names_out(columns)))
            assert np.isfinite(probability).all()
            assert np.isin(predicted, [0, 1]).all()
            is_pca = 'pca' in candidate.model
            is_ensemble = candidate.model == 'soft_voting'
            frequency = pd.DataFrame({'feature': columns,
                'selected_in_folds': [np.nan if is_pca or is_ensemble else sum(c in f for f in selected) for c in columns]})
            if is_pca:
                frequency['note'] = 'PCA components combine original columns; selection frequency is not applicable.'
            elif is_ensemble:
                frequency['note'] = 'Members fit separate transforms; ensemble-level selection frequency is not applicable.'
            frequency.to_csv(self.output_path / f'selection_frequency_{candidate.name}.csv', index=False)
            result = dict(candidate=candidate.name, model=candidate.model, groups=', '.join(candidate.groups),
                input_features=len(columns), percentile=candidate.percentile,
                mean_selected_features=float(np.mean(list(map(len, selected)))),
                cv_accuracy=float(np.mean(scores)), fold_std=float(np.std(scores)),
                mean_fit_seconds=float(np.mean(fit_times)), dummy_accuracy=self._dummy_accuracy,
                train_accuracy=float(np.mean(train_scores)),
                train_cv_gap=float(np.mean(train_scores)-np.mean(scores)),
                oof_accuracy=float(accuracy_score(y, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                roc_auc=float(roc_auc_score(y, probability)), log_loss=float(log_loss(y, probability)),
                brier_score=float(brier_score_loss(y, probability)))
            result.update({f'fold_{i+1}': float(v) for i, v in enumerate(scores)})
            self._results.append(result)
            self._candidates[candidate.name] = candidate
            self.oof_probabilities[candidate.name] = probability
            self.oof_predictions[candidate.name] = predicted
            pd.DataFrame({'battle_id': self.ids[self.development], 'actual': y,
                'probability': probability, 'predicted': predicted,
                'fold': self.assignments.loc[self.development, 'validation_fold'].to_numpy()}
            ).to_csv(self.output_path / f'oof_{candidate.name}.csv', index=False)
            self.results.to_csv(self.output_path / 'cv_results.csv', index=False)
            _write_json(self.output_path / 'candidates.json', [asdict(c) for c in self._candidates.values()])
            print(f'{candidate.name}: CV accuracy {result["cv_accuracy"]:.3%} '
                  f'across {len(self.folds)} folds.', flush=True)
        return self.results

    def paired_comparison(self, baseline, candidates):
        table = self.results.set_index('candidate')
        truth = self.y.iloc[self.development].to_numpy()
        base_correct = self.oof_predictions[baseline] == truth
        rows = []
        for name in candidates:
            difference = np.asarray([table.loc[name, f'fold_{i}'] - table.loc[baseline, f'fold_{i}']
                                     for i in range(1, len(self.folds)+1)])
            correct = self.oof_predictions[name] == truth
            rows.append(dict(candidate=name, baseline=baseline, delta_pp=100*difference.mean(),
                folds_better=int((difference > 1e-12).sum()), folds_worse=int((difference < -1e-12).sum()),
                corrected_errors=int((~base_correct & correct).sum()),
                introduced_errors=int((base_correct & ~correct).sum()),
                **{f'delta_fold_{i+1}_pp':100*v for i,v in enumerate(difference)}))
        return pd.DataFrame(rows).sort_values('delta_pp', ascending=False)

    def evaluate_holdout(self, candidate=None):
        result = super().evaluate_holdout(candidate)
        self.manifest['holdout_evaluated'] = True
        _write_json(self.output_path / 'experiment.json', self.manifest)
        return result

    def make_submission(self):
        result = super().make_submission()
        self.manifest['test_read'] = True
        _write_json(self.output_path / 'experiment.json', self.manifest)
        return result
