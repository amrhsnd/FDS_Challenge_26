"""Screen source-backed additions against an immutable Amir notebook snapshot.

Run from the repository with its .venv Python. This is exploratory development-
only selection, not an unbiased estimate of the selected features' future gain.
The existing holdout is never evaluated and official test data is never read.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pokemon_challenge.data import iter_battles
from pokemon_challenge.models.classifiers import make_pipeline
from pokemon_challenge.paths import data_dir
from pokemon_challenge.validation.splits import make_splits

MIN_GAIN = 0.00125  # Ten additional correct predictions over 8,000 rows.
MIN_FOLD_WINS = 3
MAX_CORRELATION = 0.98
MAX_ADDITIONS = 10
MAX_LOG_LOSS_INCREASE = 0.0005


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(X, y, folds, pipeline):
    accuracy, losses, predictions = [], [], np.zeros(len(y))
    for train, valid in folds:
        estimator = clone(pipeline).fit(X.iloc[train], y[train])
        probabilities = estimator.predict_proba(X.iloc[valid])[:, 1]
        predictions[valid] = probabilities
        accuracy.append(accuracy_score(y[valid], probabilities >= 0.5))
        losses.append(log_loss(y[valid], probabilities))
    return {'accuracy': float(np.mean(accuracy)), 'log_loss': float(np.mean(losses)),
            'folds': accuracy, 'losses': losses, 'probabilities': predictions}


def main():
    from feature_candidates import extract_candidate_features
    from retained_material_candidates import extract_retained_material_candidates

    run = datetime.now(ZoneInfo('Europe/Rome')).strftime('%Y%m%d_%H%M%S_feature_selection')
    output = ROOT / 'outputs' / 'amirhosein' / run
    output.mkdir(parents=True, exist_ok=False)
    notebook_path = ROOT / 'notebooks/amirhosein/amirs-notebook.ipynb'
    snapshot = output / 'baseline_notebook.ipynb'
    shutil.copyfile(notebook_path, snapshot)
    for module in ['select_amir_additions.py', 'feature_candidates.py', 'retained_material_candidates.py']:
        shutil.copyfile(Path(__file__).parent / module, output / module)
    notebook = json.loads(snapshot.read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'def extract_selected_features(' in ''.join(c['source']))
    ns = {'np': np, 'Counter': Counter}
    exec(compile(source, str(snapshot), 'exec'), ns)
    original_names = list(ns['FEATURE_COLUMNS'])
    assert len(original_names) == 28
    rows, additions, ids, labels = [], [], [], []
    source_path = data_dir() / 'train.jsonl'
    for battle in iter_battles(source_path):
        observed = {k: v for k, v in battle.items() if k not in {'battle_id', 'player_won'}}
        rows.append(ns['extract_selected_features'](observed))
        candidate = extract_candidate_features(observed)
        material = extract_retained_material_candidates(observed)
        assert not set(candidate).intersection(material)
        additions.append({**candidate, **material})
        ids.append(battle['battle_id'])
        labels.append(battle['player_won'])
    config = json.loads((ROOT / 'configs/validation.json').read_text())
    development, _, folds, assignments = make_splits(
        ids, labels, seed=config['seed'], n_splits=config['n_splits'],
        holdout_size=config['holdout_size'])
    assignments.to_csv(output / 'splits.csv', index=False)
    X = pd.DataFrame(rows, dtype=float).iloc[development].reset_index(drop=True)
    Z = pd.DataFrame(additions, dtype=float).iloc[development].reset_index(drop=True)
    y = np.asarray(labels)[development]
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(Z.to_numpy()).all()
    X.to_csv(output / 'baseline_development_features.csv', index=False)
    Z.to_csv(output / 'candidate_development_features.csv', index=False)
    pd.DataFrame({'battle_id': np.asarray(ids)[development], 'player_won': y}).to_csv(
        output / 'development_rows.csv', index=False)
    all_features = pd.concat([X, Z], axis=1)
    corr = all_features.corr().abs()
    profile = pd.DataFrame({
        'feature': Z.columns, 'n_unique': Z.nunique().values,
        'nonzero_rows': (Z != 0).sum().values,
        'max_abs_baseline_correlation': corr.loc[Z.columns, X.columns].max(axis=1).values,
    })
    eligible = profile[(profile.n_unique > 1) & (profile.nonzero_rows >= 50)
                       & (profile.max_abs_baseline_correlation < MAX_CORRELATION)].feature.tolist()
    profile.to_csv(output / 'candidate_profile.csv', index=False)
    pipeline = make_pipeline('logistic_regression', seed=config['seed'], percentile=100,
                             params={'C': 1.0, 'max_iter': 2000})
    records, selected = [], []
    with threadpool_limits(limits=1):
        baseline = evaluate(X, y, folds, pipeline)
        current = baseline
        print('OUTPUT:', output, flush=True)
        print('Baseline:', baseline['accuracy'], baseline['folds'], flush=True)
        for step in range(1, MAX_ADDITIONS + 1):
            step_results = []
            basis = all_features[original_names + selected].to_numpy()
            basis = (basis - basis.mean(axis=0)) / np.maximum(basis.std(axis=0), 1e-12)
            basis = np.column_stack([np.ones(len(basis)), basis])
            for name in eligible:
                if name in selected:
                    continue
                max_corr = corr.loc[name, original_names + selected].max()
                if max_corr >= MAX_CORRELATION:
                    continue
                target = all_features[name].to_numpy()
                target = (target - target.mean()) / max(target.std(), 1e-12)
                residual = target - basis @ np.linalg.lstsq(basis, target, rcond=None)[0]
                if np.mean(residual ** 2) < 1e-12:
                    continue  # Reject exact linear re-expressions of existing inputs.
                result = evaluate(all_features[original_names + selected + [name]], y, folds, pipeline)
                delta = result['accuracy'] - current['accuracy']
                wins = int(np.sum(np.asarray(result['folds']) > np.asarray(current['folds']) + 1e-12))
                loss_delta = result['log_loss'] - current['log_loss']
                passes = (delta + 1e-12 >= MIN_GAIN and wins >= MIN_FOLD_WINS
                          and loss_delta <= MAX_LOG_LOSS_INCREASE)
                record = {'step': step, 'feature': name, 'current_features': '|'.join(selected),
                          'accuracy': result['accuracy'], 'gain_pp': delta * 100,
                          'folds_improved': wins, 'log_loss': result['log_loss'],
                          'log_loss_change': loss_delta, 'passes': passes,
                          **{f'fold_{i+1}': score for i, score in enumerate(result['folds'])}}
                records.append(record)
                step_results.append((record, result))
            pd.DataFrame(records).to_csv(output / 'forward_screen.csv', index=False)
            qualifying = [pair for pair in step_results if pair[0]['passes']]
            if not qualifying:
                print('Stopped: no remaining feature meets all preset criteria.', flush=True)
                break
            record, current = sorted(qualifying, key=lambda p: (-p[0]['accuracy'], p[0]['log_loss'], p[0]['feature']))[0]
            selected.append(record['feature'])
            print('Selected:', record, flush=True)
        ablations = []
        for name in selected:
            reduced = evaluate(all_features[original_names + [s for s in selected if s != name]], y, folds, pipeline)
            ablations.append({'feature': name, 'full_accuracy': current['accuracy'],
                              'without_accuracy': reduced['accuracy'],
                              'removal_cost_pp': 100 * (current['accuracy'] - reduced['accuracy']),
                              'folds_worse_without': int(np.sum(np.asarray(current['folds']) > np.asarray(reduced['folds']) + 1e-12)),
                              'without_log_loss': reduced['log_loss']})
        pd.DataFrame(ablations).to_csv(output / 'leave_one_out.csv', index=False)
    manifest = {
        'selected': selected, 'baseline_features': original_names,
        'candidate_count': Z.shape[1], 'eligible_candidates': eligible,
        'development_rows': len(y), 'validation': config,
        'baseline_accuracy': baseline['accuracy'], 'baseline_folds': baseline['folds'],
        'selected_accuracy': current['accuracy'], 'selected_folds': current['folds'],
        'baseline_log_loss': baseline['log_loss'], 'selected_log_loss': current['log_loss'],
        'criteria': {'minimum_incremental_accuracy': MIN_GAIN, 'minimum_fold_wins': MIN_FOLD_WINS,
                     'maximum_abs_correlation': MAX_CORRELATION, 'maximum_additions': MAX_ADDITIONS,
                     'maximum_log_loss_increase': MAX_LOG_LOSS_INCREASE, 'minimum_nonzero_rows': 50},
        'train_sha256': digest(source_path), 'splits_sha256': digest(output / 'splits.csv'),
        'baseline_notebook_sha256': digest(snapshot),
        'limitation': 'Feature selection reuses development folds; scores are exploratory, not unbiased generalization estimates. No holdout evaluation or official test read.',
    }
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pd.DataFrame({'battle_id': np.asarray(ids)[development], 'actual': y,
                  'baseline_probability': baseline['probabilities'],
                  'selected_probability': current['probabilities']}).to_csv(output / 'paired_oof_predictions.csv', index=False)
    print('SELECTION:', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
