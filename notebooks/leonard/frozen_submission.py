"""Reproduce Leonard's frozen submission from official raw inputs only."""
from pathlib import Path
import json
import hashlib
import sys
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).parent)]
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from identity_features import IdentityEncoder, identity_features
from matchup_features import MatchupTransformer, summarize_battle
from counter_features import CounterTransformer
from run_progressive_experiments import feature_row
from pokemon_challenge.data import iter_battles

MANIFEST = Path(__file__).with_name('frozen_submission.json')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def classifier():
    return LogisticRegression(C=.01, max_iter=4000, solver='lbfgs', random_state=42)


def make_frozen_model(columns, chart_path):
    baseline = Pipeline([
        ('encoding', IdentityEncoder(columns)),
        ('scaling', ColumnTransformer([
            ('base', StandardScaler(), list(range(len(columns))))
        ], remainder='passthrough')),
        ('constant_filter', VarianceThreshold()),
        ('feature_selection', RFE(classifier(), n_features_to_select=300, step=.25)),
    ])
    def additions(transformer):
        return Pipeline([
            ('raw', transformer),
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('constant_filter', VarianceThreshold()),
            ('scaler', StandardScaler()),
        ])
    current = FeatureUnion([
        ('baseline', baseline),
        ('matchups', additions(MatchupTransformer(str(chart_path), 'team'))),
    ])
    return Pipeline([
        ('features', FeatureUnion([
            ('current', current),
            ('counters', additions(CounterTransformer(str(chart_path), 'quality'))),
        ])),
        ('model', classifier()),
    ])


def build_inputs(path, columns, labelled=False):
    rows, ids, labels = [], [], []
    for i, battle in enumerate(iter_battles(path), 1):
        numeric = feature_row(battle)
        rows.append({**{k: numeric[k] for k in columns},
                     **identity_features(battle), 'battle': summarize_battle(battle)})
        ids.append(battle['battle_id'])
        if labelled:
            labels.append(int(battle['player_won']))
        if i % 2000 == 0:
            print(f'{Path(path).name}: extracted {i} battles', flush=True)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Empty data or duplicate battle IDs')
    return pd.DataFrame(rows), ids, np.asarray(labels, dtype=int)


def selected_names(model, columns):
    branches = dict(model.named_steps['features'].transformer_list)
    current = dict(branches['current'].transformer_list)
    base = current['baseline']
    names = np.asarray([*columns, *base.named_steps['encoding'].vectorizer_.feature_names_], dtype=object)
    names = names[base.named_steps['constant_filter'].get_support()]
    names = list(names[base.named_steps['feature_selection'].get_support()])
    for prefix, branch in [('matchup', current['matchups']), ('counter', branches['counters'])]:
        raw_names = branch.named_steps['raw'].get_feature_names_out()
        names.extend(prefix + '__' + n for n in raw_names[branch.named_steps['constant_filter'].get_support()])
    return names


def create_submission(data_directory, output_directory):
    """Fit the frozen procedure; fail if data, source, selection or CSV drift."""
    spec = json.loads(MANIFEST.read_text())
    data_directory, output_directory = Path(data_directory), Path(output_directory)
    for name, expected in spec['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    for name, expected in spec['input_sha256'].items():
        if sha256(data_directory / name) != expected:
            raise ValueError(f'Official input differs from frozen run: {name}')
    versions = {name: version(name) for name in spec['versions']}
    differences = {k: {'expected': v, 'actual': versions[k]} for k, v in spec['versions'].items() if versions[k] != v}
    if differences:
        print('Runtime differs from reference; exact submission checks remain mandatory:', differences, flush=True)
    columns = spec['numeric_columns']
    X, _, y = build_inputs(data_directory / 'train.jsonl', columns, labelled=True)
    test, test_ids, _ = build_inputs(data_directory / 'test.jsonl', columns)
    if len(y) != 10000 or len(test_ids) != 5000 or set(y) != {0, 1}:
        raise ValueError('Unexpected frozen competition dataset shape or labels')
    print('Fitting frozen RFE300 + team16 + quality8 logistic regression...', flush=True)
    with threadpool_limits(limits=2):
        model = make_frozen_model(columns, data_directory / 'type_chart_gen2.csv')
        model.fit(X, y)
        names = selected_names(model, columns)
        if names != spec['selected_features']:
            raise ValueError('Feature selection differs from frozen model; use the reference runtime')
        prediction = model.predict(test)
    sample = pd.read_csv(data_directory / 'sample_submission.csv')
    if list(sample) != ['battle_id', 'player_won'] or not sample.battle_id.is_unique or set(sample.battle_id) != set(test_ids):
        raise ValueError('Sample submission IDs or columns do not match test data')
    submission = sample.copy()
    submission['player_won'] = submission.battle_id.map(pd.Series(prediction, index=test_ids)).astype(int)
    if not submission.notna().all().all() or not submission.player_won.isin([0, 1]).all():
        raise ValueError('Invalid predictions')
    # Check before writing: never silently replace the frozen submission with a changed result.
    payload = submission.to_csv(index=False, lineterminator='\n').encode()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != spec['submission_sha256']:
        raise ValueError('Predictions differ from frozen submission; use the reference runtime')
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / 'submission.csv'
    destination.write_bytes(payload)
    report = {'model': spec['model'], 'historical_development_cv_accuracy': spec['cv_accuracy'],
              'train_rows': len(y), 'submission_rows': len(submission), 'features': len(names),
              'submission_sha256': actual, 'exact_frozen_submission_reproduced': True,
              'versions': versions, 'runtime_differences': differences}
    (output_directory / 'reproduction_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Exact frozen submission reproduced: {destination}', flush=True)
    return destination, report
