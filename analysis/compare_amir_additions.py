"""Paired seven-model check of the frozen selected additions; no holdout call."""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', '/tmp/amir-matplotlib')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    root = Path(__file__).resolve().parents[1]
    selection_path = Path(sys.argv[1]).resolve()
    selection = json.loads((selection_path / 'selection.json').read_text())
    notebook = json.loads((selection_path / 'baseline_notebook.ipynb').read_text())
    sys.path.insert(0, str(selection_path))
    from feature_candidates import extract_candidate_features
    from retained_material_candidates import extract_retained_material_candidates
    namespace = {'__name__': '__main__'}
    for index in (1, 9, 11):
        exec(compile(''.join(notebook['cells'][index]['source']), f'baseline:cell_{index}', 'exec'), namespace)
    baseline = namespace['extract_selected_features']

    def expanded(battle):
        candidates = {**extract_candidate_features(battle), **extract_retained_material_candidates(battle)}
        return {**baseline(battle), **{name: candidates[name] for name in selection['selected']}}

    run = datetime.now(ZoneInfo('Europe/Rome')).strftime('%Y%m%d_%H%M%S_addition_comparison')
    experiment = namespace['AmirExperiment'](
        owner='amirhosein', run=run, data_path=namespace['DATA_DIR'],
        feature_functions={'baseline': baseline, 'expanded': expanded})
    assert experiment.manifest['train_sha256'] == selection['train_sha256']
    assert experiment.manifest['splits_sha256'] == selection['splits_sha256']
    namespace['experiment'] = experiment
    definition = ''.join(notebook['cells'][19]['source']).split('results = experiment.compare(candidates)')[0]
    exec(compile(definition, 'baseline:candidates', 'exec'), namespace)
    Candidate = namespace['Candidate']
    candidates = [Candidate(f'{group}_{candidate.name}', groups=(group,),
                            model=candidate.model, percentile=candidate.percentile,
                            params=candidate.params)
                  for group in ('baseline', 'expanded') for candidate in namespace['candidates']]
    with threadpool_limits(limits=1):
        results = experiment.compare(candidates)
    baseline_results = results[results.candidate.str.startswith('baseline_')].copy()
    expanded_results = results[results.candidate.str.startswith('expanded_')].copy()
    for data, prefix in ((baseline_results, 'baseline_'), (expanded_results, 'expanded_')):
        data['configuration'] = data.candidate.str.removeprefix(prefix)
    paired = baseline_results.merge(expanded_results, on='configuration', suffixes=('_before', '_after'))
    paired['change_pp'] = 100 * (paired.cv_accuracy_after - paired.cv_accuracy_before)
    paired['folds_improved'] = sum(
        paired[f'fold_{i}_after'] > paired[f'fold_{i}_before'] + 1e-12 for i in range(1, 6))
    paired.to_csv(experiment.output_path / 'paired_comparison.csv', index=False)
    for name in ('baseline_notebook.ipynb', 'feature_candidates.py', 'retained_material_candidates.py', 'selection.json'):
        shutil.copyfile(selection_path / name, experiment.output_path / name)
    shutil.copyfile(__file__, experiment.output_path / Path(__file__).name)
    (selection_path / 'comparison_path.txt').write_text(str(experiment.output_path) + '\n')
    print(paired[['configuration', 'cv_accuracy_before', 'cv_accuracy_after', 'change_pp', 'folds_improved']].to_string(index=False))
    print('COMPARISON_PATH:', experiment.output_path, flush=True)


if __name__ == '__main__':
    main()
