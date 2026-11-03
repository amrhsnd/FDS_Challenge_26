"""Execute notebook 05 cells sequentially without a Jupyter server.

Uses the same Python cells and captures stdout/stderr as notebook outputs.
Run from repository root: .venv/bin/python notebooks/leonard/run_ablation.py
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
p = Path(__file__).with_name('05_aimilia_feature_ablation.ipynb').resolve()
os.chdir(p.parents[2])
notebook = json.loads(p.read_text())
notebook['cells'][15]['source'] = ''.join(notebook['cells'][15]['source']).replace(
    '## 6. Choose a configuration, then optionally open the holdout once',
    '## 7. Freeze the CV winner and evaluate the holdout once').replace(
    'Enable holdout evaluation **only after all feature/model choices are finished**.',
    'Holdout evaluation below runs **after all feature/model choices are finished**.')
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell.update(execution_count=None, outputs=[])
namespace = {'__name__': '__main__'}
count = 0
for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    count += 1
    print(f'Running cell {index}', flush=True)
    output = io.StringIO()
    try:
        with redirect_stdout(output), redirect_stderr(output):
            exec(compile(''.join(cell['source']), f'{p.name}:cell_{index}', 'exec'), namespace)
            # Plain-Python display prints representations; no HTML renderer needed.
    except BaseException:
        traceback.print_exc(file=output)
        raise
    finally:
        cell.update(execution_count=count, outputs=[{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue()}])
        p.write_text(json.dumps(notebook, indent=1) + '\n')
        print(output.getvalue(), flush=True)
        if 'experiment' in namespace:
            shutil.copy2(p, namespace['experiment'].output_path / p.name)
            shutil.copy2(Path(__file__), namespace['experiment'].output_path / Path(__file__).name)
print('Experiment finished.', flush=True)
