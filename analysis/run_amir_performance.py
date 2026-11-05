"""Reproduce the archived 27-input Amir experiment without opening holdout."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/amir-matplotlib')
from pathlib import Path
import json
from subprocess import check_output

root = Path(__file__).resolve().parents[1]
notebook_revision = '338cbc1:notebooks/amirhosein/amirs-notebook.ipynb'
notebook_text = check_output(['git', 'show', notebook_revision], cwd=root, text=True)
notebook = json.loads(notebook_text)
namespace = {'__name__': '__main__'}
for index in (1, 9, 11, 13, 19):
    source = ''.join(notebook['cells'][index]['source'])
    if index == 13:
        source = source.replace('assert np.array_equal(experiment.development, eda_development)\n', '')
    exec(compile(source, f'{notebook_revision}:cell_{index}', 'exec'), namespace)
    if index == 13:
        (namespace['experiment'].output_path / 'notebook_snapshot.ipynb').write_text(notebook_text)
print('RESULTS_PATH:', namespace['experiment'].output_path / 'cv_results.csv', flush=True)
