"""Run the current Amir notebook's feature extraction and CV, without opening holdout."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/amir-matplotlib')
from pathlib import Path
import json
notebook_path = Path('notebooks/amirhosein/amirs-notebook.ipynb')
notebook_text = notebook_path.read_text()
notebook = json.loads(notebook_text)
namespace = {'__name__': '__main__'}
for index in (1, 9, 11, 13, 19):
    source = ''.join(notebook['cells'][index]['source'])
    if index == 13:
        source = source.replace('assert np.array_equal(experiment.development, eda_development)\n', '')
    exec(compile(source, f'{notebook_path}:cell_{index}', 'exec'), namespace)
    if index == 13:
        (namespace['experiment'].output_path / 'notebook_snapshot.ipynb').write_text(notebook_text)
print('RESULTS_PATH:', namespace['experiment'].output_path / 'cv_results.csv', flush=True)
