"""Execute every notebook code cell without nbclient, then write a static preview.

Run with the project interpreter, for example::

    .venv/bin/python analysis/execute_notebook.py /absolute/path/notebook.ipynb

The input notebook is never overwritten. Execution stops on the first error and
still saves the partial notebook, HTML preview, and machine-readable receipt.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output
from traitlets.config import Config


def _markdown(source):
    """Use an installed renderer; escaping is a sufficient offline fallback."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        try:
            import markdown
        except ImportError:
            return '<pre class="markdown-fallback">' + html.escape(source) + '</pre>'
        return markdown.markdown(source, extensions=['tables', 'fenced_code'])
    return MarkdownIt('commonmark').render(source)


def _text(value):
    return ''.join(value) if isinstance(value, list) else str(value)


def write_preview(notebook, destination, receipt):
    parts = [f'<h1>{html.escape(destination.stem)}</h1>',
             '<p class="receipt">' + html.escape(
                 f"{receipt['status']}: {receipt['executed_code_cells']}/"
                 f"{receipt['total_code_cells']} code cells executed; "
                 f"{receipt['png_outputs']} PNG outputs; "
                 f"{receipt['html_outputs']} HTML outputs.") + '</p>']
    for index, cell in enumerate(notebook.cells):
        parts.append(f'<section id="cell-{index}"><div class="cell-label">Cell {index}</div>')
        if cell.cell_type == 'markdown':
            parts.append(_markdown(cell.source))
        elif cell.cell_type == 'code':
            parts.append('<details><summary>Source · execution ' +
                         html.escape(str(cell.execution_count)) + '</summary><pre>' +
                         html.escape(cell.source) + '</pre></details>')
            for output in cell.outputs:
                if output.output_type == 'stream':
                    parts.append('<pre class="stream">' + html.escape(output.text) + '</pre>')
                elif output.output_type == 'error':
                    parts.append('<pre class="error">' + html.escape('\n'.join(output.traceback)) + '</pre>')
                else:
                    data = output.get('data', {})
                    if 'image/png' in data:
                        parts.append('<img alt="Notebook figure" src="data:image/png;base64,' +
                                     _text(data['image/png']).replace('\n', '') + '">')
                    elif 'text/html' in data:
                        # Keep tables and styles, but this preview never executes output scripts.
                        fragment = re.sub(r'<script\b[^>]*>.*?</script\s*>', '',
                                          _text(data['text/html']), flags=re.I | re.S)
                        parts.append('<div class="rich-output">' + fragment + '</div>')
                    elif 'text/markdown' in data:
                        parts.append(_markdown(_text(data['text/markdown'])))
                    elif 'text/plain' in data:
                        parts.append('<pre>' + html.escape(_text(data['text/plain'])) + '</pre>')
        else:
            parts.append('<pre>' + html.escape(cell.source) + '</pre>')
        parts.append('</section>')
    destination.write_text('''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Executed notebook preview</title><style>
body{max-width:1200px;margin:36px auto;padding:0 24px;font:15px/1.5 system-ui;color:#18212b}
section{margin:28px 0;border-top:1px solid #dce2e8;padding-top:14px}.cell-label{color:#607080;font-size:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px;border-radius:5px;font:12px/1.5 monospace}
.markdown-fallback{background:white;font:15px/1.6 system-ui}.error{background:#ffe8e8;color:#8e1111}
img{display:block;max-width:100%;height:auto;margin:16px auto}table{border-collapse:collapse;font-size:12px;max-width:100%}
td,th{padding:6px 9px;border-bottom:1px solid #dce2e8;text-align:right}.rich-output{overflow-x:auto}
summary{cursor:pointer;color:#3f5f79}.receipt{background:#edf4fa;padding:12px}
</style></head><body>''' + '\n'.join(parts) + '</body></html>', encoding='utf-8')


def execute(notebook_path, output_path, html_path, repo):
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    receipt = {
        'input': str(notebook_path), 'output': str(output_path), 'preview': str(html_path),
        'cwd': str(repo), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'total_code_cells': sum(cell.cell_type == 'code' for cell in notebook.cells),
        'executed_code_cells': 0, 'png_outputs': 0, 'html_outputs': 0,
        'status': 'running', 'cells': [],
    }
    original_cwd = Path.cwd()
    os.chdir(repo)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='notebook-matplotlib-') as mpl_config:
            os.environ.setdefault('MPLCONFIGDIR', mpl_config)
            os.environ.setdefault('IPYTHONDIR', str(Path(mpl_config) / 'ipython'))
            os.environ['MPLBACKEND'] = 'Agg'
            os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt

            config = Config()
            config.HistoryManager.enabled = False
            shell = InteractiveShell.instance(config=config)
            shell.user_ns['__name__'] = '__main__'
            shell.colors = 'nocolor'
            # run_cell installs its own displayhook. Route its final expression
            # through the same rich publisher as explicit display(...).
            shell.displayhook.write_output_prompt = lambda: None
            shell.displayhook.write_format_data = lambda data, metadata=None: shell.display_pub.publish(
                data=data, metadata=metadata or {})

            def capture_figures(*args, **kwargs):
                for number in list(plt.get_fignums()):
                    figure = plt.figure(number)
                    buffer = BytesIO()
                    figure.savefig(buffer, format='png', bbox_inches='tight')
                    display({'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}, raw=True)
                    plt.close(figure)

            plt.show = capture_figures
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                cell_started = time.monotonic()
                result, error = None, None
                with capture_output() as captured:
                    try:
                        result = shell.run_cell(cell.source, store_history=True, silent=False,
                                                cell_id=cell.get('id'), cell_meta=dict(cell.metadata))
                        capture_figures()  # Match inline notebooks' implicit end-of-cell figures.
                    except BaseException as unexpected:
                        error = unexpected
                cell.execution_count = (result.execution_count if result is not None
                                        else shell.execution_count)
                for name, text in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
                    if text:
                        cell.outputs.append(nbformat.v4.new_output('stream', name=name, text=text))
                for rich in captured.outputs:
                    data = dict(rich.data)
                    for mime, value in data.items():
                        if isinstance(value, bytes):
                            data[mime] = base64.b64encode(value).decode('ascii')
                    cell.outputs.append(nbformat.v4.new_output(
                        'display_data', data=data, metadata=dict(rich.metadata)))
                if result is not None:
                    error = error or result.error_before_exec or result.error_in_exec
                if error is not None:
                    cell.outputs.append(nbformat.v4.new_output(
                        'error', ename=type(error).__name__, evalue=str(error),
                        traceback=traceback.format_exception(type(error), error, error.__traceback__)))
                record = {'cell_index': index, 'execution_count': cell.execution_count,
                          'seconds': round(time.monotonic() - cell_started, 3),
                          'outputs': len(cell.outputs), 'success': error is None}
                receipt['cells'].append(record)
                receipt['executed_code_cells'] += 1
                print(f"Cell {index}: {'ERROR' if error else 'OK'}; "
                      f"{record['seconds']:.2f}s; {len(cell.outputs)} outputs", flush=True)
                if error is not None:
                    receipt.update(status='failed', failed_cell=index,
                                   error=f'{type(error).__name__}: {error}')
                    break
            else:
                receipt['status'] = 'success'
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                       setup_traceback=traceback.format_exception(type(error), error, error.__traceback__))
        print('Notebook runner failed:', receipt['error'], file=sys.stderr)
    finally:
        os.chdir(original_cwd)
        receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        for cell in notebook.cells:
            for output in cell.get('outputs', []):
                receipt['png_outputs'] += int('image/png' in output.get('data', {}))
                receipt['html_outputs'] += int('text/html' in output.get('data', {}))
        notebook.metadata['codex_execution'] = {key: receipt[key] for key in (
            'status', 'started_utc', 'finished_utc', 'executed_code_cells', 'total_code_cells')}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_path)
        write_preview(notebook, html_path, receipt)
        receipt_path = output_path.with_suffix('.execution.json')
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({key: receipt[key] for key in ('status', 'output', 'preview',
                          'executed_code_cells', 'total_code_cells', 'png_outputs', 'html_outputs')}, indent=2))
        print('Execution receipt:', receipt_path)
    return 0 if receipt['status'] == 'success' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--html', type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    output_path = (args.output or notebook_path.with_name(notebook_path.stem + '_executed.ipynb')).resolve()
    if output_path == notebook_path:
        parser.error('--output must differ from the input notebook')
    html_path = (args.html or output_path.with_suffix('.html')).resolve()
    if html_path in {notebook_path, output_path}:
        parser.error('--html must differ from the input and output notebook')
    return execute(notebook_path, output_path, html_path, args.repo.resolve())


if __name__ == '__main__':
    sys.exit(main())
