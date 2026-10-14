"""Execute from a fresh kernel and save visible outputs, including failures."""
from pathlib import Path
import os
import time
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/solarguard-matplotlib')
os.environ.setdefault('IPYTHONDIR', '/tmp/solarguard-ipython')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/solarguard-jupyter')
nb = nbformat.read(root / 'SolarGuard.ipynb', as_version=4)
started = time.perf_counter()

def on_start(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Cell {cell_index + 1}/{len(nb.cells)} started ({time.perf_counter()-started:.0f}s)', flush=True)

client = NotebookClient(nb, timeout=3600, kernel_name='python3',
                        resources={'metadata': {'path': str(root)}}, on_cell_start=on_start)
try:
    client.execute()
finally:
    nbformat.write(nb, root / 'SolarGuard.ipynb')
print(f'Fresh-kernel execution finished: {time.perf_counter()-started:.1f}s', flush=True)
