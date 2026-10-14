"""Sync authoring Markdown into executed notebook without modifying code or outputs."""
from pathlib import Path
import runpy
import nbformat

root = Path(__file__).resolve().parents[1]
authored = runpy.run_path(str(root / 'scripts' / 'build_notebook.py'))['cells']
nb = nbformat.read(root / 'SolarGuard.ipynb', as_version=4)
assert len(nb.cells) == len(authored)
changed = 0
for current, source in zip(nb.cells, authored):
    assert current.cell_type == source.cell_type
    if current.cell_type == 'code':
        assert current.source == source.source, 'Code changed; fresh execution is required'
    elif current.source != source.source:
        current.source = source.source
        changed += 1
nbformat.validate(nb)
nbformat.write(nb, root / 'SolarGuard.ipynb')
print(f'Synced {changed} Markdown cells; executable cells and outputs preserved.')
