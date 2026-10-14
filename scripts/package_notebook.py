"""Create the notebook-only handoff, excluding credentials and unrelated project files."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import nbformat

root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root / 'SolarGuard.ipynb', as_version=4)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
assert (root / 'outputs' / 'artifact_review.json').exists()
files = [root / name for name in ['SolarGuard.ipynb', 'README.md', 'requirements.txt',
    'DATA_SOURCE.md', 'CONTRIBUTIONS.md', 'NOTEBOOK_REVIEW.md']]
files += sorted((root / 'data').glob('*.csv'))
files += [root / 'data' / 'source_metadata.json']
files += sorted(p for p in (root / 'outputs').rglob('*') if p.is_file())
files += sorted((root / 'scripts').glob('*.py'))
archive = root / 'Group7_SolarGuard_Notebook.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as package:
    for path in files:
        package.write(path, arcname=path.relative_to(root))
with ZipFile(archive) as package:
    assert package.testzip() is None
    names = package.namelist()
    assert sum(n.startswith('data/') and n.endswith('.csv') for n in names) == 4
    assert 'SolarGuard.ipynb' in names and 'NOTEBOOK_REVIEW.md' in names
print(f'Notebook-only bundle: {archive.name}; {len(files)} files; {archive.stat().st_size:,} bytes')
print('This is not the complete course-submission ZIP: final slides and actual member contributions remain pending.')
