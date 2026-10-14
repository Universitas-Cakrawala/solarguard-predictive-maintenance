"""Apply Indonesian narrative changes while preserving analytical code and results."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re
import runpy
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / 'SolarGuard.ipynb'
before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
nb = nbformat.read(path, as_version=4)
authored = runpy.run_path(str(root / 'scripts' / 'build_notebook.py'))['cells']
assert len(nb.cells) == len(authored)

class StripNarrative(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'note':
            return ast.Call(func=node.func, args=[ast.Constant(value='narrative')], keywords=[])
        return self.generic_visit(node)

def analytical_structure(source):
    return ast.dump(StripNarrative().visit(ast.parse(source)), include_attributes=False)

changed_markdown = changed_code = changed_outputs = 0
for current, new in zip(nb.cells, authored):
    assert current.cell_type == new.cell_type
    if current.cell_type == 'markdown':
        changed_markdown += current.source != new.source
        current.source = new.source
        continue
    assert analytical_structure(current.source) == analytical_structure(new.source), 'Analytical code changed'
    old_notes = [n for n in ast.walk(ast.parse(current.source)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'note']
    new_notes = [n for n in ast.walk(ast.parse(new.source)) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'note']
    assert len(old_notes) == len(new_notes)
    replacements = {}
    for old_call, new_call in zip(old_notes, new_notes):
        old_nodes, new_nodes = list(ast.walk(old_call)), list(ast.walk(new_call))
        assert len(old_nodes) == len(new_nodes), 'Narrative expression structure changed'
        for old_node, new_node in zip(old_nodes, new_nodes):
            assert type(old_node) is type(new_node)
            if isinstance(old_node, ast.Constant) and isinstance(old_node.value, str):
                assert isinstance(new_node.value, str)
                if old_node.value != new_node.value:
                    assert old_node.value not in replacements or replacements[old_node.value] == new_node.value
                    replacements[old_node.value] = new_node.value
            elif isinstance(old_node, ast.Constant):
                assert old_node.value == new_node.value
    for output in current.outputs:
        if 'text/markdown' not in output.get('data', {}):
            continue
        old_text = output.data['text/markdown']
        new_text = old_text
        for old_literal in sorted(replacements, key=len, reverse=True):
            new_text = new_text.replace(old_literal, replacements[old_literal])
        assert re.findall(r'\d+(?:[.,]\d+)*', old_text) == re.findall(r'\d+(?:[.,]\d+)*', new_text), 'Numeric result changed'
        changed_outputs += old_text != new_text
        output.data['text/markdown'] = new_text
    changed_code += current.source != new.source
    current.source = new.source

nbformat.validate(nb)
nbformat.write(nb, path)
report = {'language': 'id', 'markdown_cells_translated': changed_markdown,
          'code_cells_with_narrative_only_changes': changed_code,
          'markdown_outputs_translated': changed_outputs,
          'analytical_ast_unchanged': True, 'numeric_markdown_results_unchanged': True,
          'model_refitted': False, 'before_sha256': before_hash,
          'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(root / 'outputs' / 'language_revision.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
