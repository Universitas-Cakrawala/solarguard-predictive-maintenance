"""Verify executed artifact and cross-check stored results without refitting models."""
from pathlib import Path
import ast
import hashlib
import json
import nbformat
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(__file__).resolve().parents[1]
nb = nbformat.read(root / 'SolarGuard.ipynb', as_version=4)
nbformat.validate(nb)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
errors = [o for c in code_cells for o in c.outputs if o.output_type == 'error']
assert not errors, errors
assert all(c.execution_count is not None for c in code_cells)
assert [c.execution_count for c in code_cells] == list(range(1, len(code_cells) + 1))
for c in code_cells:
    ast.parse(c.source)
headings = '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')
assert all(f'## {n}. ' in headings for n in range(1, 14))
out = root / 'outputs'
manifest = json.loads((out / 'data_manifest.json').read_text())
for item in manifest:
    assert hashlib.sha256((root / 'data' / item['file']).read_bytes()).hexdigest() == item['sha256']
ranking = pd.read_csv(out / 'default_cv_ranking.csv')
expected_top_two = ranking.loc[ranking.model.ne('Mean baseline'), 'model'].head(2).tolist()
tuning = pd.read_csv(out / 'default_vs_tuned.csv')
assert set(tuning.model) == set(expected_top_two)
for row in tuning.itertuples():
    params = json.loads(row.best_parameters)
    assert len(params) >= 2
    result = pd.read_csv(out / (row.model.lower().replace(' ', '_') + '_search_results.csv'))
    assert len(result) == row.configurations
    assert np.isclose(-result.mean_test_mae.max(), row.tuned_CV_MAE)
folds = pd.read_csv(out / 'cv_folds.csv')
assert (pd.to_datetime(folds.train_end) < pd.to_datetime(folds.validation_start)).all()
assert (folds.train_inverters >= folds.validation_inverters).all()
selection = pd.read_csv(out / 'final_development_selection.csv')
lock = json.loads((out / 'selection_lock.json').read_text())
assert selection.iloc[0].candidate == lock['selected']
assert np.isclose(selection.iloc[0].CV_MAE, lock['selected_CV_MAE'])
initial_path = out / 'initial_selection_lock.json'
if initial_path.exists():
    initial = json.loads(initial_path.read_text())
    assert initial['selected'] == lock['selected']
    assert initial['features'] == lock['features']
    # Thread count follows the machine; it is not a statistical hyperparameter.
    assert {k: v for k, v in initial['parameters'].items() if k != 'n_jobs'} == {
        k: v for k, v in lock['parameters'].items() if k != 'n_jobs'}
    for old, new in zip(initial['thresholds'], lock['thresholds']):
        assert old['PLANT_ID'] == new['PLANT_ID']
        assert np.isclose(old['gap_threshold'], new['gap_threshold'])
        assert np.isclose(old['min_expected'], new['min_expected'])
pred = pd.read_csv(out / 'final_test_predictions.csv')
score = pd.read_csv(out / 'final_test_metrics.csv').iloc[0]
assert len(pred) == score.n
assert np.isclose(mean_absolute_error(pred.actual, pred.predicted), score.MAE)
assert np.isclose(np.sqrt(mean_squared_error(pred.actual, pred.predicted)), score.RMSE)
assert np.isclose(r2_score(pred.actual, pred.predicted), score.R2)
assert np.allclose(pred.actual - pred.predicted, pred.residual)
assert np.allclose(pred.gap, -pred.residual)
alerts = pd.read_csv(out / 'test_alert_predictions.csv')
assert np.array_equal(alerts.persistent_alert, alerts.streak.ge(3))
assert np.array_equal(alerts.new_episode, alerts.streak.eq(3))
assert len(pd.read_csv(out / 'evidence_action_matrix.csv')) >= 3
summary = json.loads((out / 'run_summary.json').read_text())
assert all(summary['checks'].values())
png_outputs = sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
assert png_outputs >= 10
review = {'nbformat_valid': True, 'executed_code_cells': len(code_cells), 'errors': len(errors),
          'embedded_png_outputs': png_outputs, 'required_sections': 13,
          'dataset_hashes_verified': len(manifest), 'top_two_tuning_verified': expected_top_two,
          'test_metrics_recomputed_from_predictions': True, 'alert_logic_verified': True,
          'notebook_sha256': hashlib.sha256((root / 'SolarGuard.ipynb').read_bytes()).hexdigest()}
(out / 'artifact_review.json').write_text(json.dumps(review, indent=2))
print(json.dumps(review, indent=2))
