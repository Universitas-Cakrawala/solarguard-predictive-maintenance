# SolarGuard - Kelompok 7

Weather-conditioned solar output estimation for maintenance triage. The task is contemporaneous regression (horizon 0) at known inverters, not validated failure prediction or next-day forecasting.

## Anggota Kelompok

| No | Nama | NIM | Peran |
|:--:|---|:--:|---|
| 1 | Titanio Yudista | 24120500031 | Ketua |
| 2 | Farisyah Aulia Hanum | 24120500027 | Anggota |
| 3 | Sayyid Abdul Aziz Haidar | 24120500005 | Anggota |

## Slide Presentation

- **Home**: https://solarguard-slide-presentation.vercel.app/
- **Slide**: https://solarguard-slide-presentation.vercel.app/s/solarguard

## Run the notebook

Use Python 3.12 and the versions in `requirements.txt`. The reference execution uses an existing Python environment; `.venv/` may require installation before use.

```bash
python -m pip install -r requirements.txt
```

Open `SolarGuard.ipynb` from this project root, select the prepared Python kernel, and choose **Restart Kernel / Runtime -> Run All**. Keep all four exact CSVs in `data/`. No Kaggle login or internet access is needed to execute. See `DATA_SOURCE.md` for source, metadata, licensing limitations and version details.

For an automated fresh-kernel run:

```bash
python scripts/execute_notebook.py
```

The notebook is self-contained. `scripts/build_notebook.py` is its editable authoring source; running the builder replaces the notebook and clears outputs, so execute again after rebuilding. The execution helper requires permission to start a local Jupyter kernel/socket.

## Deliverables

- `SolarGuard.ipynb`: analytical report with visible outputs after execution.
- `data/`: four original CSVs and public source metadata snapshot.
- `outputs/`: generated audit/CV/search/test tables, selection lock, figures and run summary.
- `NOTEBOOK_REVIEW.md`: final notebook verification and brief/rubric alignment.
- `CONTRIBUTIONS.md`: actual member contributions still require team confirmation.

`Group7_SolarGuard_Notebook.zip` bundles the executed notebook, exact data and supporting files for this notebook task. It is not the complete course-submission ZIP. Final verification: 15 executable cells passed from a fresh kernel, with 11 embedded figures and no cell errors; see `NOTEBOOK_REVIEW.md` for results and limitations.

The notebook also reports a supplemental training-only mean baseline per plant, on the same chronological folds and test rows. This comparison was added after test exposure and does not reselect the final model. Reported fold standard deviations consistently use ddof=1; fold-wise paired tuning differences are descriptive, not significance tests. See `outputs/baseline_cv_comparison.csv`, `outputs/paired_default_vs_tuned_folds.csv` and `outputs/final_test_baseline_comparison_by_plant.csv` after execution.

After any rerun or notebook edit, refresh verification and the archive in this order:

```bash
python scripts/review_notebook.py
python scripts/package_notebook.py
```

Packaging rejects a stale notebook hash and verifies every archived file against the workspace. `outputs/artifact_review.json` describes the current snapshot; initial selection, pre-strengthening predictions and language-revision records are historical evidence.

Raw CSVs and reporting CSVs follow the repository's existing Git ignore rules; include the data files explicitly in the final course ZIP. The team confirmed registration, dataset uniqueness and non-UTS on 14 October 2026; archive registry evidence separately. Slides, member contribution confirmation, stakeholder participation, interview and RISE upload are separate from notebook completion.

Submission deadline: **19 October 2026, 20.00 WIB**. Full package name: `Group7_SolarGuard_PredictiveAnalytics.zip`.
