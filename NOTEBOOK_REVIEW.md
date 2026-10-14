# Notebook review - SolarGuard, Kelompok 7

Reference: `Predictive Analytics Group Project Brief and Assessment Rubric Revised.pdf`, all 11 pages, especially notebook sections and minimum regression models on pp. 4-5, validation/tuning/insights/reproducibility on p. 6, rubric on pp. 10-11. Implementation follows `planning/PLAN.md`.

This review distinguishes implemented notebook evidence from administrative and group-delivery requirements. It is not a predicted mark. On 14 October 2026, the team confirmed registration, uniqueness among groups and non-reuse of any member's midterm dataset in the project conversation. This is team confirmation, not an independent registry audit; individual contributions/ownership still require confirmation.

## Evidence map

| Rubric criterion | Weight | Notebook evidence | Remaining boundary |
|---|---:|---|---|
| 1. Problem formulation / suitability | 10 | Sections 1-2: task, horizon 0, stakeholder, target/unit, data dimensions, source/version and ethics; registry/uniqueness/non-UTS confirmed by team | Archive registry evidence; original plant owner and physical units not identified by retrieved metadata; source rights require team verification |
| 2. Data quality / preparation | 15 | Sections 3-4: schema, missing values, keys, types, invalids, gaps, coverage, safe join, development target/scale/outlier audit and row-flow | Original scale retained, no unverified correction; observed-row eligibility does not reconstruct missing generation |
| 3. EDA / features | 10 | Sections 5-6: distributions, correlations, time profiles, inverter spread, temperature gap, one-hot identities, train-fold imputation/scaling | No ablation claim; correlated and mechanically linked features complicate interpretation |
| 4. Baseline / candidates | 15 | Sections 7-8: mean baseline, linear, Ridge, RF and GB on same full-data chronological folds | Pooled mean is a minimal benchmark; linear additive representation has no plant-by-weather interactions, so comparison does not prove general algorithm superiority |
| 5. Validation / tuning / selection | 20 | Sections 7-10: grouped-date CV, top-two ranking, >=2 hyperparameters each, search spaces, best params, runtime, default/tuned comparison, pre-test selection lock | Four sampled ensemble configurations is a limited systematic search, not global optimization; CV reused for ranking/search is optimistic |
| 6. Interpretation / insights / implications | 15 | Sections 10-13: diagnostics, plant-level errors, permutation attribution, four findings and two feasible recommendations, risk/limits | Horizon 0 is not next-day forecasting; no fault labels, causal effect, ROI, fault precision/recall or confirmed false-alarm claims |
| 7. Reproducibility / quality | 5 | Section 14: exact local files/hashes, relative paths, imports, versions, seeds, computation checks, visible outputs | Execution result and artifact verification recorded below; reviewer must install listed dependencies in selected kernel |
| 8. Presentation / teamwork / ownership | 10 | Group identities and interview/submission reminders; reusable outputs exported | Slides, actual contributions, panel participation and individual interview ownership remain outside notebook execution |

## Technical review decisions

1. Whole dates, not panel rows, define holdout and folds. All plants/inverters on each date share the partition. Training must end before validation/test starts. Validation IDs are checked as a subset of training IDs; missing inverter coverage is reported rather than assumed identical.
2. Target/proxies are explicitly excluded from feature lists. Every imputer, encoder and scaler belongs to a per-fold pipeline. No target imputation, IQR target clipping or backward-fill is used.
3. Test is first predicted only after candidate ranking, tuning, final choice and OOF alert thresholds have been frozen in `outputs/selection_lock.json`. Post-selection test interpretation is descriptive; it does not alter model/thresholds.
4. DC/AC ratios differ substantially between plants. Units are not silently converted; raw-scale pooled MAE is explicitly a numerical benchmark, with plant-level metrics and development-mean normalization. It is not a shared physical or financial loss.
5. Daytime means irradiation > 0, not astronomically verified daylight. Very small positive sensor readings can include unusual clock times; the prototype minimum expected-output gate mitigates low-output alerts but does not validate the sensor clock or a new daylight threshold.
6. Per-file missing cells and missing *intervals* are distinct. Generation gaps, sensor gaps, left-join failures and eligibility exclusions are all recorded. A high match percentage alone is not complete time coverage.
7. Three consecutive 15-minute observations span 30 minutes between timestamps and represent three interval readings. Timestamp gaps reset streaks; each streak crossing 3 creates one review episode. Alert rate is not a measured false-alarm rate.
8. Source metadata is public Kaggle version 1, uploader Ani Kannal. Original facility owner and physical units are not substantiated; licensing is recorded verbatim without treating it as unrestricted reuse. Exact source files remain on disk despite Git ignore rules.

## Result interpretation

The development ranking chooses Gradient Boosting and Random Forest for tuning. Four configurations per model are evaluated on the same five grouped-date folds, using all eligible training rows. RF's best configuration uses 100 trees, depth 20, leaf size 2 and max_features=1.0. Its CV MAE changes from 294.844 to 290.620 (1.433% improvement). GB's sampled search worsens CV MAE from 294.003 to 354.049; the default GB remains eligible in final selection. The lower-CV-MAE rule selects tuned RF; the small difference is not claimed statistically significant relative to fold variation.

Test covers 11-17 June 2020 (16,588 eligible rows across 7 days). Pooled MAE is about 196.019, RMSE 502.776 and R² 0.978851. Crucially, Plant 1 R² is about 0.964396 and Plant 2 R² is 0.390257. MAE divided by the development plant mean is about 5.04% versus 23.08%; Plant 2's lower absolute-unit MAE does not imply greater reliability. The pooled R² is insufficient as a deployment argument. Further investigation of Plant 2's zero-output mismatches is appropriate, but without labels it cannot identify cause or fault accuracy.

Permutation sensitivity is largest for plant identity because of the scale difference; permuting plant independently from inverter creates inconsistent identity combinations. Irradiation and its square share information. These numbers are diagnostics, not operational interventions or causal effects. Residual persistence (contiguous within-inverter lag-one correlations about 0.596 and 0.771) and error extremes are disclosed. The proposed pilot retains human review and requires engineering verification and inspection outcomes.

## Execution history and review integrity

The first full attempt reached final model evaluation, then a recommendation-reporting string caused a syntax error. The string was corrected without changing features, split/folds, model candidates, hyperparameter spaces/budget, selection rule or alert policy. A subsequent clean-kernel attempt was terminated by the execution environment. The final verification reruns the unchanged analytical protocol from a fresh kernel. `outputs/initial_selection_lock.json` retains the original pre-test model/threshold choice; the artifact checker compares it with the reproduced choice so the reporting fix is not used to reselect on test results.

The authoring builder now parses every code cell before generating the notebook. Report-only Markdown explaining the observed results can be synchronized without changing executable cells or outputs; the sync helper rejects code differences. Final execution/verification status is recorded below only after it succeeds.

An isolated dry-run of the corrected recommendation/alert-reporting cell, using saved predictions and a temporary output directory, passed before the full rerun reached that cell. It reported 3 prototype review episodes at Plant 1 and 10 at Plant 2 over 7 test days, and rendered the four-row evidence-to-action matrix. These are workload counts, not confirmed fault counts. The thresholds are deliberately provisional and may miss smaller or persistent underperformance; sensitivity cannot be established without labeled inspection records.

## Final verification - passed, 14 October 2026

- Fresh-kernel execution completed successfully: **32 cells, 15 executable cells, 0 cell errors**. Execution counts are consecutive; all outputs are stored in `SolarGuard.ipynb`.
- End-to-end execution helper time: **630.1 seconds**; timed analysis inside the notebook: **622.5 seconds**. Reference Python: 3.12.10; exact package versions are in `requirements.txt` and `outputs/run_summary.json`. This was an equivalent local Python/Jupyter environment, not a claimed Google Colab run.
- All **13 required analytical sections** are present, plus reproducibility/review. Notebook has **11 embedded figure outputs** and visible audit/model/reporting tables.
- `scripts/review_notebook.py` passed: valid notebook schema, all code cells parse and are executed, no error outputs, all four CSV hashes match, correct top-two tuning, >=2 tuned parameters, chronological fold ordering, final selection consistent with CV, and final test MAE/RMSE/R² independently recomputed from saved predictions.
- Initial and final selection locks match in model, features, estimator parameters and OOF alert thresholds. No test-driven model revision was introduced during the reporting fix or reproduction.
- Corrected alert reporting succeeded in the full run: **3 and 10 review episodes** across 7 days; four evidence-supported findings and two recommendations are rendered.
- Report-only Markdown was synchronized after execution; the sync helper confirmed that executable cell sources and their outputs were unchanged. Artifact SHA-256 is recorded in `outputs/artifact_review.json`.

**Conclusion:** the notebook implements the required analytical workflow and has passing execution evidence. The remaining qualifications are source provenance/units/rights, limited tuning coverage and generalization, and external submission/teamwork obligations described above. The review does not certify full marks, operational fault detection or production deployment readiness.

`Group7_SolarGuard_Notebook.zip` is a notebook-only handoff containing the executed report, four exact CSVs, metadata, reporting evidence, requirements and helper scripts. It is intentionally distinct from the final course package `Group7_SolarGuard_PredictiveAnalytics.zip`, which still requires the final slides and confirmed actual member contributions.

## Evidence still requiring the team

- Archive registry evidence with group/member details; registration, uniqueness and non-UTS have been confirmed by the team.
- Verification of original-source provenance, source-use/distribution terms, physical units and equipment ratings. If the lecturer rejects suitability, dataset replacement would require a new analysis.
- Actual per-member contribution record; `CONTRIBUTIONS.md` intentionally does not invent contributions.
- Final slides and rehearsal, stakeholder-panel preparation/participation, interview readiness, full ZIP and RISE upload before 19 October 2026 at 20.00 WIB.

The notebook is a defensible proof-of-concept for contemporaneous output estimation and operator-reviewed inspection triage. It does not establish fault detection effectiveness, production readiness or financial return.
