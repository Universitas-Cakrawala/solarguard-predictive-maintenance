# Dataset source and reproducibility

Title: Solar Power Generation Data. Repository uploader: Ani Kannal.

Source: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

Public metadata endpoint: https://www.kaggle.com/api/v1/datasets/view/anikannal/solar-power-generation-data

Download endpoint: https://www.kaggle.com/api/v1/datasets/download/anikannal/solar-power-generation-data

Access date: 14 October 2026 (Asia/Jakarta). Metadata identifies version 1, initial release 18 August 2020. The exact downloaded files are in `data/`; the notebook produces SHA-256, byte sizes, dimensions and time coverage in `outputs/data_manifest.json`.

The repository description states that observations were gathered at two solar plants in India over 34 days. Generation is measured per inverter and weather comes from a plant-level sensor array. The metadata does not identify the original facility owner/collector or give authoritative physical units. Ani Kannal is credited as repository uploader, not assumed to own the plants. No physical-unit conversion is applied. The notebook labels values in original dataset units, audits DC/AC scales on development only, and reports each plant separately.

Metadata license: **Data files © Original Authors**. Public access does not establish unlimited redistribution rights. Confirm course-submission and redistribution terms with the source if necessary; do not replace this with an invented open license. Do not publish the CSV files as an unrestricted dataset.

`data/source_metadata.json` preserves the publicly retrieved source description, version and license. No credentials were used. Exact CSV names:

- `Plant_1_Generation_Data.csv`
- `Plant_1_Weather_Sensor_Data.csv`
- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

Notebook execution is offline: it reads `data/` and does not call Kaggle. Preserve these exact files with the notebook in the assignment package. `outputs/data_manifest.json` identifies the bytes actually used. Raw CSVs are ignored by the existing Git configuration; they are present on disk and must still be included in the final course ZIP.

On 14 October 2026 the team confirmed in the project conversation that this dataset is registered at s.id/listdatapred02, unique among groups, and not reused from any member's midterm. This records team confirmation; the registry itself was not independently inspected. Preserve the registration evidence in the course package. These administrative facts are not inferred from code execution.
