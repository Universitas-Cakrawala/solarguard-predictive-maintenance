"""Build the self-contained analytical notebook; does not execute models."""
from pathlib import Path
import ast
import nbformat as nbf
import textwrap

cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))

def code(s):
    source = textwrap.dedent(s).strip()
    ast.parse(source)
    cells.append(nbf.v4.new_code_cell(source))

md('''
# SolarGuard
## Weather-Conditioned Solar Output Estimation for Maintenance Triage

**Predictive Analytics — Cakrawala University | Kelompok 7**

| Nama | ID mahasiswa | Struktur kelompok |
|---|---|---|
| Titanio Yudista | 24120500031 | Ketua |
| Farisyah Aulia Hanum | 24120500027 | Anggota |
| Sayyid Abdul Aziz Haidar | 24120500005 | Anggota |

Laporan analitis ini mengikuti 13 section notebook pada brief dan `planning/PLAN.md`.
Bahasa penjelasan: Indonesia. Semua angka hasil di bawah dihitung dari empat CSV lokal,
bukan hasil contoh. Jalankan **Restart Kernel / Runtime → Run All** dari folder proyek.
Dependensi tercatat di `requirements.txt`; instalasi opsional dari terminal: `pip install -r requirements.txt`.
Notebook tidak mengunduh data atau membutuhkan kredensial saat dijalankan.

**Status administratif:** pada 14 Oktober 2026, tim mengonfirmasi melalui percakapan
bahwa dataset sudah terdaftar di s.id/listdatapred02, berbeda dari kelompok lain,
dan bukan dataset UTS salah satu anggota. Ini konfirmasi tim, bukan verifikasi registry
independen; simpan bukti entri bersama paket tugas. Kontribusi aktual setiap anggota
masih perlu dicatat. Identitas anggota bukan
klaim bahwa masing-masing telah mengerjakan kode ini. Lihat `CONTRIBUTIONS.md`.
''')
md('''
## 1. Project overview

**Tema:** Energy and Environment. **Stakeholder:** Solar Plant Operations Manager / COO.
Output mentah inverter berubah bersama cuaca; membandingkan output tanpa kondisi cuaca
dapat menghasilkan prioritas inspeksi yang keliru. Kita menguji estimasi output yang
diharapkan untuk mendukung pemeriksaan operator.

**Predictive question:** Dengan sensor cuaca yang tersedia pada timestamp t, identitas
plant/inverter, dan waktu, seberapa akurat DC_POWER pada t dapat diestimasi pada hari yang
lebih baru untuk inverter yang sudah dikenal?

**Task:** regression. **Target:** DC_POWER, unit asli dataset (unit fisik belum terverifikasi).
**Unit analisis:** satu inverter pada satu timestamp, nominal 15 menit. **Horizon:** 0;
sensor cuaca t harus tersedia saat estimasi. Actual DC_POWER baru dipakai sesudah estimasi
untuk menghitung gap. Ini tidak menguji forecast hari berikutnya, fault classification,
lead time sebelum kegagalan, maupun transfer ke plant baru.

Tujuan SolarGuard adalah pilot triage inspeksi: output estimasi dibandingkan aktual,
kemudian operator memeriksa gap persisten. Tanpa maintenance/failure logs, gap belum
membuktikan kerusakan atau manfaat ekonomi. Data publik berisi identitas alat, tanpa
identitas pribadi yang terlihat; risiko operasi berupa salah prioritas dan false alerts
tetap harus ditangani melalui human review.
''')
code('''
from pathlib import Path
import os
import sys
import json
import hashlib
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

RUN_START = time.perf_counter()
SEED = 42
np.random.seed(SEED)
DATA_DIR = Path('data')
OUT = Path('outputs')
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
# Single search worker + bounded estimator threads avoids nested parallelism.
WORKERS = min(4, os.cpu_count() or 1)
thread_limit = threadpool_limits(limits=WORKERS)
sns.set_theme(style='whitegrid', palette='colorblind', context='notebook')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'axes.titlesize': 12})
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

def show_table(frame, name):
    frame.to_csv(OUT / f'{name}.csv', index=False)
    display(frame)

def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

def note(text):
    display(Markdown(text))

def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'n': len(y), 'MAE': mean_absolute_error(y, pred),
            'RMSE': np.sqrt(mean_squared_error(y, pred)),
            'R2': r2_score(y, pred) if len(y) > 1 and np.var(y) > 0 else np.nan,
            'bias_actual_minus_pred': float(np.mean(y - pred))}

environment = {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn',
    'matplotlib', 'seaborn', 'scipy', 'nbformat', 'nbclient', 'ipykernel', 'threadpoolctl']}
display(pd.Series({'Python': sys.version.split()[0], 'OS': platform.platform(),
                   'CPUs available': os.cpu_count(), 'threads used': WORKERS, **environment}))
''')
md('''
## 2. Dataset description

Sumber: [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data),
uploader Ani Kannal, **versi 1** (rilis 18 Agustus 2020), diakses 14 Oktober 2026.
Endpoint metadata publik dan snapshot tersedia pada `data/source_metadata.json`.
Deskripsi repository menyatakan dua plant India, 34 hari, generation di tingkat inverter,
dan satu array sensor di tingkat plant. Pemilik fasilitas/original collector tidak
diidentifikasi di metadata tersebut. Lisensi tercantum **Data files © Original Authors**;
bukan pernyataan lisensi terbuka tanpa batas. Tim perlu memastikan ketentuan penggunaan
dan distribusi untuk submission; public availability tidak menggantikan pemeriksaan izin.

Satuan daya, temperatur, dan irradiation tidak dijelaskan di metadata yang diperoleh.
Grafik memakai **unit asli dataset**, tanpa koreksi faktor skala yang tidak terverifikasi.
Timestamp diperlakukan sebagai waktu lokal sumber tanpa timezone yang dikonfirmasi;
tidak dikonversi ke WIB. WIB hanya untuk administrasi tugas.

| Kolom | Definisi / peran |
|---|---|
| DATE_TIME | Timestamp observasi; format eksplisit berbeda untuk generation Plant 1 |
| PLANT_ID | ID plant asli; categorical, bukan besaran numeric |
| SOURCE_KEY generation | ID inverter; diubah nama menjadi INVERTER_ID |
| SOURCE_KEY weather | ID sensor; diubah menjadi SENSOR_ID; tidak join atas ID ini |
| DC_POWER | Target daya DC pada timestamp t, unit asli |
| AC_POWER | Daya AC bersamaan; proxy target, hanya untuk audit development |
| DAILY_YIELD / TOTAL_YIELD | Yield kumulatif; excluded dari fitur scope ini |
| AMBIENT_TEMPERATURE | Temperatur lingkungan pada t, fitur sensor |
| MODULE_TEMPERATURE | Temperatur modul pada t, fitur sensor |
| IRRADIATION | Irradiation pada t; fitur dan eligibility daytime |

Enam calon prediktor tersedia: tiga sensor cuaca, timestamp, plant ID, inverter ID.
Ukuran aktual dan hashes di bawah menjadi bukti kelayakan serta exact version.
''')
code('''
FILES = {f'Plant_{p}_{kind}_Data.csv':
         ('%d-%m-%Y %H:%M' if p == 1 and kind == 'Generation' else '%Y-%m-%d %H:%M:%S')
         for p in [1, 2] for kind in ['Generation', 'Weather_Sensor']}
raw, parsed, manifest_rows, schema_rows = {}, {}, [], []
for name, fmt in FILES.items():
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f'Missing exact dataset: {path}. See DATA_SOURCE.md.')
    df = pd.read_csv(path)
    raw[name] = df.copy()
    for col in df:
        schema_rows.append({'file': name, 'column': col, 'dtype_raw': str(df[col].dtype),
                            'missing': int(df[col].isna().sum())})
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format=fmt, errors='coerce')
    parsed[name] = df
    manifest_rows.append({'file': name, 'rows': len(df), 'columns': len(df.columns),
        'start': str(df.DATE_TIME.min()), 'end': str(df.DATE_TIME.max()),
        'days': df.DATE_TIME.dt.normalize().nunique(), 'plants': df.PLANT_ID.nunique(),
        'source_keys': df.SOURCE_KEY.nunique(), 'datetime_parse_failures': int(df.DATE_TIME.isna().sum()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
manifest = pd.DataFrame(manifest_rows)
show_table(manifest.drop(columns=['sha256']), 'dataset_dimensions')
show_table(pd.DataFrame(schema_rows), 'schema_missingness')
(OUT / 'data_manifest.json').write_text(json.dumps(manifest_rows, indent=2))
display(manifest[['file', 'sha256']])
assert sum(len(df) for n, df in parsed.items() if 'Generation' in n) >= 1000
assert all(df.DATE_TIME.notna().all() for df in parsed.values())
for name, df in parsed.items():
    print(name)
    display(df.head(2))
''')
md('''
## 3. Data-quality assessment

Audit global di section 2-4 dibatasi pada schema, keys, timestamps, missingness dan
aturan invalid yang telah ditentukan. Statistik distribusi target, korelasi, skala,
outlier dan keputusan modeling hanya pada development setelah split. Tidak mengisi
target missing dan tidak menghapus zero daytime yang bisa menjadi sinyal operasional.
Temperatur negatif tidak otomatis invalid. Infinity diperlakukan missing/invalid.
Exact duplicates dapat dihapus; duplicate keys yang konflik menghentikan eksekusi
agar tidak memilih record secara arbitrer.
''')
code('''
audit_rows, cadence_rows, coverage_rows = [], [], []
for name, df in parsed.items():
    is_gen = 'Generation' in name
    key = ['PLANT_ID', 'SOURCE_KEY', 'DATE_TIME'] if is_gen else ['PLANT_ID', 'DATE_TIME']
    numeric = df.select_dtypes(include='number')
    audit_rows.append({'file': name, 'exact_duplicates': int(df.duplicated().sum()),
        'duplicate_keys_after_exact_drop': int(df.drop_duplicates().duplicated(key).sum()),
        'missing_cells': int(df.isna().sum().sum()),
        'infinite_numeric_cells': int(np.isinf(numeric.to_numpy()).sum()),
        'missing_key_rows': int(df[key].isna().any(axis=1).sum()),
        'negative_DC_POWER': int((df.DC_POWER < 0).sum()) if is_gen else 0,
        'negative_IRRADIATION': int((df.IRRADIATION < 0).sum()) if not is_gen else 0})
    for (plant, source), part in df.groupby(['PLANT_ID', 'SOURCE_KEY']):
        ts = part.DATE_TIME.drop_duplicates().sort_values()
        delta = ts.diff().dropna().dt.total_seconds().div(60)
        cadence_rows.append({'file': name, 'plant': plant, 'source': source,
            'timestamps': len(ts), 'median_interval_minutes': delta.median(),
            'intervals_not_15_minutes': int(delta.ne(15).sum()),
            'max_interval_minutes': delta.max()})
        expected = int((ts.max() - ts.min()).total_seconds() / 900) + 1
        coverage_rows.append({'file': name, 'plant': plant, 'source': source,
            'observed': len(ts), 'expected_between_endpoints': expected,
            'absent_intervals': expected - len(ts), 'coverage_pct': 100 * len(ts) / expected})
audit = pd.DataFrame(audit_rows)
show_table(audit, 'raw_quality_audit')
cadence = pd.DataFrame(cadence_rows)
show_table(cadence.groupby('file', as_index=False).agg(
    median_interval_minutes=('median_interval_minutes', 'median'),
    non_15_minute_intervals=('intervals_not_15_minutes', 'sum'),
    max_gap_minutes=('max_interval_minutes', 'max')), 'cadence_summary')
coverage = pd.DataFrame(coverage_rows)
show_table(coverage.groupby('file', as_index=False).agg(
    absent_intervals=('absent_intervals', 'sum'),
    min_coverage_pct=('coverage_pct', 'min'), mean_coverage_pct=('coverage_pct', 'mean')),
    'coverage_summary')
coverage.to_csv(OUT / 'coverage_per_source.csv', index=False)
assert (audit.duplicate_keys_after_exact_drop == 0).all(), 'Conflicting keys require review'
assert (audit.missing_key_rows == 0).all(), 'Missing IDs/timestamps require review'
''')
md('''
### Leakage inventory

| Field | Availability / keputusan |
|---|---|
| Weather t, ID, waktu | Diizinkan untuk estimasi t, dengan asumsi sensor diterima tepat waktu |
| AC_POWER t | Excluded; proxy output bersamaan membuat estimasi tidak bermakna untuk monitoring |
| DAILY_YIELD / TOTAL_YIELD | Excluded; kumulatif terkait output, reset/availability belum memadai untuk scope |
| DC_POWER | Target, tidak pernah masuk input preprocessing |
| SENSOR_ID | Tidak digunakan: identitas sensor bukan identitas inverter |
| Statistik imputasi, encoding, scaling | Di-fit hanya pada training fold melalui pipeline |
| Lag/future values | Tidak digunakan; horizon 0 tanpa interpolasi masa depan |

Yield kumulatif tidak otomatis mengandung masa depan; alasan exclusion adalah
hubungannya dengan output dan ketidakjelasan timing untuk penggunaan yang dipilih.
''')
md('''
## 4. Data preparation

Generation dan weather digabung pada plant dan timestamp, dengan validasi many-to-one.
Kita menyimpan original plant ID dan memisahkan inverter ID dari sensor ID. Negative
sensor irradiation diubah menjadi missing dan dikeluarkan dari eligibility, target
negative/nonfinite dikeluarkan, sedangkan missing fitur lain ditangani dalam pipeline.
Daytime didefinisikan sebelum melihat hasil: irradiation valid > 0. Baris missing
irradiation dilaporkan sebagai unavailable, bukan malam. Tidak ada clipping target,
forward-fill, koreksi skala, atau pembuangan outlier valid.
''')
code('''
generations, weathers = [], []
for name, frame in parsed.items():
    df = frame.drop_duplicates().copy()
    df['PLANT_ID'] = df['PLANT_ID'].astype(str)
    df['SOURCE_KEY'] = df['SOURCE_KEY'].astype(str)
    if 'Generation' in name:
        generations.append(df.rename(columns={'SOURCE_KEY': 'INVERTER_ID'}))
    else:
        weathers.append(df.rename(columns={'SOURCE_KEY': 'SENSOR_ID'}))
gen = pd.concat(generations, ignore_index=True)
weather = pd.concat(weathers, ignore_index=True)
assert not gen.duplicated(['PLANT_ID', 'INVERTER_ID', 'DATE_TIME']).any()
assert not weather.duplicated(['PLANT_ID', 'DATE_TIME']).any()
merged = gen.merge(weather, on=['PLANT_ID', 'DATE_TIME'], how='left',
                   validate='many_to_one', indicator=True)
assert len(merged) == len(gen)
join_audit = merged.groupby('PLANT_ID', observed=True).agg(
    generation_rows=('DC_POWER', 'size'),
    matched_weather_rows=('_merge', lambda s: int(s.eq('both').sum())),
    missing_irradiation=('IRRADIATION', lambda s: int(s.isna().sum())))
join_audit['weather_match_pct'] = 100 * join_audit.matched_weather_rows / join_audit.generation_rows
show_table(join_audit.reset_index(), 'join_audit')
sensor_cols = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']
merged[sensor_cols] = merged[sensor_cols].replace([np.inf, -np.inf], np.nan)
merged.loc[merged.IRRADIATION < 0, 'IRRADIATION'] = np.nan
valid_target = np.isfinite(merged.DC_POWER) & merged.DC_POWER.ge(0)
valid_irr = merged.IRRADIATION.notna()
eligible = valid_target & valid_irr & merged.IRRADIATION.gt(0)
flow = pd.DataFrame([
    {'stage': 'raw generation', 'rows': len(gen)},
    {'stage': 'left join (no multiplication)', 'rows': len(merged)},
    {'stage': 'valid observed target', 'rows': int(valid_target.sum())},
    {'stage': 'valid target + available irradiation', 'rows': int((valid_target & valid_irr).sum())},
    {'stage': 'eligible daytime; zeros retained', 'rows': int(eligible.sum())}])
show_table(flow, 'cleaning_row_flow')
exclusions = merged.assign(invalid_target=~valid_target, unavailable_irr=~valid_irr,
    non_daytime=valid_irr & merged.IRRADIATION.le(0), eligible=eligible).groupby('PLANT_ID').agg(
    raw_rows=('DC_POWER', 'size'), invalid_target=('invalid_target', 'sum'),
    unavailable_irr=('unavailable_irr', 'sum'), non_daytime=('non_daytime', 'sum'),
    eligible_rows=('eligible', 'sum'))
exclusions['eligible_pct'] = 100 * exclusions.eligible_rows / exclusions.raw_rows
show_table(exclusions.reset_index(), 'eligibility_by_plant')
# Split dates come from structural timestamps before inspecting target distributions.
all_dates = np.sort(merged.DATE_TIME.dt.normalize().unique())
cut = int(len(all_dates) * 0.8)
dev_dates, test_dates = all_dates[:cut], all_dates[cut:]
panel = merged.loc[eligible].copy()
panel['DATE'] = panel.DATE_TIME.dt.normalize()
dev = panel.loc[panel.DATE.isin(dev_dates)].sort_values(
    ['DATE_TIME', 'PLANT_ID', 'INVERTER_ID']).reset_index(drop=True)
test = panel.loc[panel.DATE.isin(test_dates)].sort_values(
    ['DATE_TIME', 'PLANT_ID', 'INVERTER_ID']).reset_index(drop=True)
assert dev.DATE_TIME.max() < test.DATE_TIME.min()
assert set(dev.DATE).isdisjoint(test.DATE)
assert len(dev) >= 1000
show_table(pd.DataFrame([{'partition': name, 'rows': len(d), 'days': d.DATE.nunique(),
    'start': d.DATE_TIME.min(), 'end': d.DATE_TIME.max(),
    'plants': d.PLANT_ID.nunique(), 'inverters': d.INVERTER_ID.nunique()}
    for name, d in [('development', dev), ('locked test', test)]]), 'holdout_split')
show_table(dev[sensor_cols + ['DC_POWER']].isna().sum().rename('missing_development').reset_index(),
           'development_missing_features')
note(f'**Preparation:** {eligible.sum():,}/{len(merged):,} generation rows eligible '
     f'({100 * eligible.mean():.2f}%). Join unmatched: {merged._merge.eq("left_only").sum():,}. '
     'Coverage applies only to observed records: missing generation intervals are not reconstructed.')
''')
md('''
### Development-only scale, target and outlier audit

AC_POWER is retained in the audit frame only; model input excludes it explicitly.
DC/AC ratios below cannot identify a correction factor by themselves. We retain original
DC_POWER units and use a pooled model with plant identity, then report plant-level errors.
The primary objective is raw-scale pooled MAE (mean of fold MAEs), an empirical benchmark
that weights each observation equally but is numerically dominated by the larger-scale
plant. It is not a comparable physical loss or economic objective across plants.
This limitation is fixed before candidate comparison; plant-level MAE and normalized
MAE relative to development plant mean provide complementary evidence.
''')
code('''
scale_audit = dev.groupby('PLANT_ID').agg(n=('DC_POWER', 'size'), mean_dc=('DC_POWER', 'mean'),
    median_dc=('DC_POWER', 'median'), max_dc=('DC_POWER', 'max'),
    mean_ac=('AC_POWER', 'mean'), daytime_zero=('DC_POWER', lambda s: int(s.eq(0).sum())))
ratios = dev.loc[(dev.DC_POWER > 0) & (dev.AC_POWER > 0)].assign(
    DC_AC_ratio=lambda x: x.DC_POWER / x.AC_POWER).groupby('PLANT_ID').DC_AC_ratio.median()
scale_audit['median_DC_AC_ratio'] = ratios
show_table(scale_audit.reset_index(), 'development_scale_audit')
outlier_rows = []
for plant, part in dev.groupby('PLANT_ID'):
    for col in ['DC_POWER'] + sensor_cols:
        s = part[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        outlier_rows.append({'plant': plant, 'feature': col, 'min': s.min(), 'max': s.max(),
            'q01': s.quantile(.01), 'q99': s.quantile(.99),
            'IQR_flag_count': int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            'treatment': 'retain; diagnostic flag only'})
show_table(pd.DataFrame(outlier_rows), 'development_outlier_audit')
dev_full = merged.loc[merged.DATE_TIME.dt.normalize().isin(dev_dates)].copy()
dev_full['light'] = np.where(dev_full.IRRADIATION.isna(), 'unavailable',
                    np.where(dev_full.IRRADIATION > 0, 'daytime', 'non-daytime'))
show_table(dev_full.groupby(['PLANT_ID', 'light']).agg(n=('DC_POWER', 'size'),
    target_mean=('DC_POWER', 'mean'), zeros=('DC_POWER', lambda s: int(s.eq(0).sum()))).reset_index(),
    'development_day_night_target')
note('**Observed scale:** ' + '; '.join(f'Plant {p}: median positive DC/AC ratio '
    f'{ratios.loc[p]:.3f}' for p in ratios.index) +
    '. The inter-plant difference is not corrected without authoritative unit/rating metadata. '
    'All valid extremes and daytime zeros are retained.')
''')
md('''
## 5. Exploratory data analysis

All EDA below uses development only. Histograms/boxplots summarize variability; scatter
and within-plant correlations assess sensor relationships; clock profiles motivate
cyclic time encoding. Inverter differences are descriptive and may reflect coverage,
rating or equipment configuration. They do not diagnose faults.
''')
code('''
show_table(dev.groupby('PLANT_ID')[['DC_POWER'] + sensor_cols].describe().stack(level=0).reset_index(),
           'development_descriptive_statistics')
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for row, (plant, part) in enumerate(dev.groupby('PLANT_ID')):
    for col, feature in enumerate(['DC_POWER'] + sensor_cols):
        sns.histplot(part[feature], bins=40, ax=axes[row, col])
        axes[row, col].set(title=f'Plant {plant}: {feature}', xlabel='Original dataset units', ylabel='Rows')
save_plot('eda_distributions')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
correlations = []
for ax, (plant, part) in zip(axes, dev.groupby('PLANT_ID')):
    corr = part[['DC_POWER'] + sensor_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
    ax.set_title(f'Development correlations — plant {plant}')
    correlations.append({'plant': plant, 'irradiation_DC_corr': corr.loc['DC_POWER', 'IRRADIATION']})
save_plot('eda_correlations')
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (plant, part) in zip(axes, dev.groupby('PLANT_ID')):
    sample = part.sample(min(4000, len(part)), random_state=SEED)
    ax.scatter(sample.IRRADIATION, sample.DC_POWER, s=5, alpha=.25)
    ax.set(title=f'Plant {plant}: irradiation vs output', xlabel='Irradiation (original units)',
           ylabel='DC_POWER (original units)')
save_plot('eda_irradiation_power')
dev['CLOCK_HOUR'] = dev.DATE_TIME.dt.hour + dev.DATE_TIME.dt.minute / 60
hourly = dev.groupby(['PLANT_ID', 'CLOCK_HOUR']).DC_POWER.mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (plant, part) in zip(axes, hourly.groupby('PLANT_ID')):
    ax.plot(part.CLOCK_HOUR, part.DC_POWER)
    ax.set(title=f'Plant {plant}: average daytime profile', xlabel='Source-local clock hour',
           ylabel='Mean DC_POWER (original units)')
save_plot('eda_hourly_profile')
fig, axes = plt.subplots(2, 1, figsize=(15, 9))
for ax, (plant, part) in zip(axes, dev.groupby('PLANT_ID')):
    sns.boxplot(data=part, x='INVERTER_ID', y='DC_POWER', ax=ax, showfliers=False)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.set(title=f'Plant {plant}: inverter spread (fliers hidden for readability; retained in data)',
           xlabel='Inverter ID', ylabel='DC_POWER (original units)')
save_plot('eda_inverter_spread')
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (plant, part) in zip(axes, dev.groupby('PLANT_ID')):
    sample = part.sample(min(4000, len(part)), random_state=SEED)
    ax.scatter(sample.MODULE_TEMPERATURE - sample.AMBIENT_TEMPERATURE,
               sample.DC_POWER, s=5, alpha=.25)
    ax.set(title=f'Plant {plant}: temperature gap vs output', xlabel='Temperature gap (original units)',
           ylabel='DC_POWER (original units)')
save_plot('eda_temperature_gap')
show_table(pd.DataFrame(correlations), 'eda_irradiation_correlations')
peaks = hourly.loc[hourly.groupby('PLANT_ID').DC_POWER.idxmax()]
note('**EDA interpretation:** ' + '; '.join(f'Plant {r.plant}: irradiation/output Pearson r='
    f'{r.irradiation_DC_corr:.3f}' for r in pd.DataFrame(correlations).itertuples()) +
    '. This motivates retaining irradiation, but does not establish causality. '
    'Mean output peaks at ' + '; '.join(f'{r.CLOCK_HOUR:.2f} source-local hours (plant {r.PLANT_ID})'
    for r in peaks.itertuples()) + '. Clock encoding retains minutes. Temperature gap is evaluated '
    'as a predictive transformation; correlated weather/time effects prevent a causal overheating claim.')
''')
md('''
## 6. Feature engineering

Features are fixed before comparison: three sensors, fractional-hour sine/cosine,
temperature gap, irradiation × module temperature, irradiation squared, and one-hot
plant/inverter identity. Interaction/polynomial terms let linear models express some
nonlinearity; trees can capture further interactions. No claim of incremental feature
benefit without an ablation study. DAY_OF_WEEK is omitted because four weeks provide
little justification for an operational weekly cycle.

Numeric median imputation and categorical most-frequent imputation are fit within each
training fold. Linear/Ridge numeric scaling also stays inside the pipeline. Dense
one-hot input enables every estimator, including positive-constrained linear tuning.
Unknown IDs are encoded as all zero; this handles software input, not proven new-site
generalization. No target scaling or identity label encoding is used.
''')
code('''
def engineer(frame):
    out = frame.copy()
    clock = out.DATE_TIME.dt.hour + out.DATE_TIME.dt.minute / 60
    out['HOUR_SIN'] = np.sin(2 * np.pi * clock / 24)
    out['HOUR_COS'] = np.cos(2 * np.pi * clock / 24)
    out['TEMP_GAP'] = out.MODULE_TEMPERATURE - out.AMBIENT_TEMPERATURE
    out['IRR_TEMP_INTERACTION'] = out.IRRADIATION * out.MODULE_TEMPERATURE
    out['IRR_SQUARED'] = out.IRRADIATION ** 2
    return out

dev, test = engineer(dev), engineer(test)
NUMERIC = sensor_cols + ['HOUR_SIN', 'HOUR_COS', 'TEMP_GAP', 'IRR_TEMP_INTERACTION', 'IRR_SQUARED']
CATEGORICAL = ['PLANT_ID', 'INVERTER_ID']
FEATURES = NUMERIC + CATEGORICAL
assert not set(FEATURES) & {'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD', 'SENSOR_ID'}
X_dev, y_dev = dev[FEATURES], dev.DC_POWER

def make_pipeline(estimator):
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if isinstance(estimator, (LinearRegression, Ridge)):
        numeric_steps.append(('scale', StandardScaler()))
    prep = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)
    ], sparse_threshold=0)
    return Pipeline([('prep', prep), ('model', estimator)])

display(pd.DataFrame({'feature': FEATURES, 'type': ['numeric'] * len(NUMERIC) + ['categorical'] * 2}))
''')
md('''
## 7. Baseline model and chronological validation

The last 20% of unique dates are a locked test (rounding to whole dates); earlier dates
form development. Five expanding-window CV folds partition **dates**, not panel rows.
All inverters/plants on a date move together. Gap=0 is appropriate to this contemporaneous
task without lags or overlapping future labels. Training history increases across folds.

Mean baseline learns only training target mean. **Primary:** mean fold MAE; **secondary:**
RMSE, with R² supporting. MAPE is omitted because zeros/near-zeros occur. Fair comparison
uses identical folds/rows/features for all candidates. Fold dispersion is descriptive
uncertainty, not an iid confidence interval.
''')
code('''
cv_splits, fold_rows = [], []
ordered_dates = np.sort(dev.DATE.unique())
for fold, (tr_d, va_d) in enumerate(TimeSeriesSplit(n_splits=5).split(ordered_dates), 1):
    tr = np.flatnonzero(dev.DATE.isin(ordered_dates[tr_d]).to_numpy())
    va = np.flatnonzero(dev.DATE.isin(ordered_dates[va_d]).to_numpy())
    assert dev.iloc[tr].DATE_TIME.max() < dev.iloc[va].DATE_TIME.min()
    assert set(dev.iloc[tr].DATE).isdisjoint(dev.iloc[va].DATE)
    assert set(dev.iloc[va].INVERTER_ID) <= set(dev.iloc[tr].INVERTER_ID)
    cv_splits.append((tr, va))
    fold_rows.append({'fold': fold, 'train_start': dev.iloc[tr].DATE.min(),
        'train_end': dev.iloc[tr].DATE.max(), 'validation_start': dev.iloc[va].DATE.min(),
        'validation_end': dev.iloc[va].DATE.max(), 'train_days': dev.iloc[tr].DATE.nunique(),
        'validation_days': dev.iloc[va].DATE.nunique(), 'train_rows': len(tr), 'validation_rows': len(va),
        'train_plants': dev.iloc[tr].PLANT_ID.nunique(), 'validation_plants': dev.iloc[va].PLANT_ID.nunique(),
        'train_inverters': dev.iloc[tr].INVERTER_ID.nunique(),
        'validation_inverters': dev.iloc[va].INVERTER_ID.nunique()})
show_table(pd.DataFrame(fold_rows), 'cv_folds')

def evaluate_cv(name, pipe):
    rows, plant_rows, predictions = [], [], []
    for fold, (tr, va) in enumerate(cv_splits, 1):
        model = clone(pipe)
        start = time.perf_counter()
        model.fit(X_dev.iloc[tr], y_dev.iloc[tr])
        fit_s = time.perf_counter() - start
        pred = model.predict(X_dev.iloc[va])
        rows.append({'model': name, 'fold': fold, **metrics(y_dev.iloc[va], pred), 'fit_seconds': fit_s})
        frame = dev.iloc[va][['PLANT_ID', 'INVERTER_ID', 'DATE_TIME', 'DATE', 'IRRADIATION']].copy()
        frame['actual'], frame['predicted'] = y_dev.iloc[va].to_numpy(), pred
        frame['fold'] = fold
        predictions.append(frame)
        for plant, idx in frame.groupby('PLANT_ID').indices.items():
            plant_rows.append({'model': name, 'fold': fold, 'plant': plant,
                **metrics(frame.iloc[idx].actual, frame.iloc[idx].predicted)})
    return pd.DataFrame(rows), pd.DataFrame(plant_rows), pd.concat(predictions, ignore_index=True)

baseline_pipe = make_pipeline(DummyRegressor(strategy='mean'))
base_folds, base_plant, base_oof = evaluate_cv('Mean baseline', baseline_pipe)
show_table(base_folds, 'baseline_cv_folds')
note(f'Mean baseline development CV MAE = **{base_folds.MAE.mean():,.3f}** '
     f'(fold SD {base_folds.MAE.std():,.3f}), in original target units. '
     'The baseline is a performance reference, not an inspection rule.')
''')
md('''
## 8. Candidate models

Linear Regression supplies an interpretable additive comparator. Ridge regularizes
correlated engineered features. Random Forest captures nonlinear interactions by
bagging trees. Gradient Boosting Regressor is the sklearn estimator, **not XGBoost**.
RF uses the estimator default 100 trees with fixed seed and bounded threads; GB uses
default structural parameters with fixed seed. All other defaults are displayed.

No test predictions occur here. Top two non-baseline candidates are ranked by CV MAE,
with alphabetical name as a deterministic tie-break only for exact ties. Fit timings
serve as a pilot to set search budget before tuning.
''')
code('''
models = {
    'Linear Regression': make_pipeline(LinearRegression()),
    'Ridge': make_pipeline(Ridge()),
    'Random Forest': make_pipeline(RandomForestRegressor(random_state=SEED, n_jobs=WORKERS)),
    'Gradient Boosting': make_pipeline(GradientBoostingRegressor(random_state=SEED))
}
all_folds, all_plants = [base_folds], [base_plant]
default_oof = {'Mean baseline': base_oof}
for name, pipe in models.items():
    print(f'Evaluating {name}...', flush=True)
    folds, plants, oof = evaluate_cv(name, pipe)
    all_folds.append(folds); all_plants.append(plants); default_oof[name] = oof
    print(f'  CV MAE {folds.MAE.mean():.3f}; fit total {folds.fit_seconds.sum():.1f}s', flush=True)
default_folds = pd.concat(all_folds, ignore_index=True)
default_plants = pd.concat(all_plants, ignore_index=True)
ranking = default_folds.groupby('model', as_index=False).agg(
    CV_MAE=('MAE', 'mean'), CV_MAE_SD=('MAE', 'std'), CV_RMSE=('RMSE', 'mean'),
    CV_R2=('R2', 'mean'), total_fit_seconds=('fit_seconds', 'sum')).sort_values(['CV_MAE', 'model'])
show_table(default_folds, 'default_cv_fold_scores')
show_table(ranking, 'default_cv_ranking')
show_table(default_plants.groupby(['model', 'plant'], as_index=False).agg(
    CV_MAE=('MAE', 'mean'), CV_MAE_SD=('MAE', 'std'), CV_RMSE=('RMSE', 'mean')),
    'default_cv_by_plant')
selected = ranking.loc[ranking.model.ne('Mean baseline'), 'model'].head(2).tolist()
assert len(selected) == 2
default_parameters = {n: p.named_steps['model'].get_params() for n, p in models.items()}
(OUT / 'default_parameters.json').write_text(json.dumps(default_parameters, indent=2, default=str))
display(default_parameters)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(ranking.model, ranking.CV_MAE, xerr=ranking.CV_MAE_SD, capsize=3)
ax.invert_yaxis(); ax.set(title='Default models: identical development folds',
    xlabel='Mean fold MAE ± fold SD (original target units)', ylabel='Model')
save_plot('default_cv_comparison')
note(f'**Selected for tuning:** {", ".join(selected)}. Selection uses development CV only. '
     'A large SD indicates variation across chronological windows; it is not evidence of a precise iid mean.')
''')
md('''
## 9. Hyperparameter tuning

Tune two relevant parameters or more for each of the two selected candidates. Search
uses identical five date folds and negative MAE scoring (converted to positive loss).
Linear: intercept and positivity; positivity constrains coefficients on scaled numeric
and one-hot features and is an empirical candidate, not a physical causal assumption.
Ridge: alpha and intercept. RF: number/depth of trees, leaf size and feature subsampling.
GB: stages, learning rate, tree depth, subsampling and leaf size.

Budget fixed after default pilot and before search: at most **4 sampled configurations**
per ensemble, consistent with the planning maximum 12, to keep full-data execution
reasonable. Small spaces use all combinations when <=4. No row sampling for fitting;
only figures/permutation diagnostics may sample. This is a limited systematic search,
not a claim of global optimality. Search workers=1; bounded estimator/BLAS threads.
The same CV scores are reused for ranking/search, so selected CV performance is optimistic;
the untouched test is the independent final check.
''')
code('''
spaces = {
    'Linear Regression': {'model__fit_intercept': [True, False], 'model__positive': [False, True]},
    'Ridge': {'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0], 'model__fit_intercept': [True, False]},
    'Random Forest': {'model__n_estimators': [100, 200, 300], 'model__max_depth': [10, 20, None],
        'model__min_samples_leaf': [1, 2, 4], 'model__max_features': [1.0, 0.5, 'sqrt']},
    'Gradient Boosting': {'model__n_estimators': [100, 200, 300], 'model__learning_rate': [0.03, 0.05, 0.1],
        'model__max_depth': [2, 3, 5], 'model__subsample': [0.8, 1.0], 'model__min_samples_leaf': [1, 4]}
}
searches, tuning_rows = {}, []
for name in selected:
    assert len(spaces[name]) >= 2
    budget = min(4 if name != 'Ridge' else 10, len(ParameterGrid(spaces[name])))
    print(f'Tuning {name}: {budget} configs x {len(cv_splits)} folds', flush=True)
    display(spaces[name])
    start = time.perf_counter()
    search = RandomizedSearchCV(models[name], spaces[name], n_iter=budget, cv=cv_splits,
        scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
        refit='mae', random_state=SEED, n_jobs=1, error_score='raise', return_train_score=False)
    search.fit(X_dev, y_dev)
    elapsed = time.perf_counter() - start
    searches[name] = search
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(OUT / (name.lower().replace(' ', '_') + '_search_results.csv'), index=False)
    idx = search.best_index_
    base = ranking.set_index('model').loc[name]
    tuned_mae = -results.loc[idx, 'mean_test_mae']
    tuning_rows.append({'model': name, 'configurations': budget, 'folds': len(cv_splits),
        'default_CV_MAE': base.CV_MAE, 'tuned_CV_MAE': tuned_mae,
        'tuned_CV_MAE_SD': results.loc[idx, 'std_test_mae'],
        'tuned_CV_RMSE': -results.loc[idx, 'mean_test_rmse'],
        'tuned_CV_R2': results.loc[idx, 'mean_test_r2'],
        'MAE_improvement': base.CV_MAE - tuned_mae,
        'improvement_pct': 100 * (base.CV_MAE - tuned_mae) / base.CV_MAE if base.CV_MAE else np.nan,
        'search_seconds': elapsed, 'best_parameters': json.dumps(search.best_params_)})
    print(f'Best CV MAE {tuned_mae:.3f}; runtime {elapsed:.1f}s', flush=True)
tuning = pd.DataFrame(tuning_rows)
show_table(tuning, 'default_vs_tuned')
fig, ax = plt.subplots(figsize=(10, 4))
tuning.set_index('model')[['default_CV_MAE', 'tuned_CV_MAE']].plot.bar(ax=ax)
ax.set(title='Default vs tuned on identical development folds', xlabel='Selected models',
       ylabel='CV MAE (original target units)'); ax.tick_params(axis='x', rotation=0)
save_plot('default_vs_tuned')
''')
md('''
## 10. Model evaluation and final selection

Final selection considers all default candidates, both tuned candidates and baseline,
using minimum development CV MAE. There is no mandatory preference for an ensemble or
tuned version. Exact ties prefer a default configuration, then name. All features,
preprocessing, parameters, metric and alert policy are frozen before test prediction.

Alert policy is calibrated on selected-model out-of-fold **development** predictions:
per plant, the 95th percentile of positive predicted−actual gaps, with minimum expected
output at the 25th percentile of OOF predictions and **3 consecutive 15-minute intervals**.
Only positive, above-threshold gaps count. Quantile and persistence are predeclared
prototype choices, not fault-optimal thresholds. Gaps in timestamps reset persistence.
The first development training window has no OOF predictions and is excluded from calibration.
''')
code('''
pool = []
for row in ranking.itertuples():
    pipe = baseline_pipe if row.model == 'Mean baseline' else models[row.model]
    pool.append({'candidate': row.model + ' [default]', 'model': row.model, 'variant': 'default',
                 'CV_MAE': row.CV_MAE, 'pipeline': pipe})
for row in tuning.itertuples():
    pool.append({'candidate': row.model + ' [tuned]', 'model': row.model, 'variant': 'tuned',
        'CV_MAE': row.tuned_CV_MAE, 'pipeline': searches[row.model].best_estimator_})
pool.sort(key=lambda x: (x['CV_MAE'], x['variant'] != 'default', x['candidate']))
chosen = pool[0]
show_table(pd.DataFrame([{k: v for k, v in x.items() if k != 'pipeline'} for x in pool]),
           'final_development_selection')
chosen_pipe = clone(chosen['pipeline'])
if chosen['variant'] == 'default':
    selected_oof = default_oof[chosen['model']].copy()
else:
    _, _, selected_oof = evaluate_cv(chosen['candidate'], chosen_pipe)
selected_oof['gap'] = selected_oof.predicted - selected_oof.actual
threshold_rows = []
for plant, part in selected_oof.groupby('PLANT_ID'):
    positives = part.loc[part.gap > 0, 'gap']
    threshold_rows.append({'PLANT_ID': plant,
        'gap_threshold': float(positives.quantile(.95)) if len(positives) else float('inf'),
        'min_expected': max(0.0, float(part.predicted.quantile(.25))),
        'persistence_intervals': 3, 'oof_rows': len(part), 'positive_gap_rows': len(positives)})
thresholds = pd.DataFrame(threshold_rows)
show_table(thresholds, 'development_alert_thresholds')
lock = {'selected': chosen['candidate'], 'selected_CV_MAE': float(chosen['CV_MAE']),
    'metric': 'mean_fold_MAE_original_scale', 'features': FEATURES, 'horizon': 0,
    'seed': SEED, 'parameters': chosen_pipe.named_steps['model'].get_params(),
    'thresholds': threshold_rows, 'locked_before_test_prediction': True,
    'utc_lock_time': datetime.now(timezone.utc).isoformat()}
(OUT / 'selection_lock.json').write_text(json.dumps(lock, indent=2, default=str))
note(f'**Frozen selection: {chosen["candidate"]}**, CV MAE {chosen["CV_MAE"]:,.3f}. '
     'This choice and alert policy are logged before any test prediction. '
     'Comparison was on the pooled original scale; plant-level limitations remain.')

# Final test begins here. No model/feature/threshold selection follows.
final_model = clone(chosen_pipe).fit(X_dev, y_dev)
X_test, y_test = test[FEATURES], test.DC_POWER
pred_final = final_model.predict(X_test)
baseline_final = clone(baseline_pipe).fit(X_dev, y_dev)
pred_baseline = baseline_final.predict(X_test)
final_scores = pd.DataFrame([
    {'model': chosen['candidate'], **metrics(y_test, pred_final)},
    {'model': 'Mean baseline', **metrics(y_test, pred_baseline)}])
show_table(final_scores, 'final_test_metrics')
test_report = test[['PLANT_ID', 'INVERTER_ID', 'DATE_TIME', 'DATE', 'IRRADIATION']].copy()
test_report['actual'], test_report['predicted'] = y_test.to_numpy(), pred_final
test_report['residual'] = test_report.actual - test_report.predicted
test_report['abs_error'] = test_report.residual.abs()
test_report['gap'] = -test_report.residual
test_report['hour'] = test_report.DATE_TIME.dt.hour
test_report['irradiation_band'] = pd.cut(test_report.IRRADIATION, [0, .2, .6, np.inf],
                                        labels=['(0,.2]', '(.2,.6]', '>.6'])
plant_means = dev.groupby('PLANT_ID').DC_POWER.mean()
plant_scores = []
for plant, idx in test_report.groupby('PLANT_ID').indices.items():
    part = test_report.iloc[idx]
    m = metrics(part.actual, part.predicted)
    m['MAE_over_development_mean'] = m['MAE'] / plant_means.loc[plant]
    plant_scores.append({'plant': plant, 'days': part.DATE.nunique(), **m})
show_table(pd.DataFrame(plant_scores), 'final_test_by_plant')
for column in ['INVERTER_ID', 'hour', 'irradiation_band', 'DATE']:
    rows = []
    for key, part in test_report.groupby(['PLANT_ID', column], observed=True):
        rows.append({'plant': key[0], column: key[1], **metrics(part.actual, part.predicted)})
    table = pd.DataFrame(rows)
    show_table(table, 'final_error_by_' + column.lower())
test_report.to_csv(OUT / 'final_test_predictions.csv', index=False)
test_gain = 100 * (final_scores.iloc[1].MAE - final_scores.iloc[0].MAE) / final_scores.iloc[1].MAE
note(f'**Independent final test:** selected MAE {final_scores.iloc[0].MAE:,.3f}, '
     f'RMSE {final_scores.iloc[0].RMSE:,.3f}, R² {final_scores.iloc[0].R2:.4f}; '
     f'MAE reduction vs mean baseline {test_gain:.2f}%. '
     'These values describe held-out daytime observations at known plants only. '
     f'Negative predictions: {(pred_final < 0).sum():,}; predictions are not post-hoc clipped.')
''')
code('''
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
autocorr_rows = []
for row, (plant, part) in enumerate(test_report.groupby('PLANT_ID')):
    sample = part.sample(min(5000, len(part)), random_state=SEED)
    lo = min(part.actual.min(), part.predicted.min()); hi = max(part.actual.max(), part.predicted.max())
    axes[row, 0].scatter(sample.actual, sample.predicted, s=5, alpha=.25)
    axes[row, 0].plot([lo, hi], [lo, hi], 'r--')
    axes[row, 0].set(title=f'Plant {plant}: actual vs predicted', xlabel='Actual (original units)',
                      ylabel='Predicted (original units)')
    axes[row, 1].scatter(sample.predicted, sample.residual, s=5, alpha=.25)
    axes[row, 1].axhline(0, color='red', linestyle='--')
    axes[row, 1].set(title='Residual vs prediction', xlabel='Predicted (original units)',
                      ylabel='Actual − predicted (original units)')
    timeline = part.groupby('DATE_TIME').residual.mean()
    axes[row, 2].plot(timeline.index, timeline.values, linewidth=.8)
    axes[row, 2].axhline(0, color='red', linestyle='--')
    axes[row, 2].tick_params(axis='x', rotation=30)
    axes[row, 2].set(title='Mean inverter residual over time', xlabel='Source-local timestamp',
                      ylabel='Mean residual (original units)')
    temp = part.sort_values(['INVERTER_ID', 'DATE_TIME']).copy()
    temp['prev_residual'] = temp.groupby('INVERTER_ID').residual.shift()
    temp['prev_time'] = temp.groupby('INVERTER_ID').DATE_TIME.shift()
    continuous = temp.DATE_TIME.sub(temp.prev_time).eq(pd.Timedelta(minutes=15))
    autocorr_rows.append({'plant': plant, 'contiguous_pairs': int(continuous.sum()),
        'pooled_within_inverter_lag1_corr': temp.loc[continuous, ['residual', 'prev_residual']].corr().iloc[0, 1]})
save_plot('final_test_diagnostics')
show_table(pd.DataFrame(autocorr_rows), 'test_residual_autocorrelation')
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (plant, part) in zip(axes, test_report.groupby('PLANT_ID')):
    sns.histplot(part.residual, bins=50, ax=ax)
    ax.set(title=f'Plant {plant}: residual distribution', xlabel='Actual − predicted (original units)', ylabel='Rows')
save_plot('test_residual_histogram')
show_table(test_report.nlargest(10, 'abs_error')[['PLANT_ID', 'INVERTER_ID', 'DATE_TIME',
    'IRRADIATION', 'actual', 'predicted', 'residual']], 'largest_test_errors')
note('**Diagnostic interpretation:** inspect plant-wise bias, low-light/peak errors and persistent '
    'residuals above rather than requiring normality. Lag-one correlation uses only contiguous '
    '15-minute pairs within the same inverter; temporal dependence means row-iid uncertainty '
    'would be misleading. Error extremes are retained, not removed after test inspection. '
    'Test diagnostics inform limitations and future studies, not a new model choice in this run.')
''')
md('''
## 11. Model interpretation

Interpretation is descriptive after freezing selection. Built-in tree importance has
split/cardinality bias; linear coefficient magnitudes depend on scaling and correlated
features. Permutation importance below uses **negative MAE**, so a positive increase in
loss indicates useful predictive information. Repeated permutations report sensitivity
to random permutation, not a generalization confidence interval.

Correlated sensors and deterministic derived features can share/substitute importance;
permuting one feature can produce unrealistic combinations. Neither importance nor
coefficient is a causal effect or percentage of target variance explained. PDP is omitted
because temperature gap and interaction features are mechanically linked, making naive
independent interventions misleading. Inverter identity may absorb chronic poor output;
a small residual does not establish a healthy inverter.

### Interpretation of the delivered run

Random Forest tuning changes CV MAE from about 294.84 to 290.62 (1.43% improvement);
Gradient Boosting's sampled tuned candidates are worse than its default (about 354.05
vs 294.00). The minimum-CV rule selects tuned Random Forest, but the small advantage
relative to fold variability is not evidence of statistically significant superiority.

Held-out pooled R² is about 0.979, while **Plant 1 R² is 0.964 and Plant 2 R² is only 0.390**.
MAE relative to development plant mean is about 5.04% and 23.08%, respectively; this
is normalized MAE, not row-wise MAPE. Plant 2 therefore needs greater caution despite
its smaller absolute-unit MAE. The actual-predicted figure shows substantial mismatches
around zero actual output and more scatter at Plant 2. Those are inspection candidates,
not labeled faults. Contiguous residual lag-one correlations around 0.596 and 0.771
show persistence; normal/iid residual assumptions would be inappropriate.

PLANT_ID has the largest pooled permutation sensitivity, reflecting the large scale
difference. Permuting plant while leaving inverter ID unchanged also creates inconsistent
identity combinations; its magnitude cannot be treated as a clean operational effect.
IRRADIATION and its square share predictive information, so their attribution is not
an additive causal decomposition. These diagnostics support separate plant reporting,
unit verification and an operator-reviewed pilot, rather than production automation.
''')
code('''
estimator = final_model.named_steps['model']
transformed_names = final_model.named_steps['prep'].get_feature_names_out()
if hasattr(estimator, 'feature_importances_'):
    builtin = pd.DataFrame({'feature': transformed_names, 'importance': estimator.feature_importances_})
    show_table(builtin.sort_values('importance', ascending=False).head(15), 'model_builtin_importance')
elif hasattr(estimator, 'coef_'):
    builtin = pd.DataFrame({'feature': transformed_names, 'coefficient': estimator.coef_})
    builtin['absolute_coefficient'] = builtin.coefficient.abs()
    show_table(builtin.sort_values('absolute_coefficient', ascending=False).head(15), 'model_coefficients')
else:
    note('Selected estimator is a constant baseline; no learned feature attribution is available.')

# Fixed diagnostic sampling only; full test metrics above use every eligible row.
sample_idx = test_report.sample(min(3000, len(test_report)), random_state=SEED).index
perm = permutation_importance(final_model, X_test.loc[sample_idx], y_test.loc[sample_idx],
    scoring='neg_mean_absolute_error', n_repeats=5, random_state=SEED, n_jobs=1)
perm_table = pd.DataFrame({'feature': FEATURES, 'MAE_increase_mean': perm.importances_mean,
                           'permutation_SD': perm.importances_std}).sort_values('MAE_increase_mean', ascending=False)
show_table(perm_table, 'permutation_importance')
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(perm_table.feature, perm_table.MAE_increase_mean, xerr=perm_table.permutation_SD, capsize=3)
ax.invert_yaxis(); ax.set(title='Post-selection permutation sensitivity (fixed test sample)',
    xlabel='Increase in MAE ± permutation SD (original units)', ylabel='Input feature')
save_plot('permutation_importance')
note(f'Top permutation feature in this run: **{perm_table.iloc[0].feature}**, '
     f'mean MAE increase {perm_table.iloc[0].MAE_increase_mean:,.3f}. '
     'No features or parameters are changed following this diagnostic.')
''')
md('''
## 12. Actionable insights

The prototype below applies the already-locked development policy to all held-out
inverters. Three-interval persistence is 45 minutes of interval readings, with a 30-minute
timestamp span. Gaps/missing timestamps reset the streak. Count distinct review episodes,
not every continuing flagged row as a separate inspection. Alert workload is measurable;
false-positive rate, fault recall and downtime avoided require labels not supplied here.

The delivered run yields 3 review episodes at Plant 1 and 10 at Plant 2 across 7 test
days under the frozen thresholds. These numbers define a queue for operator review,
not the number of faulty inverters. The residual thresholds (about 1,577 and 917 original
units) are high, especially relative to Plant 2 output; smaller deviations may be missed.
Without inspection labels this study cannot quantify that sensitivity trade-off.

Weather join matches are 99.994% and 100%, yet the generation cadence audit still
finds 3,026 and 4,110 absent inverter intervals between source endpoints. One Plant 2
inverter has only about 72.15% endpoint-based coverage. Data-quality monitoring must
therefore include missing timestamps, not just missing cells or weather match rate.
''')
code('''
alerts = test_report.merge(thresholds, on='PLANT_ID', how='left', validate='many_to_one')
alerts = alerts.sort_values(['PLANT_ID', 'INVERTER_ID', 'DATE_TIME']).reset_index(drop=True)
alerts['candidate_alert'] = (alerts.gap.gt(alerts.gap_threshold) & alerts.gap.gt(0)
                            & alerts.predicted.ge(alerts.min_expected))
alerts['streak'] = 0
for _, idx in alerts.groupby(['PLANT_ID', 'INVERTER_ID']).groups.items():
    streak, prev = 0, None
    for i in idx:
        row = alerts.loc[i]
        adjacent = prev is not None and row.DATE_TIME - prev == pd.Timedelta(minutes=15)
        streak = (streak + 1 if adjacent else 1) if row.candidate_alert else 0
        alerts.loc[i, 'streak'] = streak
        prev = row.DATE_TIME
alerts['persistent_alert'] = alerts.streak.ge(3)
alerts['new_episode'] = alerts.streak.eq(3)
alert_summary = alerts.groupby('PLANT_ID').agg(rows=('actual', 'size'),
    candidate_rows=('candidate_alert', 'sum'), persistent_rows=('persistent_alert', 'sum'),
    review_episodes=('new_episode', 'sum'), days=('DATE', 'nunique'))
alert_summary['persistent_row_pct'] = 100 * alert_summary.persistent_rows / alert_summary.rows
alert_summary['episodes_per_day'] = alert_summary.review_episodes / alert_summary.days
show_table(alert_summary.reset_index(), 'test_alert_workload')
show_table(alerts.groupby(['PLANT_ID', 'INVERTER_ID'], as_index=False).agg(
    candidate_rows=('candidate_alert', 'sum'), persistent_rows=('persistent_alert', 'sum'),
    review_episodes=('new_episode', 'sum')).sort_values('review_episodes', ascending=False),
    'test_alerts_by_inverter')
alerts.to_csv(OUT / 'test_alert_predictions.csv', index=False)

findings = pd.DataFrame([
    {'finding': 'Model accuracy on newer dates',
     'evidence': f'{chosen["candidate"]}; test MAE={final_scores.iloc[0].MAE:.3f}, '
                 f'RMSE={final_scores.iloc[0].RMSE:.3f}, reduction vs baseline={test_gain:.2f}%',
     'action': 'Use actual-vs-expected monitoring in shadow mode, reviewed by operator',
     'stakeholder': 'Operations Manager', 'value': 'Condition-aware prioritization',
     'limitation': 'Horizon 0, known plants, not failure prediction'},
    {'finding': 'Inter-plant measurement scale differs',
     'evidence': '; '.join(f'{p}: median DC/AC={v:.3f}' for p, v in ratios.items()),
     'action': 'Verify units, ratings and sensor mappings before common physical thresholds',
     'stakeholder': 'Plant engineer / data team', 'value': 'Avoid misleading cross-site priorities',
     'limitation': 'Ratio is a diagnostic, not proof of a correction factor'},
    {'finding': 'Sensor availability constrains monitoring coverage',
     'evidence': '; '.join(f'{p}: weather match={r.weather_match_pct:.2f}%'
                           for p, r in join_audit.iterrows()),
     'action': 'Log sensor availability and missing intervals; route unavailable periods to data-quality review',
     'stakeholder': 'Data/SCADA team', 'value': 'Distinguish unavailable monitoring from low output',
     'limitation': 'Generation missing intervals also reduce observable coverage'},
    {'finding': 'Persistent gap produces a concrete review workload',
     'evidence': '; '.join(f'{p}: {int(r.review_episodes)} episodes / {int(r.days)} days'
                           for p, r in alert_summary.iterrows()),
     'action': 'Pilot a review queue with capacity agreed with operations; attach inspection outcomes',
     'stakeholder': 'Maintenance supervisor', 'value': 'Measure feasibility before production rollout',
     'limitation': 'Unlabeled alerts are not confirmed faults or estimated false alarms'}])
show_table(findings, 'evidence_action_matrix')
note('**Recommendation 1:** Operations should pilot a human-reviewed residual triage queue at one plant, '
     'using the frozen threshold policy and the measured workload above. Start in shadow mode; '
     'record inspection outcomes, review time and confirmed issues before automating maintenance. '
     '**Recommendation 2:** Plant engineering/data teams should verify units and equipment ratings, '
     'instrument missing sensor/generation coverage, and maintain separate plant-level error/threshold '
     'reports. This follows the scale and availability findings rather than assuming raw readings '
     'are physically comparable. Agreement on coverage, review capacity and error tolerance is '
     'needed before any deployment approval.')
''')
md('''
## 13. Business implications, limitations and next steps

**Expected use.** Operations receives a condition-aware review queue, data/SCADA monitors
sensor availability, and maintenance supervisors attach inspection results. A pilot
should log input timestamps/latency, predictions, thresholds, missing periods, episode
IDs, review time, outcomes and costs. A separate dashboard/application is not required
for this assessment; notebook tables are sufficient evidence for the proposed workflow.

**Implementation.** Agree data-access interfaces, sensor latency, model execution budget,
ownership, escalation and human override with plant staff. Review error/drift by plant
and irradiation condition, plus alert workload, before choosing retraining cadence.
Monthly retraining is not claimed optimal. Pilot acceptance needs stakeholder-approved
coverage/error limits and manageable episodes per day; no arbitrary project result is
treated as production certification. Integration cost and ROI are unknown without
staffing, downtime, energy price and maintenance-cost records.

**Limits.** Only 34 days and two plants; no seasonal/external-site validation. Sensor
readings are shared plant-level exposures across many inverter rows. Raw target scales
are inconsistent enough to require engineering verification; pooled MAE is not physical
or financial loss. Chronological folds/test share plant and inverter identities. CV
selection/search reuse folds and can be optimistic. Sensor missingness and daytime-only
eligibility limit coverage; no prediction is validated for missing irradiation or night.
Correlated/engineered features complicate attribution. Inverter ID can normalize chronic
underperformance. No maintenance labels, failure dates, confirmed false alarms, equipment
ratings, or weather forecasts are supplied. Associations do not establish causation.
Public provenance identifies uploader but not the original facility owner; administrative
uniqueness and non-UTS have been confirmed by the team on 14 October 2026; registry evidence
has not been independently inspected in this notebook.

**Next steps.** Verify metadata/licensing and source units, collect >=1 year with equipment
ratings and maintenance logs, validate the queue with operator outcomes and costs, and
evaluate plant transfer separately. Next-day generation forecasting would require a new
prediction origin/horizon, available weather forecasts (including a defensible module
temperature estimate), forecasting baselines and horizon-specific backtesting. A
before-after pilot alone cannot attribute changes to maintenance without addressing
weather/operations confounding.
''')
md('''
## 14. Reproducibility and assessment self-review

This notebook includes all 13 required analytical sections plus this verification block.
CSV hashes fix the actual dataset version; inputs use one relative data path. No hidden
download, login, earlier-session variables or external module is required. The notebook
writes reporting tables/figures under `outputs/` for slide reuse. Full model fitting uses
all eligible rows; fixed sampling is limited to graphics and post-selection attribution.
Notebook execution automatically runs the checks below. A clean-kernel verification and
rubric review are recorded separately in `NOTEBOOK_REVIEW.md`.

**Submission boundary:** this task covers the notebook and supporting data/reproducibility.
The full assignment also needs final slides, actual per-member contributions, archived
registry evidence (registration / uniqueness / non-UTS confirmed by the team),
stakeholder-panel participation and interview ownership.
Do not claim these from notebook execution. Deadline: 19 October 2026, 20.00 WIB; technical
interview: 26 October 2026. Group 7 presents third in Meeting 15 (panel Group 3), and serves
as panel for Group 3 in Meeting 14.
''')
code('''
checks = {
    'four exact CSV files loaded': len(manifest) == 4,
    'minimum tabular size and predictors': len(panel) >= 1000 and len(FEATURES) >= 5,
    'no target/proxy in features': not set(FEATURES) & {'DC_POWER','AC_POWER','DAILY_YIELD','TOTAL_YIELD'},
    'no join multiplication': len(merged) == len(gen),
    'strict chronological holdout': dev.DATE_TIME.max() < test.DATE_TIME.min(),
    'five grouped-date folds': len(cv_splits) == 5,
    'all required model families': {'Linear Regression','Random Forest','Gradient Boosting'} <= set(models),
    'two strongest non-baseline tuned': set(searches) == set(selected) and len(searches) == 2,
    'at least two tuning parameters each': all(len(spaces[n]) >= 2 for n in selected),
    'test predictions finite': np.isfinite(pred_final).all(),
    'three findings and two recommendations': len(findings) >= 3,
    'thresholds locked from development': thresholds.oof_rows.sum() == len(selected_oof),
    'per-plant test reporting': len(plant_scores) == dev.PLANT_ID.nunique()
}
assert all(checks.values()), checks
show_table(pd.DataFrame({'check': list(checks), 'passed': list(checks.values())}), 'verification_checks')
summary = {'selected_model': chosen['candidate'], 'test': final_scores.to_dict('records'),
    'test_MAE_reduction_vs_baseline_pct': float(test_gain),
    'rows_generation': len(gen), 'rows_daytime': len(panel), 'rows_development': len(dev),
    'rows_test': len(test), 'days_development': dev.DATE.nunique(), 'days_test': test.DATE.nunique(),
    'tuned_models': selected, 'run_seconds': time.perf_counter() - RUN_START,
    'environment': environment, 'checks': checks, 'utc_completed': datetime.now(timezone.utc).isoformat()}
(OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(f'Completed full notebook in {summary["run_seconds"]:.1f} seconds; all checks passed.')
note('**Review boundary:** passing these computational checks is evidence for the notebook, '
     'not a score guarantee. Registration, original-source suitability, rights, individual '
     'ownership and complete submission remain separately verifiable requirements.')
''')

if __name__ == '__main__':
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.10'}})
    nbf.validate(nb)
    nbf.write(nb, Path('SolarGuard.ipynb'))
    print(f'Built SolarGuard.ipynb: {len(cells)} cells')
