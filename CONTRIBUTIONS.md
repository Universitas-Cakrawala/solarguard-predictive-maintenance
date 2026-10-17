# Kelompok 7 - Statement of Contribution & Roles

Dokumen ini memuat konfirmasi kontribusi dan peran individual seluruh anggota Kelompok 7 dalam perancangan, eksplorasi data, pemodelan analitik prediktif, serta penyusunan presentasi proyek **SolarGuard (Predictive Maintenance Solar Power Plant)**.

---

## Ringkasan Kontribusi Anggota

| Anggota | NIM | Peran Tim | Fokus Kontribusi Utama |
|---|---|---|---|
| **Titanio Yudista** | 24120500031 | Ketua Kelompok | **End-to-End Pipeline, Model Engineering & Slide Development**<br>• Perancangan & implementasi arsitektur analitik di `SolarGuard.ipynb`<br>• Pipeline validasi temporal, cross-validation, & hyperparameter tuning<br>• Pengembangan aplikasi presentasi interaktif di Open Slide |
| **Farisyah Aulia Hanum** | 24120500027 | Anggota | **Storyline Concept, Visual Narrative & Strategic Synthesis**<br>• Konseptualisasi narasi presentasi dan alur komunikasi bisnis (*pitch storyline*)<br>• Sintesis temuan analitik (EDA, diagnostik residu, dan *evidence-action matrix*)<br>• Perumusan skenario interaktif simulasi triage dan batas operasional |
| **Sayyid Abdul Aziz Haidar** | 24120500005 | Anggota | **Problem Scoping, Data Sourcing & Domain Research**<br>• Inisiasi pemilihan domain energi surya dan penentuan dataset observasi<br>• Scoping awal parameter sensor cuaca (*irradiation, ambient & module temperature*)<br>• Riset domain kelayakan data pemeliharaan prediktif dan evaluasi awal |

---

## Rincian Tanggung Jawab & Luaran Kerja

### 1. Titanio Yudista (24120500031) — *Lead Developer & Modeler*
- **Implementasi Analitik & Rekayasa Kode**: Mengembangkan seluruh pipeline kode pada notebook analitik (`SolarGuard.ipynb`), mulai dari pembersihan data bersyarat (*daylight-only filtering*), join audit sensor cuaca, hingga penanganan *scaling* data mentah.
- **Validasi Temporal & Seleksi Model**: Membangun protokol evaluasi berbasis *5-fold expanding window cross-validation*, mengevaluasi performa model baseline (Linear Regression, Ridge) versus ensemble (Gradient Boosting, Random Forest), serta mengeksekusi penyetelan hiperparameter terarah menggunakan *RandomizedSearchCV*.
- **Authoring Slide Interaktif**: Menulis kode aplikasi presentasi (`slides/solarguard/index.tsx`) berbasis React/TypeScript dan Open Slide, termasuk komponen interaktif *Fold Explorer*, *Test Metric Switcher*, dan *Model Replay Simulator*.

### 2. Farisyah Aulia Hanum (24120500027) — *Presentation Strategist & Narrative Designer*
- **Konseptualisasi Alur Presentasi**: Mengembangkan kerangka komunikasi dan *storyline* presentasi agar ramah bagi eksekutif (*COO/Operations Manager*), menghubungkan metrik teknis (seperti $R^2$, MAE, dan error relatif) dengan keputusan riil di lapangan.
- **Sintesis Bukti Operasional**: Menyusun formulasi rekomendasi bisnis ke dalam matriks tindakan berbasis bukti (*Evidence-Action Matrix*), merancang skenario alokasi prioritas pilot untuk Plant 1 dan kebutuhan audit sensor untuk Plant 2.
- **Penyusunan Konten Visual**: Mengarahkan komposisi visual slide, anotasi grafik diagnostik, dan penulisan panduan pembicara (*speaker notes*) guna memastikan presentasi berlangsung terukur dalam batas waktu 11–12 menit.

### 3. Sayyid Abdul Aziz Haidar (24120500005) — *Data Ideation & Domain Analyst*
- **Inisiasi & Kurasi Dataset**: Mengusulkan penggunaan data pembangkit surya (*Solar Power Generation Dataset*) dan memetakan relevansi hubungan fisik antara radiasi matahari, suhu panel, dan degradasi output inverter terhadap kasus *predictive maintenance*.
- **Scoping Kebutuhan Data & Fitur**: Mengidentifikasi variabel kunci yang layak digunakan sebagai prediktor kontemporer (horizon 0) untuk menghindari *data leakage*, serta menelaah keterbatasan metadata awal terkait kejelasan satuan fisik (kW/W).
- **Riset Latar Belakang & Evaluasi**: Mendalami literatur dan acuan industri terkait pola kegagalan inverter surya, memberikan masukan analitis pada interpretasi korelasi fitur dan perbandingan performa antar-pembangkit.

---

## Komitmen Shared Ownership & Integritas Akademik

1. **Pemahaman Kolektif**: Meskipun terdapat pembagian fokus kerja, seluruh anggota telah meninjau alur kerja secara komprehensif (mulai dari pembersihan data, skema validasi, tuning hiperparameter, diagnostik residu, hingga rekomendasi operasional) dan siap mempertahankan argumen dalam *technical interview*.
2. **Kepatuhan Terhadap Bukti**: Seluruh pernyataan yang dibuat dalam laporan dan slide didasarkan secara ketat pada hasil kalkulasi empiris yang tersimpan di repositori, tanpa manipulasi klaim performa maupun proyeksi finansial yang tidak berdasar.
