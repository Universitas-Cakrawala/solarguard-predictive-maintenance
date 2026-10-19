# Peninjauan Notebook SolarGuard — Kelompok 7

## Acuan dan ruang lingkup peninjauan

Peninjauan ini membandingkan bukti dalam notebook dengan seluruh 11 halaman
`Predictive Analytics Group Project Brief and Assessment Rubric Revised.pdf`, terutama
ketentuan notebook dan model regresi minimum pada halaman 4–5, validasi, penyetelan,
temuan, dan reproduktibilitas pada halaman 6, serta rubrik pada halaman 10–11.

Dokumen ini membedakan bukti yang telah diterapkan dalam notebook dari persyaratan
administratif dan pelaksanaan kelompok. Dokumen ini bukan prediksi nilai. Pada
14 Oktober 2026, kelompok mengonfirmasi bahwa dataset telah didaftarkan, berbeda dari
kelompok lain, dan bukan dataset UTS salah satu anggota. Pernyataan tersebut merupakan
konfirmasi kelompok, bukan audit independen terhadap daftar registrasi.

## Kesimpulan peninjauan

Notebook telah menerapkan alur analisis yang diminta dan berhasil dijalankan dari
kernel bersih. Bukti eksekusi menunjukkan 32 sel, 15 sel kode, tanpa keluaran error,
dan 11 grafik tertanam. Seluruh 13 bagian analitis yang diwajibkan tersedia, beserta
bagian reproduktibilitas dan peninjauan.

Model final tetap Random Forest hasil penyetelan dengan MAE pengujian 196,019, RMSE
502,776, dan R² 0,978851. Penambahan pembanding rata-rata per pembangkit, penjelasan
statistik, dan sinkronisasi artefak tidak mengubah fitur, pembagian data, parameter
model terpilih, anggaran pencarian, ambang alert, maupun prediksi model final.

Notebook layak menjadi bukti konsep estimasi daya saat kondisi cuaca diketahui dan
triase inspeksi dengan tinjauan operator. Hasilnya belum membuktikan efektivitas
deteksi kerusakan, kesiapan produksi, hubungan sebab-akibat, atau keuntungan finansial.
Paket akhir mata kuliah juga masih memerlukan slide final dan catatan kontribusi nyata
setiap anggota.

## Peta bukti terhadap rubrik

| Kriteria rubrik | Bobot | Bukti dalam notebook | Batasan yang masih berlaku |
|---|---:|---|---|
| 1. Formulasi masalah dan kesesuaian | 10 | Bagian 1–2 menjelaskan tugas, horizon 0, pemangku kepentingan, target, dimensi data, sumber, versi, dan etika | Pemilik pembangkit dan satuan fisik tidak tersedia dalam metadata; hak penggunaan sumber perlu diverifikasi kelompok |
| 2. Kualitas dan persiapan data | 15 | Bagian 3–4 mencakup skema, nilai hilang, kunci, tipe, nilai tidak valid, celah waktu, cakupan, penggabungan aman, skala target, dan alur baris | Skala asli dipertahankan tanpa koreksi yang tidak terverifikasi; observasi yang hilang tidak direkonstruksi |
| 3. EDA dan fitur | 10 | Bagian 5–6 memuat distribusi, korelasi, profil waktu, variasi inverter, selisih suhu, dan rekayasa fitur | Tidak ada klaim ablasi; fitur yang berkorelasi dan saling terkait membatasi interpretasi |
| 4. Pembanding dan kandidat | 15 | Bagian 7–8 membandingkan rata-rata gabungan, rata-rata per pembangkit, Regresi Linear, Ridge, Random Forest, dan Gradient Boosting pada fold kronologis yang sama | Pembanding per pembangkit adalah analisis tambahan setelah data pengujian pernah dilihat; model linear aditif tidak memiliki interaksi pembangkit–cuaca |
| 5. Validasi, penyetelan, dan seleksi | 20 | Bagian 7–10 memakai CV berdasarkan tanggal, dua model terbaik, sedikitnya dua hiperparameter per model, ruang pencarian, parameter terbaik, perbandingan default–tuned per fold, simpangan baku sampel, dan kunci seleksi sebelum pengujian | Empat konfigurasi per ensemble merupakan pencarian terbatas; pemakaian CV untuk ranking dan pencarian dapat optimistis; eksekusi sekarang adalah reproduksi setelah data pengujian pernah dilihat |
| 6. Interpretasi, temuan, dan implikasi | 15 | Bagian 10–13 memuat diagnostik, galat per pembangkit, permutation importance, empat temuan, dua rekomendasi, risiko, dan keterbatasan | Horizon 0 bukan prediksi hari berikutnya; tidak tersedia label kerusakan, ROI, precision/recall kerusakan, atau alarm palsu terverifikasi |
| 7. Reproduksibilitas dan kualitas | 5 | Bagian 14 mencatat file, hash, path relatif, versi, seed, pemeriksaan komputasi, dan keluaran terlihat | Penilai perlu memasang dependensi pada kernel yang dipilih |
| 8. Presentasi, kerja tim, dan kepemilikan | 10 | Identitas kelompok serta pengingat presentasi dan submission tersedia; keluaran dapat dipakai untuk slide | Slide, kontribusi aktual, partisipasi panel, dan penguasaan materi saat wawancara berada di luar eksekusi notebook |

## Keputusan teknis utama

1. Pembagian holdout dan fold menggunakan tanggal utuh. Semua pembangkit dan inverter
   pada tanggal yang sama berada dalam partisi yang sama. Pelatihan selalu berakhir
   sebelum validasi atau pengujian dimulai.
2. Target dan proksinya tidak digunakan sebagai fitur. Imputer, encoder, dan scaler
   dipasang ulang di dalam pipeline setiap fold. Target tidak diimputasi, dipotong
   dengan IQR, atau diisi menggunakan data masa depan.
3. Pemilihan awal model, fitur, parameter, dan ambang alert dikunci sebelum prediksi
   pengujian pertama. Kunci awal tersedia dalam `outputs/initial_selection_lock.json`.
   Eksekusi saat ini mereproduksi keputusan tersebut setelah data pengujian pernah dilihat.
4. Rasio DC/AC berbeda besar antar pembangkit. Notebook tidak melakukan konversi satuan
   tanpa dasar. MAE gabungan pada skala asli disertai metrik per pembangkit dan
   normalisasi terhadap rata-rata development.
5. Siang didefinisikan sebagai irradiation lebih besar dari nol, bukan berdasarkan
   perhitungan astronomis. Nilai sensor positif yang sangat kecil dapat muncul pada
   jam yang tidak lazim.
6. Nilai sel yang hilang berbeda dari interval waktu yang hilang. Celah pembangkitan,
   celah sensor, kegagalan penggabungan, dan eksklusi kelayakan dicatat terpisah.
7. Tiga observasi berturut-turut dengan interval 15 menit membentang 30 menit antar
   timestamp. Celah timestamp mengatur ulang streak. Alert adalah episode untuk
   tinjauan operator, bukan kerusakan yang telah dikonfirmasi.
8. Metadata sumber mencatat dataset Kaggle publik versi 1 dengan pengunggah Ani Kannal.
   Pemilik fasilitas, satuan fisik, dan lisensi penggunaan ulang yang tidak terbatas
   tidak dapat disimpulkan dari metadata tersebut.

## Penguatan pembanding dan penjelasan statistik

`PlantMeanRegressor` mempelajari rata-rata target setiap pembangkit hanya dari bagian
pelatihan pada masing-masing fold. Kategori yang belum dikenal menggunakan rata-rata
gabungan pelatihan. Pembanding ini diuji pada lima fold dan baris yang sama tanpa
memilih ulang model final.

Rata-rata per pembangkit menghasilkan CV MAE 1.963,493 dan MAE pengujian 1.661,945.
Rata-rata gabungan menghasilkan CV MAE 3.349,278 dan MAE pengujian 3.070,496. Model final
menurunkan MAE pengujian sebesar 88,205% terhadap pembanding per pembangkit dan sekitar
93,62% terhadap pembanding rata-rata gabungan. Setiap klaim penurunan MAE harus selalu
menyebut pembandingnya. Perbandingan ini bersifat pelengkap karena ditambahkan setelah
data pengujian pernah diperiksa dan tidak boleh disebut evaluasi independen baru.

Seluruh simpangan baku pada tabel laporan dihitung sebagai simpangan baku sampel dengan
`ddof=1`. Kolom mentah `std_test_mae` dari scikit-learn tetap memakai `ddof=0`, sehingga
laporan menghitung ulang nilai dari skor setiap fold. Fold diberi bobot sama dalam
fungsi objektif, sedangkan bobot setiap baris development tidak persis sama karena ukuran
fold berbeda. Simpangan baku bukan galat baku, interval kepercayaan, atau bukti
signifikansi karena hanya terdapat lima fold dengan jendela pelatihan bertumpang tindih.

Model linear menggunakan one-hot pembangkit sehingga dapat memiliki intersep berbeda,
tetapi belum mempunyai kemiringan cuaca–daya yang berbeda untuk setiap pembangkit.
Model linear terpisah per pembangkit atau interaksi pembangkit–cuaca dapat menjadi
pengembangan berikutnya dan harus dinilai melalui evaluasi baru.

## Interpretasi hasil model

Ranking development memilih Gradient Boosting dan Random Forest untuk disetel. Empat
konfigurasi tiap model diuji pada lima fold berdasarkan tanggal dengan baris pelatihan
yang sama. Konfigurasi Random Forest terbaik memakai 100 pohon, kedalaman 20,
`min_samples_leaf=2`, dan `max_features=1.0`. CV MAE berubah dari 294,844 menjadi
290,620 atau membaik 1,433%. Pencarian Gradient Boosting justru mengubah CV MAE dari
294,003 menjadi 354,049, sehingga konfigurasi default tetap menjadi kandidat final.
Aturan MAE CV terendah memilih Random Forest hasil penyetelan. Selisih yang kecil tidak
diklaim signifikan secara statistik terhadap variasi antar-fold.

Data pengujian mencakup 11–17 Juni 2020 dengan 16.588 baris yang memenuhi syarat pada tujuh hari.
Plant 1 memiliki MAE 291,823 dan R² 0,964396, sedangkan Plant 2 memiliki MAE 104,683
dan R² 0,390257. MAE terhadap rata-rata development pembangkit sekitar 5,04% untuk
Plant 1 dan 23,08% untuk Plant 2. MAE absolut Plant 2 yang lebih kecil tidak berarti
model lebih andal karena skala targetnya berbeda. R² gabungan tidak cukup sebagai
dasar deployment.

Permutation importance terbesar muncul pada identitas pembangkit karena perbedaan
skala. Permutasi pembangkit secara terpisah dari inverter juga membentuk kombinasi
identitas yang tidak konsisten. Irradiation dan kuadratnya berbagi informasi. Semua
angka tersebut merupakan diagnostik, bukan estimasi efek sebab-akibat.

Korelasi residual lag satu pada observasi inverter yang berurutan sekitar 0,596 dan
0,771. Prototype menghasilkan 3 episode tinjauan di Plant 1 dan 10 di Plant 2 selama
tujuh hari pengujian. Jumlah tersebut adalah beban tinjauan, bukan jumlah kerusakan atau
alarm palsu. Pilot diprioritaskan pada Plant 1 dengan pemeriksaan operator, sementara
ketidaksesuaian output nol di Plant 2 perlu diselidiki tanpa mengubah model berdasarkan
hasil pengujian.

## Integritas eksekusi dan artefak

Percobaan penuh pertama mencapai evaluasi final, lalu gagal pada string pelaporan
rekomendasi. Perbaikan hanya menyentuh pelaporan dan tidak mengubah fitur, split, fold,
kandidat model, ruang dan anggaran penyetelan, aturan seleksi, atau kebijakan alert.
Satu eksekusi kernel bersih berikutnya dihentikan lingkungan eksekusi. Verifikasi akhir
kemudian menjalankan ulang protokol analitis yang sama dari kernel bersih hingga selesai.

Skrip pembuat notebook memeriksa sintaks setiap sel kode sebelum membuat notebook.
Sinkronisasi Markdown menolak perubahan pada sel kode dan mempertahankan keluaran.
Pemeriksa artefak menghitung ulang rata-rata training tiap fold dari empat CSV asli,
menghitung ulang metrik dari prediksi, memeriksa kunci seleksi, dan memastikan prediksi
model final sama dengan snapshot sebelum penguatan dalam toleransi numerik.

Verifikasi terakhir pada 14 Oktober 2026 berhasil dengan waktu helper 571,0 detik dan
waktu analisis notebook 562,3 detik. Python acuan adalah 3.12.10; versi paket tepat
tersedia di `requirements.txt`. Lingkungan yang digunakan adalah Python/Jupyter lokal,
bukan Google Colab. Ringkasan eksekusi tersedia di `outputs/run_summary.json`, sedangkan
hasil pemeriksaan terstruktur tersedia di `outputs/artifact_review.json`.

Hash SHA-256 notebook yang diverifikasi adalah
`9b9573553a5f59ee6e4560fffa55e028519dd589ffaef63a7b2aee2abaa8e613`.
Paket notebook dibuat dari snapshot yang sama, lulus pemeriksaan CRC, dan diperiksa
kesamaan byte setiap file terhadap workspace. `Group7_SolarGuard_Notebook.zip` adalah
paket serah-terima notebook, berbeda dari paket akhir mata kuliah
`Group7_SolarGuard_PredictiveAnalytics.zip`.

## Konsistensi bahasa

Seluruh 17 sel Markdown dan 11 penjelasan Markdown yang dihasilkan notebook menggunakan
bahasa Indonesia. Judul bagian, alasan metodologis, interpretasi hasil, rekomendasi,
dan keterbatasan telah diseragamkan. Nama dataset, model, metrik, parameter, kolom,
fungsi, serta istilah teknis yang diperlukan tetap dipertahankan agar cocok dengan
kode dan artefak.

Perubahan bahasa pada notebook hanya menyentuh Markdown dan argumen helper narasi
`note()`. Pemeriksaan AST memastikan komputasi analitis di luar narasi tidak berubah,
dan seluruh angka pada keluaran tetap sama. Bukti pemeriksaan tersimpan dalam
`outputs/language_revision.json`.

## Bukti yang masih harus dilengkapi kelompok

- Arsip bukti registrasi yang mencantumkan kelompok dan anggota.
- Verifikasi sumber asli, ketentuan penggunaan dan distribusi, satuan fisik, serta
  rating peralatan. Jika dosen menolak kesesuaian dataset, penggantian dataset akan
  memerlukan analisis baru.
- Catatan kontribusi nyata setiap anggota. `CONTRIBUTIONS.md` sengaja tidak mengarang
  pembagian kerja.
- Slide final, latihan presentasi, persiapan panel pemangku kepentingan, kesiapan
  wawancara, ZIP lengkap, dan unggahan RISE sebelum 19 Oktober 2026 pukul 20.00 WIB.
