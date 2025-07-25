# GAMMAFEST-2025
GammaFest 2025 challenges participants to build a machine learning model that predicts citation links between scientific papers. Using full texts, metadata, and topic labels, the goal is to help the Elbaf Library recommend relevant research. Models are evaluated using the Matthews Correlation Coefficient (MCC).

# Competition Overview
## Background
For centuries, the Elbaf Library has served as the guardian of the world's knowledge. From ancient wisdom to modern science, everything is meticulously recorded in thousands of papers stored within its walls. The library is watched over by Biblo, a faithful guardian who ensures that knowledge remains accessible to those in need. However, as time has passed, the library’s reference management system has begun to show various weaknesses, hindering access to the vast knowledge it holds.

Now, a group of Holy Knights has arrived, seeking to document their intellectual journeys in the form of scientific papers. They hope to write works that will become a legacy for future generations, but they face a major challenge—how can they find the right references to support their research? They turn to Biblo for help, but Biblo no longer has the strength to assist the Holy Knights in finding, recommending, or connecting relevant papers.

## Problem
The current paper recommendation systems are not effective enough in providing relevant literature suggestions for the Holy Knights. The Elbaf Library needs a system that can help manage and present appropriate references for various research needs. Moreover, mapping relationships between papers remains a complex task, resulting in important references often being overlooked.

Participants in this competition are asked to build a machine learning model to solve this challenge. The model should be able to predict citation relationships between papers. The core issue is to help the Elbaf Library build an optimal reference recommendation system.

To evaluate the models, the Matthews Correlation Coefficient (MCC) will be used as the evaluation metric.

## Dataset Description
The dataset consists of pairs of scientific documents, each equipped with metadata such as titles, abstracts, and publication years. Each row in the dataset represents the possibility that the first document (paper) cites the second document (referenced_paper).

Participants are tasked with building a machine learning model to predict the value of is_referenced, indicating whether a citation relationship actually exists based on the available information.

Files:
- Folder Paper Database: Contains the full texts of documents in .txt format, with filenames corresponding to paper_id.
- papers_metadata.csv: Contains complete metadata for each document.
- train.csv: Contains document pairs with labels indicating whether the paper cites the referenced_paper.
- test.csv: Contains document pairs without labels, to be predicted by participants.
- sample_submission.csv: Sample format for submitting predictions on the test data.

## Metode

### Exploratory Data Analysis (EDA)

Tahap EDA dilakukan untuk memahami struktur data, mengidentifikasi pola, dan menemukan insight awal dari dataset. Proses EDA mencakup beberapa aktivitas analisis sebagai berikut:

**1. Analisis Informasi Umum Data**
Analisis dimulai dengan menggunakan fungsi `df.info()` untuk mendapatkan ringkasan DataFrame, termasuk jumlah non-null pada setiap kolom dan tipe datanya. Proses ini membantu mengidentifikasi kolom dengan missing values dan tipe data yang perlu dikonversi.

**2. Eksplorasi Data Awal**
Dilakukan pemeriksaan terhadap 5 baris pertama dataset menggunakan `df.head()` untuk memberikan gambaran awal tentang format data dan nilai-nilai yang ada dalam dataset.

**3. Analisis Perbedaan Tahun Publikasi**
Dibuat kolom baru `year_difference` dengan mengurangi tahun publikasi dokumen utama dengan tahun publikasi dokumen yang direferensikan. Nilai negatif pada `year_difference` diubah menjadi -1 untuk konsistensi data.

**4. Analisis Data Gabungan**
Dilakukan pemeriksaan informasi gabungan menggunakan `merged_df.info()` untuk melihat informasi umum setelah penggabungan data `train.csv` dengan `metadata.csv`.

**5. Analisis Data Training**
Pemeriksaan informasi data training asli dilakukan menggunakan `train_data.info()` sebelum digabungkan dengan metadata untuk memahami struktur data dasar.

### Data Cleaning

Tahap cleaning data berfokus pada penanganan missing values, duplikat, dan inkonsistensi data. Proses cleaning meliputi:

**1. Penghapusan Kolom Redundan**
Kolom `publication_date` dihapus dari DataFrame karena kolom `publication_year` sudah tersedia dan fitur `year_difference` telah dibuat sebagai pengganti yang lebih informatif.

**2. Pembersihan Data Penulis**
Fungsi `clean_authors` diterapkan untuk membersihkan kolom `authors` dengan proses sebagai berikut:
- Mengganti nilai NaN dengan string kosong
- Mengganti titik koma (`;`) dengan koma (`,`) untuk standardisasi
- Menghapus karakter non-alfabet/non-angka (kecuali koma dan spasi) dengan spasi
- Menghilangkan spasi ganda dan spasi di awal/akhir string

**3. Pembersihan Data Teks**
Kolom `title_paper` dan `title_referenced` dibersihkan menggunakan fungsi `clean_text` yang melakukan:
- Konversi teks menjadi huruf kecil
- Penghapusan karakter non-alfanumerik (kecuali koma dan spasi)
- Penghilangan spasi berlebih untuk konsistensi format

**4. Konversi Format Konsep**
Kolom `concepts_paper` dan `concepts_referenced` diubah dari string yang dipisahkan titik koma menjadi list string untuk mempersiapkan proses encoding selanjutnya.

## Data Preprocessing

Tahap preprocessing mempersiapkan data untuk pemodelan dengan melakukan feature engineering, encoding, dan scaling:

**1. Penggabungan Data**
Data `train.csv` digabungkan dengan `metadata.csv` menggunakan dua tahap:
- Penggabungan pertama untuk data `paper` menggunakan `paper_id` sebagai kunci
- Penggabungan kedua untuk data `referenced_paper` menggunakan `paper_id` sebagai kunci
Proses ini memperkaya data training dengan metadata dari kedua dokumen yang terlibat dalam referensi.

**2. Feature Engineering**
Beberapa fitur baru dibuat untuk meningkatkan kualitas prediksi:

*Kesamaan Judul (title_similarity)*
- Menggunakan `HashingVectorizer` untuk efisiensi memori dalam mengubah judul menjadi vektor numerik
- Menggunakan `TfidfVectorizer` pada subset data untuk kualitas yang lebih baik
- Menghitung kesamaan kosinus antara judul dokumen utama dan dokumen yang direferensikan

*Kesamaan Konsep (concept_sim)*
- Menghitung kesamaan Jaccard antara list konsep dokumen utama dan dokumen yang direferensikan
- Memberikan nilai numerik untuk mengukur kemiripan topik penelitian

*Tumpang Tindih Penulis (author_overlap)*
- Menghitung kesamaan Jaccard antara list penulis dokumen utama dan dokumen yang direferensikan
- Mencocokkan nama belakang untuk menangani variasi penulisan nama penulis

*Fitur Sitasi*
- `citation_ratio`: Rasio sitasi menggunakan formula `(cited_by_count_paper + 1) / (cited_by_count_referenced + 1)`
- `citation_diff`: Perbedaan logaritma sitasi menggunakan `np.log1p(cited_by_count_paper) - np.log1p(cited_by_count_referenced)`
- Penambahan +1 dan `np.log1p` digunakan untuk menangani nilai nol dan normalisasi

*Dampak Sitasi yang Disesuaikan Usia*
- `years_since_pub_ref`: Menghitung jumlah tahun sejak publikasi dokumen yang direferensikan
- `citation_rate_ref`: Tingkat sitasi per tahun untuk mengukur dampak penelitian

**3. Penskalaan Fitur Numerik**
Penerapan `StandardScaler` untuk menormalisasi fitur numerik agar memiliki skala yang seragam dan meningkatkan performa model.

**4 Penanganan Ketidakseimbangan Kelas**
Dilakukan penyeimbangan dataset melalui:
- Oversampling pada kelas minoritas (`is_referenced = 1`)
- Undersampling pada kelas mayoritas (`is_referenced = 0`)
- Penggabungan semua sampel positif dengan sampel negatif yang di-downsample

**5. Pembagian Dataset**
Dataset dibagi menjadi training dan testing set menggunakan `train_test_split` untuk evaluasi model yang objektif.

### Modeling

Tahap pemodelan menggunakan berbagai algoritma machine learning dan teknik ensemble:

**1. Model Individu**
Empat model dasar digunakan dalam penelitian ini:
- `HistGradientBoostingClassifier (HGB)`: Model boosting berbasis pohon yang efisien untuk dataset besar
- `RandomForestClassifier (RF)`: Model ensemble berbasis pohon dengan multiple decision trees
- `XGBoost (XGB)`: Implementasi gradient boosting yang optimized dan populer
- `Logistic Regression (LR)`: Model linear dasar untuk klasifikasi biner

**2. Model Ensemble**
Implementasi `VotingClassifier` dengan konfigurasi:
- Menggunakan `voting='soft'` untuk menggabungkan prediksi probabilitas dari semua model individu
- Optimasi bobot untuk setiap model menggunakan `GridSearchCV` berdasarkan metrik Matthews Correlation Coefficient (MCC)
- Training model individu dalam proses Grid Search untuk menemukan kombinasi bobot optimal
- Training ensemble akhir menggunakan bobot optimal pada seluruh data training

## Evaluation
Evaluasi model dilakukan menggunakan berbagai metrik klasifikasi untuk menilai performa prediksi:

**Metrik Evaluasi**
- `Classification Report`: Menyediakan Precision, Recall, dan F1-Score untuk setiap kelas, serta accuracy, macro avg, dan weighted avg
- `Matthews Correlation Coefficient (MCC)`: Metrik yang seimbang untuk dataset tidak seimbang dengan nilai antara -1 (prediksi salah) dan +1 (prediksi sempurna)

**Analisis Performa Model**
Hasil evaluasi menunjukkan:
- Random Forest: MCC = 0.5672
- HistGradientBoosting: MCC = 0.5822 (performa terbaik di antara model individu)
- Voting Classifier: MCC = 0.5793

**Analisis Importansi Fitur**
Menggunakan `feature_importances_` dari HistGradientBoostingClassifier untuk mengidentifikasi fitur paling berpengaruh:
1. `primaryGenreName_freq`
2. `downloads_encoded`
3. `appAge`
4. `userRatingCount`
5. `developerCountry_freq`

Fitur-fitur ini menunjukkan pengaruh signifikan terhadap hasil prediksi dan memberikan insight tentang faktor-faktor yang mempengaruhi keputusan referensi dalam penelitian.

```
Submissions are evaluated using .csv files with the specified column headers (see the sample submission). The MCC (Matthews Correlation Coefficient) metric will be used to assess the model’s performance. A portion of the test dataset results will form the public leaderboard. The full results will be revealed in the private leaderboard after the competition ends.
```
# Competition Link
https://www.kaggle.com/competitions/gammafest25
