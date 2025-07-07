# GAMMAFEST-2025
GammaFest 2025 challenges participants to build a machine learning model that predicts citation links between scientific papers. Using full texts, metadata, and topic labels, the goal is to help the Elbaf Library recommend relevant research. Models are evaluated using the Matthews Correlation Coefficient (MCC).

# Competition Overview
## Latar Belakang
Selama berabad-abad, Perpustakaan Elbaf telah menjadi penjaga pengetahuan dunia. Dari pengetahuan kuno hingga modern, semua tercatat dalam ribuan paper yang tersimpan dengan rapi. Perpustakaan ini dijaga oleh Biblo, satu-satunya penjaga yang setia memastikan bahwa pengetahuan tetap tersedia bagi mereka yang membutuhkannya. Namun, seiring berjalannya waktu, sistem manajemen referensi di perpustakaan mulai menunjukkan berbagai kelemahan, yang menghambat akses terhadap pengetahuan yang telah dikumpulkan selama ini.

Kini, sekelompok Ksatria Suci yang berambisi untuk mendokumentasikan perjalanan intelektual mereka dalam bentuk paper ilmiah datang ke Perpustakaan Elbaf. Mereka ingin menulis paper yang akan menjadi warisan bagi generasi mendatang, tetapi mereka menghadapi satu masalah besar—Bagaimana mereka bisa menemukan referensi yang tepat untuk mendukung penelitian mereka? Mereka meminta bantuan Biblo, tetapi Biblo tidak memiliki kekuatan untuk membantu para Ksatria Suci dalam menemukan, merekomendasikan, atau menghubungkan paper-paper yang relevan.

## Problem
Sistem rekomendasi paper yang ada saat ini belum cukup efektif dalam memberikan saran literatur yang relevan bagi para Ksatria Suci. Perpustakaan Elbaf membutuhkan sistem yang dapat membantu mengelola dan menyajikan referensi yang sesuai dengan kebutuhan penelitian. Selain itu, hubungan antar-paper masih sulit dipetakan dengan akurat, menyebabkan referensi yang relevan sering kali terlewatkan.

Para Ksatria Suci dalam kompetisi ini diminta untuk membangun model pembelajaran mesin yang dapat mengatasi tantangan tersebut. Model ini harus mampu memprediksi hubungan kutipan antar-paper. Persoalan utama yang dihadapi adalah bagaimana membantu Perpustakaan Elbaf dalam membangun sistem rekomendasi referensi paper yang optimal.

Untuk mengevaluasi model yang dikembangkan, akan digunakan metrik evaluasi MCC (Matthews Correlation Coefficient).

## Deskripsi Dataset
Dataset ini terdiri dari pasangan dokumen ilmiah, masing-masing dilengkapi dengan metadata seperti judul, abstrak, dan tahun publikasi. Setiap baris data merepresentasikan kemungkinan bahwa dokumen pertama (paper) mengutip dokumen kedua (referenced_paper).

Tugas peserta adalah membangun model pembelajaran mesin untuk memprediksi nilai is_referenced, yaitu apakah hubungan kutipan tersebut benar-benar terjadi, berdasarkan informasi yang tersedia.

Files:
- Folder Paper Database: Folder yang berisi isi dokumen dalam format .txt, dengan nama file sesuai paper_id.
- papers_metadata.csv: Berisi metadata lengkap untuk setia dokumen.
- train.csv: Berisi pasangan dokumen dengan label apakah paper mengutip referenced_paper.
- test.csv: Berisi pasangan dokumen tanpa label yang perlu diprediksi oleh peserta.
- sample_submission.csv: Contoh format pengumpulan prediksi untuk data uji.

## Evaluation
Evaluasi submission diberikan dengan format .csv yang memiliki header nama kolom (lihat contoh submisi). Hasil submisi akan dinilai dengan metrik MCC (Matthews Correlation Coefficient). Sebagian hasil penilaian test dataset akan digunakan untuk membentuk public leaderboard. Hasil utuh akan diberikan pada private leaderboard setelah kompetisi berakhir.

# Competition Link
https://www.kaggle.com/competitions/gammafest25
