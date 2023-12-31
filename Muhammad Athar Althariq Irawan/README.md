# Laporan Proyek Machine Learning - Muhammad Athar Althariq Irawan

## Domain Proyek

Risiko kredit adalah salah satu aspek yang sangat penting dalam industri keuangan, khususnya dalam pengelolaan pemberian pinjaman. Risiko ini muncul ketika sebuah perusahaan atau lembaga keuangan meminjamkan uang kepada individu atau entitas lain, dan kemungkinan pembayaran pinjaman tersebut tidak sesuai dengan yang diharapkan. Oleh karena itu, pemahaman yang mendalam tentang risiko kredit sangat penting untuk mengambil keputusan yang bijak dalam pemberian pinjaman.

Dalam rangka mengelola risiko kredit dengan lebih baik, perusahaan seringkali mengumpulkan data tentang pinjaman yang telah diberikan dan yang telah ditolak. Data ini mencakup berbagai informasi, seperti profil peminjam, riwayat kredit, tingkat pendapatan, jumlah pinjaman, dan sebagainya. Menggunakan dataset ini, perusahaan dapat menganalisis risiko kredit dengan lebih cermat dan mengambil tindakan yang sesuai untuk menguranginya.

### Contoh Kasus:
Misalnya, bank ABC ingin mengoptimalkan persetujuan pinjaman. Dengan menganalisis data historis mereka, mereka dapat mengidentifikasi faktor-faktor yang paling mempengaruhi risiko kredit, seperti riwayat kredit peminjam, jumlah pinjaman, dan jenis pekerjaan. Dengan pemahaman ini, mereka dapat mengembangkan model prediksi risiko kredit yang membantu mereka membuat keputusan persetujuan pinjaman yang lebih baik.

Format Referensi: [Credit risk assessment mechanism of personal auto loan based on PSO-XGBoost Model](https://doi.org/10.1007/s40747-022-00854-y) 

## Business Understanding

### Problem Statements

Masalah latar belakang:
- Bagaimana dapat mengidentifikasi pola yang membedakan peminjam yang gagal membayar dengan yang berhasil membayar, berdasarkan data pinjaman yang telah diterima dan yang ditolak?
- Apa saja kriteria persetujuan pinjaman yang paling mempengaruhi tingkat risiko kredit, dan bagaimana dapat menyempurnakan kriteria ini untuk mengurangi risiko kredit seperti gagal bayar?
- Bagaimana dapat menggunakan dataset ini untuk membangun model prediksi risiko kredit yang dapat membantu perusahaan dalam pengambilan keputusan yang lebih baik dalam pemberian pinjaman?

### Goals

Tujuan dari pernyataan masalah:
- untuk mengidentifikasi faktor-faktor yang paling berpengaruh dalam menentukan apakah peminjam akan gagal atau berhasil membayar pinjamannya.
- untuk memahami kriteria persetujuan pinjaman yang paling memengaruhi risiko kredit
- Pertanyaan ini ingin mengetahui bagaimana dataset tersebut dapat digunakan untuk membangun model prediksi risiko kredit.


### Solution statements
- Melakukan analisa, eksplorasi, pre-processing pada data seperti menanganimissing value pada data.
- lalu membuat model dengan menggunakan algoritma XGBoost.

## Data Understanding
 Data yang didapat dan digunakan pada proyek ini didapat dari: [Loan Data 2007 - 2014](https://www.kaggle.com/datasets/rashahmosaad/loan-data-2007-2014).

- dataset ini berformat feather yang dikonversi menjadi csv.
- dataset memiliki 466285 baris
- dataset memiliki 6 kolom yang sangat penting
- Terdapat beberapa baris yang kosong (missing value) tapi karena tidak berpengaruh maka dibiarkan saja. 

### Variabel yang penting pada dataset:
dikarenakan ada banyak kolom maka dibatasi yang penting saja.

- loan_status : kolom kategorikal berisi status dari para peminjam, mulai dari baik sampai telat bayar.
- grade : merupakan resiko peminjaman yang dibuat oleh perusahaan, dari A-G dimana G merupakan resiko peminjaman paling besar.
- emp_title : merupakan kolom berisi pekerjaan dari para peminjam
- emp_length : berisi lama pinjaman yang diberikan perusahaan
- home_ownership : berisi status kepemilikan rumah para peminjam
- purpose : berisi alasan kenapa para peminjam meminjam dari perusahaan

dikarenakan akan membuat model dari peminjaman jadi variabel ini sangat penting. yang mana nantinya akan membuat kelas dengan mengubah yaitu peminjaman yang baik dan buruk, 0_good_loan dan 1_bad_loan

### Exploratory Data Analysis
- Grade
![Grade](https://github.com/athar3/ml-terapan/assets/72434013/767b3bd6-0fbc-466a-a51f-d4e2a0270dda)

merupakan resiko peminjaman yang dibuat oleh perusahaan, dari A-G dimana G merupakan resiko peminjaman paling besar.

- emp_title
![download](https://github.com/athar3/ml-terapan/assets/72434013/17868bb3-5983-4ed7-ae7b-a362d27587be)

merupakan kolom berisi pekerjaan dari para peminjam. disini dapat kita lihat bahwa walaupun peminjam terbanyak berasal dari kalangan guru, tetapi bad loan terbesar datang dari manajer, jadi manajer memiliki resiko menjadi peminjam terburuk terbesar disini

- emp_length
![download](https://github.com/athar3/ml-terapan/assets/72434013/a6e5243a-1365-4552-9134-883c6bc68487)

kebanyakan peminjam meminjam dengan rentang lebih dari 1 tahun.

- home_ownership 
![download](https://github.com/athar3/ml-terapan/assets/72434013/96f9a550-f2d6-40ec-b246-c930ba89f65d)

berisi status kepemilikan rumah para peminjam. kebanyakan rumah yang dimiliki dalam tahap mortgage atau digadaikan.

- Purpose
![download](https://github.com/athar3/ml-terapan/assets/72434013/06e87f0c-af48-4701-a2ae-7e294966989c)

disini terlihat jelas bahwa debt_consolidation yang berarti menggabungkan beberapa utang menjadi satu pinjaman yang lebih besar, bisa dibilang mereka menumpuk hutang mereka untuk membayar hutang yang lain

## Data Preparation
Teknik yang digunakan dalam Data Preparation yaitu:
- Menghapus fitur yang tidak diperlukan: akan dilakukan drop kolom dikarenakan kebanyakan kolom pada dataset tidak digunakan dalam proses menganalisa kredit, maka sebagian akan dihapus.
- _Splitting Dataset_ : Akan dilakukan pembagian dataset menjadi 2 bagian dengan proporsi yaitu 80% training dan 20% test. Train data digunakan sebagai training model dan test data digunakan sebagai validasi apakah model sudah akurat atau belum.
- _Missing Value_: Terdapat beberapa baris yang kosong (missing value) tapi karena sedikit maka tidak berpengaruh jadi dibiarkan saja.


## Modeling
Model yang digunakan proyek kali ini yaitu XGBoost (eXtreme Gradient Boosting). extreme gradien merupakan algoritma _Machine Learning_ yang mencoba memprediksi variabel target secara akurat dengan menggabungkan gabungan perkiraan dari serangkaian model yang lebih sederhana. Algoritma XGBoost berkinerja baik dalam _machine learning_ karena penanganannya yang kuat untuk berbagai jenis data, hubungan, distribusi, dan variasi hyperparameter yang dapat disesuaikan.

- fungsi define_models yang digunakan untuk mendefinisikan berbagai model klasifikasi yang akan dievaluasi.

- fungsi define_gbm_models yang digunakan untuk mendefinisikan model Gradient Boosting dengan berbagai konfigurasi. Ini mencakup XGBoost (jika use_xgb=True) dan Gradient Boosting Classifier (jika use_xgb=False).

- Fungsi make_pipeline digunakan untuk membuat sebuah pipeline yang terdiri dari tahap pemrosesan data (preprocessor) dan model yang akan dievaluasi.

- Sebuah fungsi evaluate_model digunakan untuk mengevaluasi kinerja model dengan menggunakan metrik tertentu (dalam hal ini, F1-score).

- Fungsi robust_evaluate_model digunakan untuk menangani kesalahan dan peringatan yang mungkin muncul selama evaluasi.

- Fungsi evaluate_models digunakan untuk mengevaluasi semua model yang telah didefinisikan sebelumnya.

- Fungsi summarize_results digunakan untuk mencetak ringkasan hasil evaluasi model.

## Evaluation
di proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan beberapa metriks seperti:
- akurasi: Akurasi merupakan kalkulasi presentase jumlah ketepatan prediksi dari jumlah seluruh data yang diprediksi.
Dinyatakan dalam persentase, akurasi = (Jumlah prediksi benar) / (Jumlah total data).
$\text{akurasi} = \frac{\text{Jumlah prediksi benar}}{\text{Jumlah total data}} $

pada model ini mendapatkan hasil:

|Train Score   |0.8864233625971649   |
|---|---|
|Test Score   |0.8926334625708321   |

Hasil penerapan akurasi pada proyek ini:

Train Score  = 0.8864233625971649

yang berarti pada data pelatihan model berhasil memprediksi dengan benar sekitar 88.64% dari semua sampel.

Test Score   = 0.8926334625708321 

yang berarti pada data pengujian model berhasil memprediksi dengan benar sekitar 89.26% dari semua sampel.


- f1 score: nilai _Harmonic Mean_ (Rata-rata Harmonik) dari Precision dan Recall.
Precision adalah sejauh mana prediksi positif model adalah benar. Precision dihitung sebagai:

$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $

Recall adalah sejauh mana model dapat mendeteksi semua instance yang benar. Recall dihitung sebagai:

$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $

F1-score memberikan keseluruhan pengukuran performa model yang mempertimbangkan trade-off antara Precision dan Recall.

|   |  f1-score |
|---|---|
|  0_good_loan |  0.99 |
|  1_bad_loan |  0.89 |

Hasil penerapan F1-score pada proyek ini:

F1-score untuk kelas pertama: 0.99 (dapat diinterpretasikan sebagai sangat baik)
F1-score untuk kelas kedua: 0.89 (dapat diinterpretasikan sebagai baik)

- analisis prediksi probabilitas
![download](https://github.com/athar3/ml-terapan/assets/72434013/29218e75-2396-4c2e-b292-15a54c7b81e1)

penggunaan model XGBoost untuk melakukan prediksi probabilitas dan kemudian melakukan analisis terhadap metrik presisi (precision) dan recall.

Garis recall yang dimulai dari 1.0 dan menurun: Ini berarti bahwa pada awalnya, model sangat baik dalam mengidentifikasi sebagian besar kasus positif yang sebenarnya (true positive) di antara semua kasus positif yang sebenarnya yang ada dalam data uji.

Garis precision yang dimulai dari 0.0 dan menaik: Ini berarti bahwa pada awalnya, model tidak terlalu memfilter contoh negatif (false positive) yang salah sebagai positif. Oleh karena itu, presisi awalnya rendah.

sedangkan pada titik (0.1, 0.87): Titik ini menunjukkan di mana kedua kurva tersebut bertemu. Ini adalah titik di mana nilai ambang tertentu telah mempengaruhi baik presisi (0.87) dan recall (0.1). Ini mengindikasikan bahwa pada nilai ambang tersebut, model memiliki keseimbangan antara menjaga presisi yang tinggi.

