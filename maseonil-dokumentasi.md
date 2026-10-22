# Submission 1: Klasifikasi Experience Level Member Gym

Nama: Rifky Rais Maselino

Username Dicoding: [maseonil](https://www.dicoding.com/users/maseonil/academies)

| Komponen                | Deskripsi                                                                                                                                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | [Gym Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)                                                                                                                                  |
| Masalah                 | Klasifikasi experience level member gym untuk membantu personalisasi program latihan berdasarkan karakteristik pengguna.                                                                                                   |
| Solusi Machine Learning | Pipeline MLOps menggunakan TensorFlow Extended (TFX), TensorFlow, dan TensorFlow Serving untuk membangun sistem klasifikasi experience level secara otomatis dan scalable.                                                 |
| Metode Pengolahan       | Data diproses menggunakan pipeline TFX meliputi ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Evaluator, dan Pusher.                                                                  |
| Arsitektur Model        | Model deep learning berbasis Artificial Neural Network (ANN) menggunakan Embedding Layer untuk fitur kategorikal, Dense Layer, Dropout, dan Softmax Output Layer.                                                          |
| Metrik Evaluasi         | Sparse Categorical Accuracy dan Loss.                                                                                                                                                                                      |
| Performa Model          | Model memperoleh akurasi validasi sekitar 92% dan menunjukkan performa klasifikasi yang stabil pada data evaluasi.                                                                                                         |
| Opsi Deployment         | Model serving dideploy menggunakan platform Railway dengan TensorFlow Serving sebagai backend deployment model.                                                                                                            |
| Web App                 | [gym-experience-model](https://gym-exp-level.up.railway.app/v1/models/gym-experience-model/metadata)                                                                                                                       |
| Monitoring              | Monitoring model serving dilakukan menggunakan Prometheus yang terintegrasi dengan Grafana untuk memantau jumlah request API, latensi API, waktu cold start, serta stabilitas artefak model selama deployment berlangsung. |

---

# Dataset

Dataset yang digunakan adalah Gym Dataset yang diperoleh dari Kaggle. Dataset ini berisi informasi karakteristik member gym seperti usia, BMI, detak jantung, durasi latihan, jenis workout, dan tingkat pengalaman latihan.

Jumlah data yang digunakan sebanyak 973 data member gym dengan total 15 fitur.

Fitur yang digunakan terdiri dari:

## Fitur Numerikal

* Age
* Weight (kg)
* Height (m)
* Max_BPM
* Avg_BPM
* Resting_BPM
* Session_Duration (hours)
* Calories_Burned
* Fat_Percentage
* Water_Intake (liters)
* Workout_Frequency (days/week)
* BMI

## Fitur Kategorikal

* Gender
* Workout_Type

## Label / Target

* Experience_Level

Dataset ini dipilih karena memiliki kombinasi fitur numerikal dan kategorikal yang cocok untuk implementasi pipeline machine learning berbasis TensorFlow Extended (TFX).

---

# Masalah Machine Learning

Menentukan experience level member gym secara manual membutuhkan waktu dan dapat menghasilkan klasifikasi yang tidak konsisten. Dengan machine learning, proses klasifikasi dapat dilakukan secara otomatis berdasarkan data karakteristik member.

Model ini diharapkan dapat membantu:

* personalisasi program latihan,
* rekomendasi intensitas workout,
* monitoring perkembangan member gym,
* serta optimalisasi strategi pelatihan.

---

# Solusi Machine Learning

Solusi machine learning dibangun menggunakan TensorFlow dan TensorFlow Extended (TFX) dengan pendekatan deep learning berbasis Artificial Neural Network (ANN).

Metode ANN dipilih karena:

* mampu menangani hubungan non-linear antar fitur,
* cocok untuk kombinasi fitur numerikal dan kategorikal,
* memiliki fleksibilitas tinggi untuk tuning hyperparameter,
* serta menghasilkan performa klasifikasi yang baik pada dataset tabular.

Selain itu, penggunaan TFX memungkinkan proses machine learning dilakukan secara end-to-end mulai dari data ingestion, preprocessing, training, evaluation, hingga model deployment.

Untuk meningkatkan performa model, digunakan komponen Tuner dengan Keras Tuner RandomSearch untuk mencari kombinasi hyperparameter terbaik seperti:

* jumlah neuron,
* dropout rate,
* dan learning rate.

Hasil tuning kemudian digunakan kembali pada proses training model final.

---

# Metode Pengolahan

Pipeline preprocessing dilakukan menggunakan TensorFlow Transform (TFT).

Tahapan preprocessing meliputi:

## 1. Data Ingestion

Dataset dimasukkan menggunakan ExampleGen dan dibagi menjadi:

* training split,
* evaluation split.

## 2. Data Validation

* StatisticsGen digunakan untuk menghasilkan statistik dataset.
* SchemaGen digunakan untuk membuat schema otomatis.
* ExampleValidator digunakan untuk mendeteksi anomali data.

## 3. Feature Transformation

Jumlah fitur yang digunakan:

* 12 fitur numerikal,
* 2 fitur kategorikal,
* 1 label target.

Transformasi yang digunakan:

### Fitur Numerikal

Dilakukan normalisasi menggunakan:

* z-score scaling.

Tujuan:

* mempercepat konvergensi model,
* menjaga stabilitas training,
* dan mengurangi perbedaan skala antar fitur.

### Fitur Kategorikal

Dilakukan encoding integer menggunakan vocabulary TensorFlow Transform.

Tujuan:

* mengubah data kategorikal menjadi representasi numerik,
* memungkinkan embedding layer mempelajari representasi fitur secara lebih efektif.

---

# Arsitektur Model

Model dibangun menggunakan TensorFlow Keras Functional API.

## Struktur Model

### Input Layer

* 12 input numerikal
* 2 input kategorikal

### Embedding Layer

Fitur kategorikal diproses menggunakan:

* Embedding dimension = 4

### Hidden Layer

* Dense Layer 1

  * jumlah neuron hasil tuning: 64
  * aktivasi: ReLU

* Dropout Layer

  * dropout rate: 0.4

* Dense Layer 2

  * jumlah neuron hasil tuning: 16
  * aktivasi: ReLU

### Output Layer

* Dense Layer

  * 4 neuron
  * aktivasi Softmax

---

## Hyperparameter

Hyperparameter tuning dilakukan menggunakan Keras Tuner RandomSearch.

Hasil hyperparameter terbaik:

* units_1 = 64
* units_2 = 16
* dropout = 0.4
* learning_rate = 0.001

Score tuning terbaik:

* validation accuracy = 0.9214

Konfigurasi training:

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: Sparse Categorical Crossentropy
* Batch Size: 32
* Epoch: 10
* Metric: Accuracy

Selain itu digunakan:

* EarlyStopping
* restore_best_weights=True

untuk menyimpan model dengan performa validasi terbaik dan mengurangi overfitting.

---

# Performa Model

Model berhasil memperoleh:

* akurasi validasi sekitar 92%,
* loss yang stabil selama training,
* dan performa klasifikasi yang konsisten pada data evaluasi.

Metrik evaluasi yang digunakan:

* Sparse Categorical Accuracy
* Loss

Berdasarkan hasil training:

* model tidak menunjukkan indikasi underfitting,
* overfitting relatif kecil karena gap training dan validation accuracy tidak terlalu jauh.

Penggunaan:

* dropout,
* hyperparameter tuning,
* dan early stopping

membantu meningkatkan generalisasi model.

---
# Opsi Deployment

Model serving dideploy menggunakan platform Railway dengan TensorFlow Serving sebagai backend deployment model. Railway dipilih karena menyediakan proses deployment container yang sederhana, cepat, dan mudah diintegrasikan dengan workflow MLOps berbasis Docker.

Pada deployment ini, model hasil training dari pipeline TFX disimpan dalam format TensorFlow SavedModel dan dijalankan menggunakan TensorFlow Serving sehingga endpoint inference dapat diakses secara realtime melalui REST API.

Alur deployment yang digunakan meliputi:

Model hasil Trainer dan Pusher pada pipeline TFX disimpan ke direktori serving model.
Artefak model kemudian dimasukkan ke dalam container Docker berbasis TensorFlow Serving.
Container deployment dijalankan pada Railway menggunakan konfigurasi port dan environment yang sesuai.
Endpoint inference tersedia secara online dan dapat digunakan untuk prediksi maupun metadata model.

Keunggulan deployment menggunakan Railway pada proyek ini:

Deployment lebih sederhana karena terintegrasi langsung dengan Docker container.
Mendukung continuous deployment melalui integrasi repository GitHub.
Mempermudah proses scaling dan restart deployment model serving.
Cocok untuk implementasi MLOps berbasis cloud dengan konfigurasi ringan.
Memungkinkan monitoring deployment secara realtime menggunakan Prometheus dan Grafana.

Endpoint model serving yang digunakan:

[gym-experience-model](https://gym-exp-level.up.railway.app/v1/models/gym-experience-model/metadata)

Endpoint tersebut digunakan untuk melihat metadata model,
memastikan model aktif,
serta memverifikasi versi model yang sedang berjalan pada TensorFlow Serving.
---

# Monitoring Model Serving

Sistem monitoring deployment dibangun menggunakan Prometheus yang terintegrasi dengan Grafana.

Monitoring dilakukan untuk memastikan model serving berjalan stabil dan mampu menangani request inference secara realtime.

Metrik yang dipantau meliputi:

## 1. Jumlah Request API

Digunakan untuk memantau total request inference yang masuk ke model serving.

## 2. Latensi API

Digunakan untuk mengukur waktu respons endpoint inference agar performa serving tetap optimal.

## 3. Waktu Cold Start

Digunakan untuk memantau waktu inisialisasi model serving saat pertama kali dijalankan atau setelah restart deployment.

## 4. Stabilitas Artefak Model

Monitoring dilakukan untuk memastikan artefak TensorFlow Serving tetap termuat dengan baik tanpa error selama deployment berlangsung.

Grafana digunakan untuk memvisualisasikan seluruh metrik monitoring dalam bentuk dashboard interaktif sehingga mempermudah observability dan troubleshooting sistem deployment.

---

# Kesimpulan

Pipeline machine learning berhasil dibangun menggunakan TensorFlow Extended (TFX) untuk melakukan klasifikasi experience level member gym secara otomatis.

Model deep learning yang dibangun mampu menghasilkan performa klasifikasi yang baik dengan akurasi validasi sekitar 92%.

Integrasi komponen:

* preprocessing,
* hyperparameter tuning,
* evaluasi,
* deployment,
* serta monitoring

membuat pipeline menjadi lebih modular, reproducible, dan siap digunakan pada skenario MLOps production pipeline.
