# Prediksi Harga Rumah dengan Machine Learning Sederhana

## Deskripsi
Proyek ini menggunakan dataset harga rumah di Indonesia untuk memprediksi harga rumah berdasarkan fitur-fitur yang tersedia. Model yang digunakan adalah Linear Regression.

## Instalasi
Pastikan Anda memiliki Python dan pip terinstal di sistem Anda. Kemudian, instal semua dependensi yang diperlukan dengan perintah berikut:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### 1. Data Preprocessing
Jalankan skrip `data_preprocessing.py` untuk memuat dan memproses dataset harga rumah.
```bash
python src/data_preprocessing.py
```

### 2. Melatih Model
Jalankan skrip `train_model.py` untuk melatih model Linear Regression dan menyimpannya.
```bash
python src/train_model.py
```

### 3. Melakukan Prediksi
Jalankan skrip `predict.py` untuk melakukan prediksi harga rumah berdasarkan input fitur.
```bash
python src/predict.py
```
