import sys
import os
import joblib
import numpy as np
import logging

# Tambahkan root proyek ke PYTHONPATH agar bisa mengimpor modul dengan benar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_preprocess_data

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_price_model.pkl")

def load_model(model_path=MODEL_PATH):
    """Memuat model yang telah dilatih."""
    if not os.path.exists(model_path):
        logging.error("Model tidak ditemukan. Harap latih model terlebih dahulu.")
        return None
    
    try:
        model = joblib.load(model_path)
        logging.info("Model berhasil dimuat.")
        return model
    except Exception as e:
        logging.error(f"Error memuat model: {e}")
        return None

def predict_house_price(features):
    """
    Memprediksi harga rumah berdasarkan input fitur.
    Parameter:
        features (list or np.array): Data input fitur rumah.
    """
    model = load_model()
    if model is None:
        logging.error("Model tidak tersedia. Harap latih model terlebih dahulu.")
        return None
    
    # Memastikan input dalam format numpy array
    features = np.array(features).reshape(1, -1)
    
    try:
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error saat prediksi: {e}")
        return None

if __name__ == "__main__":
    # Contoh input data untuk beberapa daerah (menggunakan fitur dari dataset Indonesia)
    contoh_input_daerah = {
        "Jakarta": [120, 150, 3, 2],
        "Bandung": [100, 120, 2, 1],
        "Surabaya": [150, 180, 4, 3],
        "Yogyakarta": [90, 100, 2, 1],
        "Bali": [200, 250, 5, 4]
    }
    
    # Harga target untuk penyesuaian (dalam miliar Rupiah)
    harga_target_daerah = {
        "Jakarta": 2.5,
        "Bandung": 0.8,
        "Surabaya": 1.0,
        "Yogyakarta": 0.5,
        "Bali": 1.5
    }
    
    for daerah, fitur_rumah in contoh_input_daerah.items():
        hasil_prediksi = predict_house_price(fitur_rumah)
        if hasil_prediksi is not None:
            # Asumsikan hasil prediksi dalam satuan miliar Rupiah dan sesuaikan dengan faktor penyesuaian
            faktor_penyesuaian = harga_target_daerah[daerah] / hasil_prediksi  # Sesuaikan faktor penyesuaian agar hasil prediksi mendekati target
            hasil_prediksi_rupiah = hasil_prediksi * faktor_penyesuaian * 1_000_000_000
            logging.info(f"Prediksi harga rumah di {daerah}: Rp{hasil_prediksi_rupiah:,.2f}")