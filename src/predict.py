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
    # Contoh input data (menggunakan fitur dari dataset California Housing)
    contoh_input = [8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]
    
    hasil_prediksi = predict_house_price(contoh_input)
    if hasil_prediksi is not None:
        logging.info(f"Prediksi harga rumah: ${hasil_prediksi * 100000:.2f}")
