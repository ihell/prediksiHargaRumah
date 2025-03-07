import sys
import os
import joblib
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Tambahkan root proyek ke PYTHONPATH agar bisa mengimpor `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_preprocess_data

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path untuk menyimpan model dan hasil evaluasi
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "house_price_model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.txt")

def train_and_save_model(model_path=MODEL_PATH, retrain=False):
    """ Melatih model Linear Regression dan menyimpannya. """

    # Jika model sudah ada dan tidak ingin melatih ulang
    if os.path.exists(model_path) and not retrain:
        logging.info(f"Model sudah ada di {model_path}. Gunakan 'retrain=True' untuk melatih ulang.")
        return
    
    logging.info("Memuat dan memproses data...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    if X_train is None:
        logging.error("Gagal memuat dan memproses data. Pastikan file CSV ada di path yang benar.")
        return

    logging.info("Melatih model Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Mean Squared Error: {mse:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")

    # Simpan model ke dalam folder models/
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model berhasil disimpan di {model_path}")

    # Simpan hasil evaluasi ke file
    with open(METRICS_PATH, "w") as f:
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
    
    logging.info(f"Hasil evaluasi disimpan di {METRICS_PATH}")

if __name__ == "__main__":
    # Tambahkan argumen retrain=True jika ingin melatih ulang model
    train_and_save_model(retrain=True)