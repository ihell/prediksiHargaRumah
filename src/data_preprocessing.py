import os
import pickle
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_preprocess_data(save_scaler=True, scaler_path="../models/scaler.pkl"):
    """ Memuat dan memproses dataset harga rumah dari file CSV. """
    
    # Gunakan path absolut
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "house_prices_indonesia.csv"))
    logging.info(f"Path ke file CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        logging.error(f"File CSV tidak ditemukan di path: {csv_path}")
        return None, None, None, None, None
    
    logging.info("Memuat dataset harga rumah dari Indonesia...")
    df = pd.read_csv(csv_path)
    
    # Asumsikan kolom terakhir adalah target variable
    target_column = df.columns[-1]
    logging.info(f"Target variable: {target_column}")

    # Hapus kolom 'lokasi' jika ada
    if 'lokasi' in df.columns:
        df = df.drop(columns=['lokasi'])
        logging.info("Kolom 'lokasi' dihapus dari dataset.")

    logging.info("Membagi dataset menjadi training dan testing...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Melakukan standarisasi fitur...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Simpan scaler agar bisa digunakan saat prediksi
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler berhasil disimpan di {scaler_path}")

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    logging.info("Data preprocessing selesai!")