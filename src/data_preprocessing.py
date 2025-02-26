import os
import pickle
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_preprocess_data(save_scaler=True, scaler_path="../models/scaler.pkl"):
    """ Memuat dan memproses dataset California Housing. """

    logging.info("Memuat dataset California Housing...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target  # Target variable

    logging.info("Membagi dataset menjadi training dan testing...")
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
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
