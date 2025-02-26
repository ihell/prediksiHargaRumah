from src.train_model import train_and_save_model
from src.predict import predict_house_price
import numpy as np

def main():
    print("Training model...")
    train_and_save_model()
    
    # Test prediction with dummy data
    sample_input = np.random.rand(8)  # Dummy input
    print("Predicted house price:", predict_house_price(sample_input))

if __name__ == "__main__":
    main()
