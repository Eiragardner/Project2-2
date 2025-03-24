import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data import encode_data, load_data, prepare_train_test_data, preprocessed_data, save_prepared_data

if __name__ == "__main__":
    data = load_data("RentNL.csv")
    data = preprocessed_data(data)
    data_encoded = encode_data(data)

    save_prepared_data(data_encoded, 'prepared_data.csv')

    X_train, X_test, y_train, y_test = prepare_train_test_data(data_encoded)

    print(f"Training: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
