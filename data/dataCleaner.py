import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class dataCleaner:
    def load_data(path):
        return pd.read_csv(path)

    def preprocessed_data(data):
        for col in data.columns:
            if data[col].isnull().any():
                mode = data[col].mode()[0]
                data[col].fillna(mode, inplace=True)

        data["Price"] = data["Price"].replace({r'\$':'', ',':'', '€':'', ' ':''}, regex=True)
        data["Price"] = pd.to_numeric(data["Price"],errors="coerce")
        data["Price"].fillna(data["Price"].median(),inplace=True)
        for col in ["Total Rooms", "Bedrooms"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                data[col].fillna(data[col].median(), inplace=True)

        return data
    def encode_data(data):
        data["City"] = LabelEncoder().fit_transform(data["City"])
        categorical_cols = ["Build type", "House type", "Roof", "Energy label", "Garden", "Position", "Floors"]
        data = pd.get_dummies(data.drop(columns=["Address"]),columns=categorical_cols, drop_first=True)
        return data

    def prepare_train_test_data(data, target_col="Price", test_size=0.2, random_state=42):
        y = data[target_col]
        X = data.drop(columns=[target_col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    #scaler can be used for LR model 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        

        return X_train_scaled, X_test_scaled, y_train, y_test
    def save_prepared_data(data, filepath="prepared_data.csv"):
        data.to_csv(filepath, index=False)

