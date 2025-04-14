import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class dataOptimization:
    def __init__(self, data):
        self.data = data
    def optimize(self, max_remove=100):
        results = []
        sorted_data = self.data.sort_values("Price", ascending=False).reset_index(drop=True)

        for i in range(max_remove + 1):
            current_data = sorted_data[i:]

            if len(current_data) < 100:
                break

            current_data = current_data.select_dtypes(include=["number"])
            if "Price" not in current_data.columns:
                print("price is not found")
                continue

            X = current_data.drop(columns=["Price"])
            y = current_data["Price"]
            if X.shape[1] == 0:
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            r2 = r2_score(y_test, y_pred)
            results.append((i, r2))

        ##best result
        if results:
            best = max(results, key=lambda x: x[1])
            print(f"Best R² = {best[1]:.4f} delete {best[0]} most expensive objects")

            xs, ys = zip(*results)
            plt.figure(figsize=(10, 5))
            plt.plot(xs, ys, marker='o')
            plt.title("R² vs. Number of top expensive houses removed")
            plt.xlabel("Number of removed houses")
            plt.ylabel("R²")
            plt.grid(True)
            plt.tight_layout()
           # plt.show()
        else:
            print("not enoght data")


    def remove (self, count=30):
        self.data=self.data.sort_values("Price", ascending=False).iloc[count:].reset_index(drop=True)

    def train_and_evaluate(self):
        df = self.data.select_dtypes(include=["number"])
        X = df.drop(columns=["Price"])
        y = df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
        print(f"R² on current data: {r2:.4f}")

    def save_without_top_outliers(self, count=30, filename="data_without_top30.csv"):
        data_cleaned = self.data.sort_values("Price", ascending=False).iloc[count:]
        data_cleaned.to_csv(filename, index=False)
