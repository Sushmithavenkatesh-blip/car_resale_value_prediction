from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "car_resale_value.csv")

X = df.drop(columns=["resale_value"])
y = df["resale_value"]

categorical = ["fuel_type", "brand"]
numeric = [
    "car_age_years", "kilometers_driven",
    "engine_liters", "number_of_owners"
]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.2f}")
print(f"R2 Score: {r2_score(y_test, pred):.4f}")

joblib.dump(model, BASE / "car_resale_value_model.pkl")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred, alpha=0.6)
plt.xlabel("Actual Resale Value")
plt.ylabel("Predicted Resale Value")
plt.title("Actual vs Predicted Car Resale Value")
plt.tight_layout()
plt.savefig(BASE / "actual_vs_predicted.png", dpi=150)
print("Model and chart saved successfully.")
