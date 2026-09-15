from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "car_resale_value_model.pkl")

sample = pd.DataFrame([{
    "car_age_years": 5,
    "kilometers_driven": 65000,
    "engine_liters": 1.5,
    "number_of_owners": 1,
    "fuel_type": "Petrol",
    "brand": "Toyota"
}])

prediction = max(0, float(model.predict(sample)[0]))
print(f"Predicted Car Resale Value: ₹{prediction:,.2f}")
