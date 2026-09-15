# Car Resale Value Prediction using Python and Scikit-learn

## Project Overview
This machine learning project predicts the resale value of a car using car age, kilometers driven, engine size, number of owners, fuel type, and brand.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- OneHotEncoder
- Joblib
- Matplotlib

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts resale value for a sample car
- `car_resale_value_model.pkl` - trained model
- `data/car_resale_value.csv` - synthetic dataset
- `actual_vs_predicted.png` - evaluation chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Output
The model predicts the estimated car resale value in Indian Rupees.

Note: The dataset is synthetic and intended for educational purposes only.
