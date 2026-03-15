import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import utils

weather = pd.read_csv("metherology_dataset.csv")

def trainLevel2(df):
    df = df.copy()
    
    # --- Feature Engineering: Time-Series Context ---
    # We must groupby location so shifts don't mix different cities
    
    # TARGET: The temperature 1 hour in the future
    df['target_next_temp'] = df.groupby('location')['temperature_2m'].shift(-1)
    
    # LAGS: Give the model 'memory'
    df['temp_lag_1h'] = df.groupby('location')['temperature_2m'].shift(1)
    df['temp_lag_24h'] = df.groupby('location')['temperature_2m'].shift(24)
    
    # TRENDS: How are things changing?
    df['press_diff_1h'] = df['pressure_msl'] - df.groupby('location')['pressure_msl'].shift(1)
    df['hum_diff_1h'] = df['relative_humidity_2m'] - df.groupby('location')['relative_humidity_2m'].shift(1)

    # Drop the NaN rows created by shifting
    df = df.dropna()
    
    # --- Feature Selection ---
    # We keep current sensors + lags, but REMOVE the 'cheats' for the target hour
    X = df.drop(columns=[
        'target_next_temp', 
        'rain', 
        'dew_point_2m', 
        'relative_humidity_2m' # Removing current humidity ensures no math-leakage
    ], errors='ignore')

    y = df['target_next_temp']
    
    seed = 42
    # Regression split (no stratification)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_validate_test_split(X, y)
    
    model = RandomForestRegressor(n_estimators=250, max_depth=15, n_jobs=-1)
    
    print("Training...")
    model.fit(X_train, y_train)
    print("Finnished train.")
    
    return model, X_val, X_test, y_val, y_test



weather_cleaned = utils.setUp(weather)

# Identify the "Heavy Lifters" - these must be clean!
# We clean 'temperature_2m' specifically because it will become our target AND our lags.
important_sensor_cols = [
    'temperature_2m', 
    'pressure_msl', 
    'surface_pressure', 
    'wind_speed_10m',
    'relative_humidity_2m'
]

weather_cleaned = utils.remove_outliers(weather_cleaned, important_sensor_cols)

model, X_val, X_test, y_val, y_test = trainLevel2(weather_cleaned)

y_pred_val = model.predict(X_val)
print(f"\n===== Level 2 Forecasting Evaluation =====")
print(f"MAE: {mean_absolute_error(y_val, y_pred_val):.4f} °C")
print(f"R² Score: {r2_score(y_val, y_pred_val):.4f}")
print("=====================================\n")