import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import utils
import joblib

def prepare_level2_data(df):
    df = df.copy()
    df = utils.setUp(df)
    
    # --- Feature Engineering ---
    # Target: 1 hour in the future
    df['target_next_temp'] = df.groupby('location')['temperature_2m'].shift(-1)
    
    # Lags and Trends
    df['temp_lag_1h'] = df.groupby('location')['temperature_2m'].shift(1)
    df['temp_lag_24h'] = df.groupby('location')['temperature_2m'].shift(24)
    df['press_diff_1h'] = df['pressure_msl'] - df.groupby('location')['pressure_msl'].shift(1)
    df['hum_diff_1h'] = df['relative_humidity_2m'] - df.groupby('location')['relative_humidity_2m'].shift(1)

    # Drop NaNs from shifts
    df = df.dropna()

    # Outlier Removal
    important_sensor_cols = [
        'temperature_2m', 'pressure_msl', 'surface_pressure', 
        'wind_speed_10m', 'relative_humidity_2m'
    ]
    df = utils.remove_outliers(df, important_sensor_cols)

    # X/y Split
    X = df.drop(columns=[
        'target_next_temp', 'rain', 'dew_point_2m', 'relative_humidity_2m'
    ], errors='ignore')
    y = df['target_next_temp']
    
    return X, y

def trainLevel2(X, y):
    # Pure ML logic
    model = RandomForestRegressor(n_estimators=250, max_depth=15, n_jobs=-1, random_state=42)
    print("Training Level 2...")
    model.fit(X, y)
    print("Finished training.")
    return model

def testLevel2(df):
    # 1. Prepare
    X, y = prepare_level2_data(df)
    
    # 2. Split (Time-series safe)
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, shuffle=False, test_size=0.15)

    # 3. Train
    model = trainLevel2(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)

    print(f"\n===== Level 2 Forecasting Evaluation =====")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f} °C")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print("==========================================\n")


# function to train level2 with all of the csv, to then predict the values that are going to be tested
def Level2(df):
    X, y = prepare_level2_data(df)
    model = trainLevel2(X, y)
    return model

def exportLevel2(df):
    X, y = prepare_level2_data(df)
    model = trainLevel2(X, y)
    joblib.dump(model, 'finalModelLevel2.pkl')


weather = pd.read_csv("metherology_dataset.csv")
exportLevel2(weather)