import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import utils
import joblib
import Level3 as l3

# --- PREPARATION ---
def prepare_level4_data(weather_raw, accidents_raw):
    # 1. Level 3 Integration
    weather = l3.addSnowIndicator(weather_raw) 
    
    # 2. Add temporal columns to Weather
    weather['day_of_week'] = pd.to_datetime(weather_raw['time']).dt.dayofweek
    weather['is_weekend'] = weather['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Setup Accidents
    accidents_raw['day_of_week'] = pd.to_datetime(accidents_raw['time']).dt.dayofweek
    accidents_raw['is_weekend'] = accidents_raw['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    accidents = utils.setUpAccidents(accidents_raw)
    
    # 4. Outlier Removal
    important_cols = ['temperature_2m', 'rain', 'wind_speed_10m', 'relative_humidity_2m']
    weather = utils.remove_outliers(weather, important_cols)

    # 5. Aggregate Weather to Daily
    weather_daily = weather.groupby(['location', 'day_sin', 'day_cos', 'month_sin', 'month_cos']).agg({
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'rain': 'sum',
        'wind_gusts_10m': 'max',
        'cloud_cover_low': 'mean',
        'detected_snow': 'max',
        'is_weekend': 'first',
        'day_of_week': 'first'
    }).reset_index()

    # 6. Final Merge - ADDING day_of_week and is_weekend to 'on'
    # This ensures these columns exist in the resulting 'df'
    merge_keys = ['location', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'day_of_week', 'is_weekend']
    
    df = pd.merge(
        accidents, 
        weather_daily, 
        on=merge_keys, 
        how='inner'
    )
    
    # 7. Feature Engineering
    # Now df['day_of_week'] will exist!
    df = df.sort_values(['location', 'day_sin', 'day_cos']) 

    # Calculate mean of all accidents seen SO FAR in this location
    df['location_avg'] = df.groupby('location')['accidents'].expanding().mean().reset_index(level=0, drop=True)

    # Shift it by 1 so the current day's accidents aren't included in its own average
    df['location_avg'] = df.groupby('location')['location_avg'].shift(1)

    # Fill the first day for each location (which will be NaN) with a global constant
    df['location_avg'] = df['location_avg'].fillna(df['accidents'].mean())
    
    df['is_peak_traffic'] = df['day_of_week'].isin([0, 1, 2, 3]).astype(int) 

    features = [
        'location', 'location_avg', 'day_of_week', 'is_weekend', 
        'detected_snow', 'is_peak_traffic', 'day_sin', 'day_cos', 
        'month_sin', 'month_cos', 'temperature_2m', 'rain', 
        'wind_gusts_10m', 'relative_humidity_2m'
    ]
    
    X = df[features]
    y = df['accidents'] * 3
    
    return X, y

# --- TRAINING ---
def trainLevel4(X, y):
    # Using absolute_error to align with the "Vehicle Error" metric
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10, 
        min_samples_leaf=15, 
        criterion='absolute_error', 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Training Level 4 (Brisa Logistics)...")
    model.fit(X, y)
    print("Finished training.")
    return model

# --- TESTING / EVALUATION ---
def testLevel4(weather_raw, accidents_raw):
    # 1. Prepare
    X, y = prepare_level4_data(weather_raw, accidents_raw)
    
    # 2. Split (Keeping time sequence if possible)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 3. Train
    model = trainLevel4(X_train, y_train)
    
    # 4. Predict & Round
    # Brisa needs whole vehicles, so we round the regression output
    raw_preds = model.predict(X_test)
    final_preds = np.round(raw_preds).astype(int)
    
    # 5. Metric Calculation: | 3 * actual - predicted_vehicles |
    vehicle_error = np.abs(y_test - final_preds).mean()
    
    print(f"\n===== Level 4 Highway Operational Evaluation =====")
    print(f"Mean Vehicle Error: {vehicle_error:.2f} vehicles")
    print(f"Total Daily Demand (Avg): {final_preds.mean():.1f} vehicles")
    print("==================================================\n")

# --- PRODUCTION ---
def Level4(weather_raw, accidents_raw):
    X, y = prepare_level4_data(weather_raw, accidents_raw)
    model = trainLevel4(X, y)
    return model

def exportLevel4(dfw, dfa):
    X, y = prepare_level4_data(dfw, dfa)
    model = trainLevel4(X, y)
    joblib.dump(model, 'finalModelLevel4.pkl')
# Execution
weather_init = pd.read_csv("metherology_dataset.csv")
accidents_init = pd.read_csv("accidents_dataset.csv")
testLevel4(weather_init, accidents_init)