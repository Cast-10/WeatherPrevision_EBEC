import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import utils


weather = pd.read_csv("metherology_dataset.csv")

def trainLevel3(df):
    df = df.copy()
    
    # --- 1. Feature Engineering for Snow ---
    # Snow is defined by the relationship between Temp, Dew Point, and Humidity.
    # We create "Dew Point Depression" - the closer this is to 0, the more likely precipitation.
    
    # We select the physical features that define "Snowy" conditions
    df['cloud_density'] = df['cloud_cover_low'] + df['cloud_cover_mid']
    
    # 2. Wind Turbulence (Gustiness often accompanies fronts)
    df['wind_turbulence'] = df['wind_gusts_10m'] - df['wind_speed_10m']
    
    # 3. Pressure Gap (Proxy for elevation/terrain)
    df['topo_gap'] = df['pressure_msl'] - df['surface_pressure']


    features_to_use = [
        'temperature_2m', 
        'dew_point_2m', 
        'relative_humidity_2m',
        'cloud_density',      # New!
        'wind_turbulence',    # New!
        'topo_gap',           # New!
        'surface_pressure'
    ]
    
    X = df[features_to_use]
    
    # --- 2. Scaling ---
    # Unsupervised models (Distance/Density based) require scaling!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 3. Unsupervised Model (Isolation Forest) ---
    # contamination=0.01 assumes snow is very rare (1% of the data)
    model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    
    print("Training...")
    # fit_predict returns -1 for outliers (potential snow) and 1 for normal
    df['is_anomaly'] = model.fit_predict(X_scaled)
    print("Finnished train.")
    
    # --- 4. Defining "Snow" from Anomalies ---
    # An anomaly is only "Snow" if it's actually cold.
    # We refine the model's discovery using meteorological logic.
    df['detected_snow'] = (
        (df['is_anomaly'] == -1) & 
        (df['temperature_2m'] < 2.5) &  # Standard snow threshold
        (df['cloud_density'] > 100)     # Ensures there's actually a thick cloud layer
    )

    
    print("Discovery Finished.")
    print(f"Detected Snow Events: {df['detected_snow'].sum()}")
    print(f"Detected Anomalies: {(df['is_anomaly'] == -1).sum()}")
    
    return df[df['detected_snow'] == True], model


weather_cleaned = utils.setUp(weather)

important_sensor_cols = [
    'temperature_2m', 
    'dew_point_2m', 
    'relative_humidity_2m',
    'surface_pressure',
    'cloud_cover_mid',
    'cloud_cover_low',
    'wind_gusts_10m',
    'wind_speed_10m'
]

weather_cleaned = utils.remove_outliers(weather_cleaned, important_sensor_cols)



# Run the Detection
snow_df, snow_model = trainLevel3(weather_cleaned)

# --- Level 3 Evaluation ---
print(f"\n===== Level 3 Snow Detection Evaluation =====")
if len(snow_df) > 0:
    print(f"Avg Temp during Snow: {snow_df['temperature_2m'].mean():.2f}°C")
    print(f"Avg Humidity during Snow: {snow_df['relative_humidity_2m'].mean():.2f}%")
    print("Top 5 Snow Events Found:")
    print(snow_df[['location', 'temperature_2m', 'dew_point_2m', 'relative_humidity_2m']].head())
else:
    print("No snow detected. Try increasing 'contamination' in the IsolationForest.")
print("=============================================\n")