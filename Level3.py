from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import utils

def addSnowIndicator(df):
    df = df.copy()
    df = utils.setUp(df)
    
    # --- 1. Feature Engineering for Snow ---
    # Snow is defined by the relationship between Temp, Dew Point, and Humidity.
    # We create "Dew Point Depression" - the closer this is to 0, the more likely precipitation.
    
    # We select the physical features that define "Snowy" conditions
    df['cloud_density'] = df['cloud_cover_low'] + df['cloud_cover_mid']
    
    # 2. Wind Turbulence (Gustiness often accompanies fronts)
    df['wind_turbulence'] = df['wind_gusts_10m'] - df['wind_speed_10m']
    
    # 3. Pressure Gap (Proxy for elevation/terrain)
    df['topo_gap'] = df['pressure_msl'] - df['surface_pressure']

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

    df = utils.remove_outliers(df, important_sensor_cols)

    features_to_use = [
        'temperature_2m', 
        'dew_point_2m', 
        'relative_humidity_2m',
        'cloud_density',
        'wind_turbulence',
        'topo_gap',
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
    ).astype(int)
    
    # it returns a df with a column with value 0 if it does not snowed and 1 if it did
    return df