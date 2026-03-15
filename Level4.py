import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import utils
import Level3 as l3

def trainLevel4(df):
    # Separate Train and Validation before mean so we don't fall in Data Leakage
    train_size = int(len(df) * 0.7)
    df_train = df.iloc[:train_size].copy()
    
    # Accident mean by location based only on train
    loc_avg_map = df_train.groupby('location')['accidents'].mean().to_dict()
    df['location_avg'] = df['location'].map(loc_avg_map).fillna(df_train['accidents'].mean())
    
    df['is_peak_traffic'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
    
    features = [
        'location', 'location_avg',
        'day_of_week', 'is_weekend', 'is_snowing', 'is_peak_traffic',
        'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'temperature_2m', 'rain', 
        'wind_gusts_10m', 'relative_humidity_2m'
    ]
    
    X = df[features]
    y = df['accidents'] * 3
    
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_validate_test_split(X, y)
    
    model = RandomForestRegressor(n_estimators=300,max_depth=10, min_samples_leaf=15, criterion='absolute_error', random_state=42, n_jobs=-1)
    
    print("Training...")
    model.fit(X_train, y_train)
    print("Finnished train.")
    
    # Previsões
    predicted_vehicles = model.predict(X_val)
    
    vehicle_error = np.abs(y_val - predicted_vehicles).mean()   
     
    print(f"Mean Vehicle Error: {vehicle_error:.2f}")
    
    return model, vehicle_error

weather_init = pd.read_csv("metherology_dataset.csv")
accidents_init = pd.read_csv("accidents_dataset.csv")

accidents_init['day_of_week'] = pd.to_datetime(accidents_init['time']).dt.dayofweek
accidents_init['is_weekend'] = accidents_init['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
weather_init['is_snowing'] = l3.Level3(weather_init)['detected_snow']

weather = utils.setUp(weather_init)
accidents = utils.setUpAccidents(accidents_init)

# Limpeza de outliers nas colunas meteorológicas
important_cols = ['temperature_2m', 'rain', 'wind_speed_10m', 'relative_humidity_2m']
weather = utils.remove_outliers(weather, important_cols)

weather_daily = weather.groupby(['location', 'day_sin', 'day_cos', 'month_sin', 'month_cos']).agg({
    'temperature_2m': 'mean',
    'relative_humidity_2m': 'mean',
    'rain': 'sum',
    'wind_gusts_10m': 'max',
    'cloud_cover_low': 'mean',
    'is_snowing' : 'max',
    'hour_sin': 'first' # Mantemos as coordenadas temporais
}).reset_index()

# Merge Final
df_final = pd.merge(
    accidents, 
    weather_daily, 
    on=['location', 'day_sin', 'day_cos', 'month_sin', 'month_cos'], 
    how='inner'
)

model, avg_error = trainLevel4(df_final)