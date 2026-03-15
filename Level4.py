import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import utils
import Level3 as l3

def trainLevel4(df):
    # E nas features do trainLevel4:
    features = [
        'location', 'location_avg', 
        'day_of_week', 'is_weekend', 'is_snowing',
        'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'temperature_2m', 'rain', 'wind_gusts_10m', 'relative_humidity_2m'
    ]
    
    X = df[features]
    y = df['accidents'] * 3
    
    # Split Temporal (Não usamos shuffle para manter a lógica de tempo)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_validate_test_split(X, y)
    
    # Modelo Regressor (Previsão de números, não classes)
    # Usamos criterion='absolute_error' porque a métrica da Brisa é baseada em erro absoluto
    model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', random_state=42, n_jobs=-1)
    
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
weather_init['is_snowing'] = l3.cleanAndSnow(weather_init)['detected_snow']

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

# 1. Média por Localização
df_final['location_avg'] = df_final.groupby('location')['accidents'].transform('mean')

model, avg_error = trainLevel4(df_final)