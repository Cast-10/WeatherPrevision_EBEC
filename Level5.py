import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import utils
import joblib

def prepare_level5_data(df):
    df = df.copy()
    df = utils.setUp(df)
    
    # 4 Labels Estratégicas: Alta correlação e MAE controlado
    targets = ['temperature_2m', 'wind_speed_10m', 'relative_humidity_2m', 'rain']
    
    y_cols = []
    for target in targets:
        col_name = f'target_{target}_next_day'
        df[col_name] = df.groupby('location')[target].shift(-24)
        df[f'{target}_lag_24h'] = df.groupby('location')[target].shift(24)
        y_cols.append(col_name)
    
    df = df.dropna()
    X = df.drop(columns=y_cols)
    y = df[y_cols]
    
    return X, y, targets

def train_and_export_level5(df):
    X, y, target_names = prepare_level5_data(df)
    
    # Conta quantas colunas meteorológicas reais existem no seu CSV original
    # (Ajuste o filtro se o seu CSV tiver nomes de colunas diferentes)
    total_possible_labels = len([c for c in df.columns if c not in ['location', 'time', 'date', 'latitude', 'longitude']])

    model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    
    print(f"Treinando Level 5... Objetivos: {target_names}")
    model.fit(X, y)
    
    # Validação interna
    y_pred = model.predict(X)
    maes = [mean_absolute_error(y.iloc[:, i], y_pred[:, i]) for i in range(len(target_names))]
    avg_mae = np.mean(maes)
    
    # Cálculo Final do Score
    score = (2.5 / (1 + avg_mae)) * (len(target_names) / total_possible_labels) * 100
    
    print(f"\n===== RESULTADO LEVEL 5 =====")
    print(f"MAE Médio: {avg_mae:.4f}")
    print(f"Bônus de Labels: {len(target_names)}/{total_possible_labels}")
    print(f"Score Final: {score:.2f} pontos")
    print("==============================\n")

    # Exporta o modelo e a lista de labels para o Streamlit
    joblib.dump(model, 'finalModelLevel5.pkl')
    joblib.dump(target_names, 'labelsLevel5.pkl')

weather = pd.read_csv("metherology_dataset.csv")
train_and_export_level5(weather)