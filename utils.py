import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd

def remove_outliers(df, columns, k=7):
    mask = pd.Series(True, index=df.index)
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        mask &= (df[col] - mean).abs() <= k * std
    return df[mask]


def setUp(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna()

    # 1. DateTime Decomposition
    dt = pd.to_datetime(df['time'])
    
    # Hour of day (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    
    # Day of month (1-31)
    df['day_sin'] = np.sin(2 * np.pi * (dt.dt.day - 1) / 31)
    df['day_cos'] = np.cos(2 * np.pi * (dt.dt.day - 1) / 31)
    
    # Month of year (1-12)
    df['month_sin'] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)


    # 3. Location Mapping
    district_map = {
        "Aveiro": 1, "Beja": 2, "Braga": 3, "Bragança": 4, "Castelo Branco": 5,
        "Coimbra": 6, "Évora": 7, "Faro": 8, "Guarda": 9, "Leiria": 10,
        "Lisboa": 11, "Portalegre": 12, "Porto": 13, "Santarém": 14,
        "Setúbal": 15, "Viana do Castelo": 16, "Vila Real": 17, "Viseu": 18
    }

    df['location'] = df["location"].map(district_map)
    
    # 4. Clean up raw columns
    # Dropping 'time' as it's now encoded. 
    # Optional: drop 'wind_direction_10m' since we have Sin/Cos.
    df = df.drop(columns=['time'], errors='ignore')
    return df

def train_validate_test_split(X, y, val_size=0.15, test_size=0.15):
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, shuffle=False
    )
    relative_test_size = test_size / temp_size  # e.g. 0.15/0.30 = 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def setUpAccidents(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna()

    # 1. DateTime Decomposition
    dt = pd.to_datetime(df['time'])
    
    # Day of month (1-31)
    df['day_sin'] = np.sin(2 * np.pi * (dt.dt.day - 1) / 31)
    df['day_cos'] = np.cos(2 * np.pi * (dt.dt.day - 1) / 31)
    
    # Month of year (1-12)
    df['month_sin'] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)


    # 3. Location Mapping
    district_map = {
        "Aveiro": 1, "Beja": 2, "Braga": 3, "Bragança": 4, "Castelo Branco": 5,
        "Coimbra": 6, "Évora": 7, "Faro": 8, "Guarda": 9, "Leiria": 10,
        "Lisboa": 11, "Portalegre": 12, "Porto": 13, "Santarém": 14,
        "Setúbal": 15, "Viana do Castelo": 16, "Vila Real": 17, "Viseu": 18
    }

    df['location'] = df["location"].map(district_map)
    
    # 4. Clean up raw columns
    # Dropping 'time' as it's now encoded. 
    df = df.drop(columns=['time'], errors='ignore')
    return df

def print_f1_score(y_true, y_pred, dataset_name="Validation"):
    """
    Prints a full evaluation report including F1 score.
    
    Args:
        y_true       : real labels
        y_pred       : model predicted labels
        dataset_name : just a label for printing (e.g. "Validation", "Test")
    """
    f1 = f1_score(y_true, y_pred)
    
    print(f"===== {dataset_name} Evaluation =====")
    print(f"F1 Score: {f1:.4f}")
    print("=====================================\n")