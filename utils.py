import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd

weather = pd.read_csv("metherology_dataset.csv")
weather.columns = weather.columns.str.strip()

def setUp(df):
    # 1. Init
    df = df.copy()
    df = df.dropna() # Remove lines with NaN values
    
    # 2. Feature Engineering
    df['month'] = pd.to_datetime(df['time']).dt.month
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['isRaining'] = (df['rain'] > 0).astype(int)
    
    district_map = {
        "Aveiro": 1,
        "Beja": 2,
        "Braga": 3,
        "Bragança": 4,
        "Castelo Branco": 5,
        "Coimbra": 6,
        "Évora": 7,
        "Faro": 8,
        "Guarda": 9,
        "Leiria": 10,
        "Lisboa": 11,
        "Portalegre": 12,
        "Porto": 13,
        "Santarém": 14,
        "Setúbal": 15,
        "Viana do Castelo": 16,
        "Vila Real": 17,
        "Viseu": 18
    }
    df['location'] = df["location"].map(district_map)
    # 3. Return result
    return df

def train_validate_test_split(X, y, val_size=0.15, test_size=0.15, randomState=42):
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=randomState, stratify=y
    )
    relative_test_size = test_size / temp_size  # e.g. 0.15/0.30 = 0.50
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=randomState, stratify=y
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
