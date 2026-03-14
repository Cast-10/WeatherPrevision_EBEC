import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import setUp

def trainLevel1(df):
    # Preparation
    df = setUp.setUp(df)
    
    
    
    # Split (80% treino / 20% teste)
    # Usamos stratify=y para garantir que a proporção de "chuva" e "sol" 
    # seja a mesma no treino e no teste (vital para um bom F1-score)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # O RandomForest é ótimo, mas vamos ajustar o 'class_weight'
    # Se chover pouco no teu dataset, o modelo tende a ignorar a chuva.
    # 'balanced' força o modelo a dar mais atenção à classe minoritária.
    modelo = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        class_weight='balanced', 
        random_state=42
    )
    
    modelo.fit(X_train, y_train)
    
    # Predições
    y_pred = modelo.predict(X_test)
    
    # Avaliação focada em F1-Score
    f1 = f1_score(y_test, y_pred)
    
    print("--- Relatório de Performance ---")
    print(classification_report(y_test, y_pred))
    print(f"F1-Score Final: {f1:.4f}")
    
    # Mostrar quais colunas foram mais importantes
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    print("\nVariáveis mais influentes:")
    print(importancias.sort_values(ascending=False).head(5))
    
    return modelo

weather = setUp(weather)
final = trainLevel1(weather)