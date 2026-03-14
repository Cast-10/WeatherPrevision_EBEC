import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import utils

def trainLevel1(df):
    # Preparation
    df = utils.setUp(df)
    
    X = df.drop(columns=['rain', 'isRaining', 'time'])
    y = df['isRaining']
    
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_validate_test_split(X,y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training...")
    model.fit(X_train, y_train)
    print("Finnished train.")
    
    return model, X_val, X_test, y_val, y_test