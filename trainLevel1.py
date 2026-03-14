import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import utils

weather = pd.read_csv("metherology_dataset.csv")

def trainLevel1(df):
    # Preparation
    df = utils.setUp(df)
    
    X = df.drop(columns=['rain', 'isRaining'])
    y = df['isRaining']
    
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_validate_test_split(X,y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    return model

weather = utils.setUp(weather)
model = trainLevel1(weather)