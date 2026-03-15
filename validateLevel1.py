import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import random
import utils
import joblib

weather = pd.read_csv("metherology_dataset.csv")

def trainLevel1(df, seed, estimators=100):
    X = df.drop(columns=['isRaining'])
    y = df['isRaining']
    
    X_train, X_val, X_test, y_train, y_val, y_test = utils.train_validate_test_split(X,y, randomState=seed)
    
    model = RandomForestClassifier(n_estimators=estimators, random_state=seed)
    
    print("Training...")
    model.fit(X_train, y_train)
    print("Finnished train.")
    
    return model, X_val, X_test, y_val, y_test

weather = utils.setUp(weather)
weather['isRaining'] = (weather['rain'] > 0).astype(int)

features = [
    'location',             
    'month',                
    'hour',                  
    'relative_humidity_2m',  
    'dew_point_2m',         
    'pressure_msl',          
    'cloud_cover',           
    'cloud_cover_low',       
    'wind_speed_10m',        
    'wind_direction_10m' 
]

weather = weather[features + ['isRaining']]
weather = utils.remove_outliers(weather, features)

model, X_val, X_test, y_val, y_test = trainLevel1(weather, seed=13, estimators=100)
y_pred_val = model.predict(X_val)
current_f1 = utils.print_f1_score(y_val, y_pred_val)
joblib.dump(model, 'finalModelLevel1.pkl')