import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import utils
import joblib

def prepare_level1_data(df):
    df = df.copy()
    df = utils.setUp(df)
    
    # Target creation
    df['isRaining'] = (df['rain'] > 0).astype(int)

    features = [
        'location', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'hour_sin', 'hour_cos', 'relative_humidity_2m', 'dew_point_2m',
        'pressure_msl', 'cloud_cover', 'cloud_cover_low',
        'wind_speed_10m', 'wind_direction_10m'
    ]

    cols_to_clean = ['relative_humidity_2m', 'dew_point_2m', 'pressure_msl', 'wind_speed_10m']
    df = utils.remove_outliers(df, cols_to_clean)

    X = df[features]
    y = df['isRaining']
    return X, y

def trainLevel1(X, y):
    # This function now ONLY handles the ML training
    model = RandomForestClassifier(n_estimators=200, random_state=13, n_jobs=-1)
    print("Training Level 1...")
    model.fit(X, y)
    print("Finished training.")
    return model

def testLevel1(df):
    # 1. Prepare data
    X, y = prepare_level1_data(df)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.15)

    # 3. Train using the specific train function
    model = trainLevel1(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n===== Level 1 Evaluation =====")
    print(f"F1 Score: {f1:.4f}")
    print("==============================\n")

# function to train level2 with all of the csv, to then predict the values that are going to be tested
def Level1(df):
    X, y = prepare_level1_data(df)
    model = trainLevel1(X, y)
    return model
def exportLevel1(df):
    X, y = prepare_level1_data(df)
    model = trainLevel1(X, y)
    joblib.dump(model, 'finalModelLevel1.pkl')

weather = pd.read_csv("metherology_dataset.csv")
testLevel1(weather)