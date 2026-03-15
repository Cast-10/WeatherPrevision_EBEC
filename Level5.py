import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import utils
import joblib


TARGETS = ['temperature_2m', 'wind_speed_10m', 'relative_humidity_2m', 'rain']

# Ajusta este valor conforme o número REAL de labels meteorológicas permitidas no challenge
TOTAL_LABELS_ALLOWED = 4


def prepare_level5_data(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["location", "time"]).reset_index(drop=True)

    df = utils.setUp(df)

    y_cols = []

    for target in TARGETS:
        target_col = f"target_{target}_next_day"
        y_cols.append(target_col)

        # Target: valor da mesma variável 24h no futuro
        df[target_col] = df.groupby("location")[target].shift(-24)

        # Lags
        df[f"{target}_lag_1h"] = df.groupby("location")[target].shift(1)
        df[f"{target}_lag_3h"] = df.groupby("location")[target].shift(3)
        df[f"{target}_lag_6h"] = df.groupby("location")[target].shift(6)
        df[f"{target}_lag_12h"] = df.groupby("location")[target].shift(12)
        df[f"{target}_lag_24h"] = df.groupby("location")[target].shift(24)

        # Rolling means usando apenas histórico anterior
        df[f"{target}_roll_mean_6h"] = (
            df.groupby("location")[target]
              .transform(lambda s: s.shift(1).rolling(6).mean())
        )

        df[f"{target}_roll_mean_24h"] = (
            df.groupby("location")[target]
              .transform(lambda s: s.shift(1).rolling(24).mean())
        )

    df = df.dropna().reset_index(drop=True)

    # Remove colunas que não devem entrar como features
    drop_cols = y_cols.copy()
    if "time" in df.columns:
        drop_cols.append("time")

    X = df.drop(columns=drop_cols)
    y = df[y_cols]

    feature_cols = X.columns.tolist()

    return X, y, TARGETS, feature_cols


def temporal_train_test_split(X, y, train_ratio=0.8):
    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def train_level5_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print(f"Training Level 5 for targets: {TARGETS}")
    model.fit(X_train, y_train)
    print("Finished training Level 5.")

    return model


def evaluate_level5(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)

    maes = {}
    mae_values = []

    for i, target in enumerate(target_names):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        maes[target] = mae
        mae_values.append(mae)

    avg_mae = float(np.mean(mae_values))

    score = (2.5 / (1 + avg_mae)) * (len(target_names) / TOTAL_LABELS_ALLOWED) * 100

    print("\n===== LEVEL 5 TEST RESULTS =====")
    for target, mae in maes.items():
        print(f"{target}: MAE = {mae:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Predicted labels: {len(target_names)}/{TOTAL_LABELS_ALLOWED}")
    print(f"Final Score: {score:.2f} points")
    print("================================\n")

    return maes, avg_mae, score


def train_and_export_level5(df):
    X, y, target_names, feature_cols = prepare_level5_data(df)

    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, train_ratio=0.8)

    model = train_level5_model(X_train, y_train)

    evaluate_level5(model, X_test, y_test, target_names)

    # Treina novamente em todos os dados antes de exportar
    final_model = train_level5_model(X, y)

    joblib.dump(final_model, "finalModelLevel5.pkl")
    joblib.dump(target_names, "labelsLevel5.pkl")
    joblib.dump(feature_cols, "featuresLevel5.pkl")

    print("Exported:")
    print("- finalModelLevel5.pkl")
    print("- labelsLevel5.pkl")
    print("- featuresLevel5.pkl")


# Execution
weather = pd.read_csv("metherology_dataset.csv")
train_and_export_level5(weather)