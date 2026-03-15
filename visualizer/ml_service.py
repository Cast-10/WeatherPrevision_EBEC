import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import utils

from visualizer.ml_day_result import MLDayResult
from visualizer.ml_hour_result import MLHourResult
from visualizer.accident_forecast_result import AccidentForecastResult


class MLService:
    def __init__(
        self,
        df: pd.DataFrame,
        accidents_df: pd.DataFrame = None,
        temperature_model=None,
        level4_model=None
    ):
        self.df = df.copy()
        self.accidents_df = accidents_df.copy() if accidents_df is not None else None
        self.temperature_model = temperature_model
        self.level4_model = level4_model

        self.location_map = {
            "Aveiro": 1, "Beja": 2, "Braga": 3, "Bragança": 4, "Castelo Branco": 5,
            "Coimbra": 6, "Évora": 7, "Faro": 8, "Guarda": 9, "Leiria": 10,
            "Lisboa": 11, "Portalegre": 12, "Porto": 13, "Santarém": 14,
            "Setúbal": 15, "Viana do Castelo": 16, "Vila Real": 17, "Viseu": 18
        }

        self.snow_df = self._build_snow_dataframe()

    def get_last_available_day(self):
        return self.df["time"].dt.date.max()

    def is_future_day(self, selected_day) -> bool:
        return selected_day > self.get_last_available_day()

    def _encode_time_features(self, dt_value):
        hour = dt_value.hour
        day = dt_value.day
        month = dt_value.month

        return {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * (day - 1) / 31),
            "day_cos": np.cos(2 * np.pi * (day - 1) / 31),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        }

    def _encode_daily_time_features(self, date_value):
        ts = pd.to_datetime(date_value)

        return {
            "day_of_week": ts.dayofweek,
            "is_weekend": 1 if ts.dayofweek >= 5 else 0,
            "is_peak_traffic": 1 if ts.dayofweek in [0, 1, 2, 3] else 0,
            "day_sin": np.sin(2 * np.pi * (ts.day - 1) / 31),
            "day_cos": np.cos(2 * np.pi * (ts.day - 1) / 31),
            "month_sin": np.sin(2 * np.pi * (ts.month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (ts.month - 1) / 12),
        }

    def _get_location_avg_before_day(self, location: str, selected_day):
        if self.accidents_df is None or self.accidents_df.empty:
            return 0.0

        df = self.accidents_df.copy()
        df["time"] = pd.to_datetime(df["time"])
        selected_day = pd.to_datetime(selected_day).date()

        df = df[df["location"] == location].copy()
        df["date"] = df["time"].dt.date

        past_rows = df[df["date"] < selected_day]

        if past_rows.empty:
            if "accidents" in df.columns and not df["accidents"].empty:
                return float(df["accidents"].mean())
            return 0.0

        if "accidents" in past_rows.columns:
            return float(past_rows["accidents"].mean())

        return 0.0

    def _get_daily_weather_features(self, location: str, selected_day):
        snow_df = self.snow_df.copy()
        snow_df["time"] = pd.to_datetime(snow_df["time"])
        selected_day = pd.to_datetime(selected_day).date()

        day_rows = snow_df[
            (snow_df["location_name"] == location) &
            (snow_df["time"].dt.date == selected_day)
        ].copy()

        if day_rows.empty:
            return None

        return {
            "temperature_2m": float(day_rows["temperature_2m"].mean()) if "temperature_2m" in day_rows.columns else 0.0,
            "relative_humidity_2m": float(day_rows["relative_humidity_2m"].mean()) if "relative_humidity_2m" in day_rows.columns else 0.0,
            "rain": float(day_rows["rain"].sum()) if "rain" in day_rows.columns else 0.0,
            "wind_gusts_10m": float(day_rows["wind_gusts_10m"].max()) if "wind_gusts_10m" in day_rows.columns else 0.0,
            "detected_snow": int(day_rows["detected_snow"].max()) if "detected_snow" in day_rows.columns else 0
        }

    def _get_future_daily_weather_features(self, location: str, selected_day):
        hour_results = self._build_future_hour_results(location, selected_day)

        predicted_temps = [
            h.get_temperature() for h in hour_results
            if h.get_temperature() is not None
        ]

        if not predicted_temps:
            return None

        snow_df = self.snow_df.copy()
        snow_df["time"] = pd.to_datetime(snow_df["time"])

        location_rows = snow_df[snow_df["location_name"] == location].copy()
        location_rows = location_rows.sort_values("time")

        if location_rows.empty:
            return None

        last_day = location_rows["time"].dt.date.max()
        fallback_rows = location_rows[location_rows["time"].dt.date == last_day].copy()

        return {
            "temperature_2m": float(np.mean(predicted_temps)),
            "relative_humidity_2m": float(fallback_rows["relative_humidity_2m"].mean()) if "relative_humidity_2m" in fallback_rows.columns else 0.0,
            "rain": float(fallback_rows["rain"].sum()) if "rain" in fallback_rows.columns else 0.0,
            "wind_gusts_10m": float(fallback_rows["wind_gusts_10m"].max()) if "wind_gusts_10m" in fallback_rows.columns else 0.0,
            "detected_snow": 0
        }

    def _build_level4_feature_row(self, location: str, selected_day):
        if self.level4_model is None:
            return None

        time_features = self._encode_daily_time_features(selected_day)
        location_code = self.location_map.get(location)

        if location_code is None:
            return None

        if self.is_future_day(selected_day):
            weather_features = self._get_future_daily_weather_features(location, selected_day)
        else:
            weather_features = self._get_daily_weather_features(location, selected_day)

        if weather_features is None:
            return None

        location_avg = self._get_location_avg_before_day(location, selected_day)

        row = {
            "location": location_code,
            "location_avg": location_avg,
            "day_of_week": time_features["day_of_week"],
            "is_weekend": time_features["is_weekend"],
            "detected_snow": weather_features["detected_snow"],
            "is_peak_traffic": time_features["is_peak_traffic"],
            "day_sin": time_features["day_sin"],
            "day_cos": time_features["day_cos"],
            "month_sin": time_features["month_sin"],
            "month_cos": time_features["month_cos"],
            "temperature_2m": weather_features["temperature_2m"],
            "rain": weather_features["rain"],
            "wind_gusts_10m": weather_features["wind_gusts_10m"],
            "relative_humidity_2m": weather_features["relative_humidity_2m"]
        }

        return pd.DataFrame([row])

    def _build_feature_row(self, history_df: pd.DataFrame):
        history_df = history_df.copy().sort_values("time").reset_index(drop=True)

        if len(history_df) < 25:
            return None

        current = history_df.iloc[-1]
        prev_1 = history_df.iloc[-2]
        prev_24 = history_df.iloc[-25]

        time_features = self._encode_time_features(pd.to_datetime(current["time"]))

        feature_row = {
            "location": self.location_map.get(current["location"], current["location"]),
            "temperature_2m": current["temperature_2m"],
            "cloud_cover": current["cloud_cover"],
            "cloud_cover_low": current["cloud_cover_low"],
            "cloud_cover_mid": current["cloud_cover_mid"],
            "cloud_cover_highh": current["cloud_cover_highh"],
            "wind_speed_10m": current["wind_speed_10m"],
            "wind_direction_10m": current["wind_direction_10m"],
            "wind_gusts_10m": current["wind_gusts_10m"],
            "wind_direction_100m": current["wind_direction_100m"],
            "wind_speed_100m": current["wind_speed_100m"],
            "pressure_msl": current["pressure_msl"],
            "surface_pressure": current["surface_pressure"],
            "hour_sin": time_features["hour_sin"],
            "hour_cos": time_features["hour_cos"],
            "day_sin": time_features["day_sin"],
            "day_cos": time_features["day_cos"],
            "month_sin": time_features["month_sin"],
            "month_cos": time_features["month_cos"],
            "temp_lag_1h": prev_1["temperature_2m"],
            "temp_lag_24h": prev_24["temperature_2m"],
            "press_diff_1h": current["pressure_msl"] - prev_1["pressure_msl"],
            "hum_diff_1h": current["relative_humidity_2m"] - prev_1["relative_humidity_2m"],
        }

        return pd.DataFrame([feature_row])

    def _add_snow_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["cloud_density"] = df["cloud_cover_low"] + df["cloud_cover_mid"]
        df["wind_turbulence"] = df["wind_gusts_10m"] - df["wind_speed_10m"]
        df["topo_gap"] = df["pressure_msl"] - df["surface_pressure"]

        features_to_use = [
            "temperature_2m",
            "dew_point_2m",
            "relative_humidity_2m",
            "cloud_density",
            "wind_turbulence",
            "topo_gap",
            "surface_pressure"
        ]

        X = df[features_to_use]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)

        df["is_anomaly"] = model.fit_predict(X_scaled)

        df["detected_snow"] = (
            (df["is_anomaly"] == -1) &
            (df["temperature_2m"] < 2.5) &
            (df["cloud_density"] > 100)
        ).astype(int)

        return df

    def _build_snow_dataframe(self) -> pd.DataFrame:
        raw_df = self.df.copy()
        raw_df["time"] = pd.to_datetime(raw_df["time"])

        raw_time = raw_df["time"].copy()
        raw_location = raw_df["location"].copy()

        processed_df = utils.setUp(raw_df.copy())

        important_sensor_cols = [
            "temperature_2m",
            "dew_point_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "cloud_cover_mid",
            "cloud_cover_low",
            "wind_gusts_10m",
            "wind_speed_10m"
        ]

        processed_df = utils.remove_outliers(processed_df, important_sensor_cols)

        processed_df["time"] = raw_time.loc[processed_df.index]
        processed_df["location_name"] = raw_location.loc[processed_df.index]

        snow_df = self._add_snow_indicator(processed_df)
        return snow_df

    def _build_future_hour_results(self, location: str, selected_day):
        if self.temperature_model is None:
            return [
                MLHourResult(
                    hour=hour,
                    temperature=None,
                    rain=None,
                    humidity=None,
                    wind=None,
                    pressure=None,
                    snow_detected=None
                )
                for hour in range(24)
            ]

        location_df = self.df[self.df["location"] == location].copy()

        if location_df.empty:
            return []

        location_df = location_df.sort_values("time").reset_index(drop=True)
        target_day = pd.to_datetime(selected_day).date()
        history = location_df.copy()

        while True:
            last_time = pd.to_datetime(history.iloc[-1]["time"])

            if last_time.date() > target_day:
                break
            if last_time.date() == target_day and last_time.hour == 23:
                break

            X_next = self._build_feature_row(history)
            if X_next is None:
                return []

            predicted_temp = float(self.temperature_model.predict(X_next)[0])

            last_row = history.iloc[-1].copy()
            new_time = pd.to_datetime(last_row["time"]) + pd.Timedelta(hours=1)

            new_row = last_row.copy()
            new_row["time"] = new_time
            new_row["temperature_2m"] = predicted_temp

            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        selected_rows = history[pd.to_datetime(history["time"]).dt.date == target_day].copy()
        selected_rows = selected_rows.sort_values("time")

        hour_results = []
        for _, row in selected_rows.iterrows():
            hour_results.append(
                MLHourResult(
                    hour=int(pd.to_datetime(row["time"]).hour),
                    temperature=float(row["temperature_2m"]) if pd.notna(row["temperature_2m"]) else None,
                    rain=None,
                    humidity=None,
                    wind=None,
                    pressure=None,
                    snow_detected=None
                )
            )

        return hour_results

    def _build_historical_hour_results(self, location: str, selected_day):
        snow_df = self.snow_df.copy()
        snow_df["time"] = pd.to_datetime(snow_df["time"])

        day_rows = snow_df[
            (snow_df["location_name"] == location) &
            (snow_df["time"].dt.date == selected_day)
        ].copy()

        day_rows = day_rows.sort_values("time")

        if day_rows.empty:
            return []

        hour_results = []
        for _, row in day_rows.iterrows():
            hour_results.append(
                MLHourResult(
                    hour=int(pd.to_datetime(row["time"]).hour),
                    temperature=None,
                    rain=None,
                    humidity=None,
                    wind=None,
                    pressure=None,
                    snow_detected=bool(row["detected_snow"]) if pd.notna(row["detected_snow"]) else None
                )
            )

        return hour_results

    def _build_historical_accident_result(self, location: str, selected_day):
        if self.accidents_df is None or self.accidents_df.empty:
            return AccidentForecastResult(
                actual_accidents=None,
                actual_vehicles=None
            )

        df = self.accidents_df.copy()
        df["time"] = pd.to_datetime(df["time"])

        day_rows = df[
            (df["location"] == location) &
            (df["time"].dt.date == selected_day)
        ].copy()

        if day_rows.empty:
            return AccidentForecastResult(
                actual_accidents=None,
                actual_vehicles=None
            )

        actual_accidents = int(day_rows["accidents"].sum()) if "accidents" in day_rows.columns else 0
        actual_vehicles = actual_accidents * 3

        return AccidentForecastResult(
            actual_accidents=actual_accidents,
            actual_vehicles=actual_vehicles
        )
        
    def _build_future_accident_result(self, location: str, selected_day):
        if self.level4_model is None:
            return AccidentForecastResult(
                predicted_accidents=None,
                predicted_vehicles=None
            )

        X = self._build_level4_feature_row(location, selected_day)

        if X is None or X.empty:
            return AccidentForecastResult(
                predicted_accidents=None,
                predicted_vehicles=None
            )

        predicted_vehicles = float(self.level4_model.predict(X)[0])
        predicted_vehicles = max(0, int(round(predicted_vehicles)))
        predicted_accidents = max(0, int(round(predicted_vehicles / 3)))

        return AccidentForecastResult(
            predicted_accidents=predicted_accidents,
            predicted_vehicles=predicted_vehicles
        )

    def _build_accident_forecast(self, location: str, selected_day):
        if self.is_future_day(selected_day):
            return self._build_future_accident_result(location, selected_day)

        return self._build_historical_accident_result(location, selected_day)

    def build_hour_results(self, location: str, selected_day):
        if self.is_future_day(selected_day):
            return self._build_future_hour_results(location, selected_day)

        return self._build_historical_hour_results(location, selected_day)

    def build_result(self, location: str, selected_day) -> MLDayResult:
        is_future = self.is_future_day(selected_day)
        hour_results = self.build_hour_results(location, selected_day)
        accident_forecast = self._build_accident_forecast(location, selected_day)

        return MLDayResult(
            location=location,
            selected_day=selected_day,
            is_future=is_future,
            hour_results=hour_results,
            accident_forecast=accident_forecast
        )