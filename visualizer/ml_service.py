import pandas as pd

from visualizer.ml_day_result import MLDayResult


class MLService:
    def __init__(self, df: pd.DataFrame):
        # Save the dataset to use historical vs future logic
        self.df = df.copy()

    def get_last_available_day(self):
        # Return the last historical day that exists in the dataset
        return self.df["time"].dt.date.max()

    def is_future_day(self, selected_day) -> bool:
        # Check if the selected day is after the last historical day
        return selected_day > self.get_last_available_day()

    def build_result(self, location: str, selected_day) -> MLDayResult:
        # Decide if the selected day is future or historical
        is_future = self.is_future_day(selected_day)

        # For now, return an empty MLDayResult
        # Later, this method will call the real models and fill the values
        return MLDayResult(
            location=location,
            selected_day=selected_day,
            is_future=is_future,
            temperature=None,
            rain=None,
            humidity=None,
            wind=None,
            pressure=None,
            snow_detected=None,
            snow_details=None,
            snow_hours=[]
        )
