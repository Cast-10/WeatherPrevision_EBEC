from typing import List, Optional

from visualizer.ml_hour_result import MLHourResult
from visualizer.accident_forecast_result import AccidentForecastResult


class MLDayResult:
    def __init__(
        self,
        location: str,
        selected_day,
        is_future: bool,
        hour_results: Optional[List[MLHourResult]] = None,
        accident_forecast: Optional[AccidentForecastResult] = None
    ):
        self.location = location
        self.selected_day = selected_day
        self._is_future = is_future
        self._hour_results = hour_results if hour_results is not None else []
        self._accident_forecast = (
            accident_forecast if accident_forecast is not None else AccidentForecastResult()
        )

    def is_future_day(self) -> bool:
        return self._is_future

    def is_historical_day(self) -> bool:
        return not self._is_future

    def get_hour_results(self) -> List[MLHourResult]:
        return self._hour_results

    def get_hour_result(self, hour: int):
        for result in self._hour_results:
            if result.get_hour() == hour:
                return result
        return None

    def get_accident_forecast(self) -> AccidentForecastResult:
        return self._accident_forecast

    def to_dict(self):
        return {
            "location": self.location,
            "selected_day": self.selected_day,
            "is_future": self._is_future,
            "hour_results": [hour_result.to_dict() for hour_result in self._hour_results],
            "accident_forecast": self._accident_forecast.to_dict()
        }