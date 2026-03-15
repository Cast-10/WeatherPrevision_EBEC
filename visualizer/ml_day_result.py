from typing import Optional, List, Dict


class MLDayResult:
    def __init__(
        self,
        location: str,
        selected_day,
        is_future: bool,
        temperature: Optional[float] = None,
        rain: Optional[float] = None,
        humidity: Optional[float] = None,
        wind: Optional[float] = None,
        pressure: Optional[float] = None,
        snow_detected: Optional[bool] = None,
        snow_details: Optional[str] = None,
        snow_hours: Optional[List[int]] = None,
        extra_insights: Optional[Dict[str, object]] = None
    ):
        # Basic context
        self.location = location
        self.selected_day = selected_day
        self._is_future = is_future

        # Forecast-style values
        self._temperature = temperature
        self._rain = rain
        self._humidity = humidity
        self._wind = wind
        self._pressure = pressure

        # Unsupervised / ML insight values
        self._snow_detected = snow_detected
        self._snow_details = snow_details
        self._snow_hours = snow_hours if snow_hours is not None else []

        # Extra optional ML outputs for future extension
        self._extra_insights = extra_insights if extra_insights is not None else {}

    def is_future_day(self) -> bool:
        # Tells the UI whether this result belongs to a future day
        return self._is_future

    def is_historical_day(self) -> bool:
        # Tells the UI whether this result belongs to a historical day
        return not self._is_future

    def has_forecast_values(self) -> bool:
        # True if at least one forecast-style value exists
        return any(
            value is not None
            for value in [
                self._temperature,
                self._rain,
                self._humidity,
                self._wind,
                self._pressure
            ]
        )

    def has_unsupervised_insights(self) -> bool:
        # True if there is at least one unsupervised/ML insight
        return (
            self._snow_detected is not None
            or self._snow_details is not None
            or len(self._snow_hours) > 0
        )

    def get_temperature(self) -> Optional[float]:
        return self._temperature

    def get_rain(self) -> Optional[float]:
        return self._rain

    def get_humidity(self) -> Optional[float]:
        return self._humidity

    def get_wind(self) -> Optional[float]:
        return self._wind

    def get_pressure(self) -> Optional[float]:
        return self._pressure

    def get_snow_detected(self) -> Optional[bool]:
        return self._snow_detected

    def get_snow_details(self) -> Optional[str]:
        return self._snow_details

    def get_snow_hours(self) -> List[int]:
        return self._snow_hours

    def get_extra_insight(self, key: str):
        return self._extra_insights.get(key)

    def to_dict(self) -> Dict[str, object]:
        # Useful for debugging or passing around structured data
        return {
            "location": self.location,
            "selected_day": self.selected_day,
            "is_future": self._is_future,
            "temperature": self._temperature,
            "rain": self._rain,
            "humidity": self._humidity,
            "wind": self._wind,
            "pressure": self._pressure,
            "snow_detected": self._snow_detected,
            "snow_details": self._snow_details,
            "snow_hours": self._snow_hours,
            "extra_insights": self._extra_insights
        }
