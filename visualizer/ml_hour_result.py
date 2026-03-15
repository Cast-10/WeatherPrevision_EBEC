from typing import Optional, Dict


class MLHourResult:
    def __init__(
        self,
        hour: int,
        temperature: Optional[float] = None,
        rain: Optional[float] = None,
        humidity: Optional[float] = None,
        wind: Optional[float] = None,
        pressure: Optional[float] = None,
        snow_detected: Optional[bool] = None,
        extra_insights: Optional[Dict[str, object]] = None
    ):
        self._hour = hour
        self._temperature = temperature
        self._rain = rain
        self._humidity = humidity
        self._wind = wind
        self._pressure = pressure
        self._snow_detected = snow_detected
        self._extra_insights = extra_insights if extra_insights is not None else {}

    def get_hour(self) -> int:
        return self._hour

    def get_temperature(self):
        return self._temperature

    def get_rain(self):
        return self._rain

    def get_humidity(self):
        return self._humidity

    def get_wind(self):
        return self._wind

    def get_pressure(self):
        return self._pressure

    def get_snow_detected(self):
        return self._snow_detected

    def get_extra_insight(self, key: str):
        return self._extra_insights.get(key)

    def to_dict(self):
        return {
            "hour": self._hour,
            "temperature": self._temperature,
            "rain": self._rain,
            "humidity": self._humidity,
            "wind": self._wind,
            "pressure": self._pressure,
            "snow_detected": self._snow_detected,
            "extra_insights": self._extra_insights
        }