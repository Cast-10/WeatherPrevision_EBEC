class AccidentForecastResult:
    def __init__(
        self,
        predicted_accidents=None,
        predicted_vehicles=None,
        actual_accidents=None,
        actual_vehicles=None
    ):
        self._predicted_accidents = predicted_accidents
        self._predicted_vehicles = predicted_vehicles
        self._actual_accidents = actual_accidents
        self._actual_vehicles = actual_vehicles

    def get_predicted_accidents(self) -> int:
        return 0 if self._predicted_accidents is None else self._predicted_accidents

    def get_predicted_vehicles(self) -> int:
        return 0 if self._predicted_vehicles is None else self._predicted_vehicles

    def get_actual_accidents(self) -> int:
        return 0 if self._actual_accidents is None else self._actual_accidents

    def get_actual_vehicles(self) -> int:
        return 0 if self._actual_vehicles is None else self._actual_vehicles

    def has_prediction(self) -> bool:
        return self._predicted_accidents is not None

    def has_actuals(self) -> bool:
        return self._actual_accidents is not None

    def to_dict(self):
        return {
            "predicted_accidents": self.get_predicted_accidents(),
            "predicted_vehicles": self.get_predicted_vehicles(),
            "actual_accidents": self.get_actual_accidents(),
            "actual_vehicles": self.get_actual_vehicles()
        }