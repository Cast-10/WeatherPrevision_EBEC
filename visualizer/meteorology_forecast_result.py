class MeteorologyForecastResult:
    def __init__(self, predictions=None):
        self._predictions = predictions if predictions is not None else {}

    def get_prediction(self, label: str):
        return self._predictions.get(label)

    def get_all_predictions(self):
        return self._predictions

    def has_predictions(self) -> bool:
        return len(self._predictions) > 0

    def to_dict(self):
        return self._predictions.copy()