import streamlit as st


class MeteorologyForecastPanel:
    def __init__(self):
        pass

    def render(self, ml_result):
        if ml_result is None or not ml_result.is_future_day():
            return

        meteorology_forecast = ml_result.get_meteorology_forecast()

        if meteorology_forecast is None or not meteorology_forecast.has_predictions():
            return

        predictions = meteorology_forecast.get_all_predictions()

        st.markdown(
            "<h3 style='color:#0f172a; font-weight:700;'>Predicted Day Summary</h3>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h4 style='color:#0f172a; font-weight:600;'>Expected conditions for {ml_result.location} on {ml_result.selected_day}</h4>",
            unsafe_allow_html=True
        )

        cards = [
            ("Temperature", predictions.get("temperature_2m"), "°C"),
            ("Wind Speed", predictions.get("wind_speed_10m"), "km/h"),
            ("Humidity", predictions.get("relative_humidity_2m"), "%"),
            ("Rain", predictions.get("rain"), "mm"),
        ]

        cols = st.columns(4)

        for col, (label, value, unit) in zip(cols, cards):
            shown_value = 0.0 if value is None else value

            with col:
                st.markdown(
                    f"""
<div class="weather-card">
    <div class="weather-card-label">{label}</div>
    <div class="weather-card-value">{shown_value:.2f}</div>
    <div class="weather-card-unit">{unit}</div>
</div>
""",
                    unsafe_allow_html=True
                )