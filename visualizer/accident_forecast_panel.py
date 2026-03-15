import streamlit as st


class AccidentForecastPanel:
    def __init__(self):
        pass

    def render(self, ml_result):
        accident_forecast = ml_result.get_accident_forecast()

        if ml_result.is_future_day():
            title = "🚨 Road Safety Forecast"
            subtitle = "Predicted operational support for the selected future day."
            accidents_label = "Predicted Accidents"
            vehicles_label = "Predicted Vehicles Needed"
            accidents = accident_forecast.get_predicted_accidents()
            vehicles = accident_forecast.get_predicted_vehicles()
        else:
            title = "📍 Recorded Road Accidents"
            subtitle = "Observed accident data for the selected historical day."
            accidents_label = "Recorded Accidents"
            vehicles_label = "Recorded Vehicles Used"
            accidents = accident_forecast.get_actual_accidents()
            vehicles = accident_forecast.get_actual_vehicles()

        st.markdown(
            f"""
<div class="accident-card">
    <div class="accident-badge">{title}</div>
    <div class="accident-title">Road Accident Overview</div>
    <div class="accident-subtitle">{subtitle}</div>
</div>
""",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
<div class="weather-card">
    <div class="weather-card-label">{accidents_label}</div>
    <div class="weather-card-value">{accidents}</div>
    <div class="weather-card-unit">accidents/day</div>
</div>
""",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
<div class="weather-card">
    <div class="weather-card-label">{vehicles_label}</div>
    <div class="weather-card-value">{vehicles}</div>
    <div class="weather-card-unit">vehicles/day</div>
</div>
""",
                unsafe_allow_html=True
            )