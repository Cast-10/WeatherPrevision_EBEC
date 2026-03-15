import streamlit as st


class FuturePredictionPanel:
    def __init__(self):
        pass

    def _get_weather_icon(self, rain_value: float, hour_value: int, is_snow: bool = False) -> str:
        if is_snow:
            return "❄️"

        if rain_value >= 1.0:
            return "🌧️"

        if hour_value >= 20 or hour_value < 6:
            return "🌙"

        return "☀️"

    def render(self, ml_result):
        if ml_result is None or not ml_result.is_future_day():
            return

        hour_results = ml_result.get_hour_results()

        st.markdown(
            "<h3 style='color:#0f172a; font-weight:700;'>Future Forecast (ML Prediction)</h3>",
            unsafe_allow_html=True
        )

        if not hour_results:
            st.info("No hourly future forecast is available yet.")
            return

        st.markdown(
            f"<h4 style='color:#0f172a; font-weight:600;'>Hourly forecast for {ml_result.location} on {ml_result.selected_day}</h4>",
            unsafe_allow_html=True
        )

        st.markdown("""
            <style>
                .hour-card {
                    background: white;
                    border: 1px solid #dbe6f3;
                    border-radius: 16px;
                    padding: 14px 10px;
                    text-align: center;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                    min-height: 170px;
                    margin-bottom: 12px;
                }

                .hour-time {
                    font-size: 16px;
                    font-weight: 700;
                    color: #0f172a;
                    margin-bottom: 6px;
                }

                .hour-icon {
                    font-size: 28px;
                    margin-bottom: 6px;
                }

                .hour-temp {
                    font-size: 24px;
                    font-weight: 700;
                    color: #16325c;
                    margin-bottom: 10px;
                }

                .hour-detail {
                    font-size: 13px;
                    color: #1f2937;
                    margin-bottom: 4px;
                }
            </style>
        """, unsafe_allow_html=True)

        for start in range(0, len(hour_results), 6):
            row_data = hour_results[start:start + 6]
            cols = st.columns(6)

            for col, hour_result in zip(cols, row_data):
                hour = hour_result.get_hour()
                temperature = hour_result.get_temperature()
                rain = hour_result.get_rain()
                wind = hour_result.get_wind()
                snow_detected = hour_result.get_snow_detected()

                shown_temp = 0.0 if temperature is None else temperature
                shown_rain = 0.0 if rain is None else rain
                shown_wind = 0.0 if wind is None else wind

                icon = self._get_weather_icon(
                    shown_rain,
                    hour,
                    is_snow=(snow_detected is True)
                )

                with col:
                    st.markdown(f"""
                        <div class="hour-card">
                            <div class="hour-time">{hour:02d}:00</div>
                            <div class="hour-icon">{icon}</div>
                            <div class="hour-temp">{shown_temp:.1f}°C</div>
                            <div class="hour-detail">Rain: {shown_rain:.1f} mm</div>
                            <div class="hour-detail">Wind: {shown_wind:.1f} km/h</div>
                        </div>
                    """, unsafe_allow_html=True)
