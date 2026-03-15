import streamlit as st


class HourlyWeatherTimeline:
    def __init__(self, weather_service):
        # Save the service used to get filtered weather data
        self.weather_service = weather_service

    def _get_weather_icon(self, rain_value: float, cloud_value: float, hour_value: int, is_snow: bool = False) -> str:
        # Snow must come from the ML model, not from manual rules
        if is_snow:
            return "❄️"

        # Rain icon
        if rain_value > 0.0:
            return "🌧️"

        # Night icons
        if hour_value >= 20 or hour_value < 6:
            if cloud_value >= 70:
                return "☁️"
            return "🌙"

        # Day icons
        if cloud_value >= 70:
            return "☁️"
        if cloud_value >= 30:
            return "⛅"
        return "☀️"

    def render(self, location: str, selected_day, snow_hours=None):
        # Get the weather data for the chosen district and day
        day_data = self.weather_service.get_data_by_location_and_day(location, selected_day)

        # If no ML result exists yet, assume no snow hours
        if snow_hours is None:
            snow_hours = set()

        st.markdown(
            f"<h4 style='color:#0f172a; font-weight:600;'>Hourly forecast for {location} on {selected_day}</h4>",
            unsafe_allow_html=True
        )

        if day_data.empty:
            st.warning("No historical data available for this date.")
            return

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

        for start in range(0, len(day_data), 6):
            row_data = day_data.iloc[start:start + 6]
            cols = st.columns(6)

            for col, (_, row) in zip(cols, row_data.iterrows()):
                hour = row["time"].strftime("%H:%M")
                hour_number = row["time"].hour
                temperature = f'{row["temperature_2m"]:.1f}°C'
                rain = f'{row["rain"]:.1f} mm'
                wind = f'{row["wind_speed_10m"]:.1f} km/h'

                # Snow is only shown if ML says so
                is_snow = hour_number in snow_hours

                icon = self._get_weather_icon(
                    row["rain"],
                    row["cloud_cover"],
                    hour_number,
                    is_snow=is_snow
                )

                precipitation_label = "Snow" if is_snow else "Rain"

                with col:
                    st.markdown(f"""
                        <div class="hour-card">
                            <div class="hour-time">{hour}</div>
                            <div class="hour-icon">{icon}</div>
                            <div class="hour-temp">{temperature}</div>
                            <div class="hour-detail">{precipitation_label}: {rain}</div>
                            <div class="hour-detail">Wind: {wind}</div>
                        </div>
                    """, unsafe_allow_html=True)
