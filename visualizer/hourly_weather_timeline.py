import streamlit as st
import pandas as pd


class HourlyWeatherTimeline:
    def __init__(self, weather_service):
        # Save the service used to get filtered weather data
        self.weather_service = weather_service

    def render(self, location: str, selected_day):
        # Get the weather data for the chosen district and day
        day_data = self.weather_service.get_data_by_location_and_day(location, selected_day)

        # Show a section title
        st.subheader(f"Hourly forecast for {location} on {selected_day}")

        # If there is no data, show a warning message and stop
        if day_data.empty:
            st.warning("No historical data available for this date.")
            return

        # Create a simpler hour column to display in the interface
        display_data = day_data.copy()
        display_data["hour"] = display_data["time"].dt.strftime("%H:%M")

        # Select only the most important columns for the user
        display_data = display_data[
            [
                "hour",
                "temperature_2m",
                "rain",
                "relative_humidity_2m",
                "wind_speed_10m",
                "pressure_msl"
            ]
        ]

        # Rename columns to more user-friendly names
        display_data = display_data.rename(columns={
            "hour": "Hour",
            "temperature_2m": "Temperature (°C)",
            "rain": "Rain (mm)",
            "relative_humidity_2m": "Humidity (%)",
            "wind_speed_10m": "Wind Speed (km/h)",
            "pressure_msl": "Pressure (hPa)"
        })

        # Show the hourly weather table
        st.dataframe(display_data, use_container_width=True)

# A classe HourlyWeatherTimeline serve para mostrar a evolução horária da meteorologia ao longo do dia escolhido.