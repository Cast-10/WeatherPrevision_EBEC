import streamlit as st
from datetime import timedelta


class WeatherSelector:
    def __init__(self, weather_service, future_days: int = 7):
        # Save the service used to get available locations and dates
        self.weather_service = weather_service

        # Define how many days into the future the calendar should allow
        self.future_days = future_days

    def render(self):
        # Get all available districts/locations
        locations = self.weather_service.get_locations()

        # Create two columns: one for district and one for date
        col1, col2 = st.columns(2)

        with col1:
            # Let the user choose the district
            selected_location = st.selectbox("Select district", locations)

        with col2:
            # Get the available historical days for the selected district
            available_days = self.weather_service.get_available_days(selected_location)

            # Define the first and last date shown in the calendar
            min_day = min(available_days)
            max_day = max(available_days) + timedelta(days=self.future_days)

            # Use the latest available real day as the default selected value
            default_day = max(available_days)

            # Show a calendar-like date selector
            selected_day = st.date_input(
                "Select day",
                value=default_day,
                min_value=min_day,
                max_value=max_day
            )

        # Return both chosen values
        return selected_location, selected_day

# A classe WeatherSelector serve para deixar o utilizador escolher o distrito e a data, incluindo datas futuras para previsão.