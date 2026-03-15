import streamlit as st
from datetime import timedelta


class WeatherSelector:
    def __init__(self, weather_service, future_days: int = 7):
        # Save the service used to get available locations and dates
        self.weather_service = weather_service
        self.future_days = future_days

    def render(self):
        # Get all available districts
        locations = self.weather_service.get_locations()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<div style='font-size:15px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Select District</div>",
                unsafe_allow_html=True
            )

            selected_location = st.selectbox(
                "District",
                options=locations,
                index=None,
                placeholder="Choose a district",
                label_visibility="collapsed"
            )

        with col2:
            st.markdown(
                "<div style='font-size:15px; font-weight:600; color:#0f172a; margin-bottom:8px;'>Select Date</div>",
                unsafe_allow_html=True
            )

            if selected_location is None:
                selected_day = None

                st.date_input(
                    "Date",
                    value=None,
                    disabled=True,
                    format="DD/MM/YYYY",
                    label_visibility="collapsed"
                )

            else:
                available_days = self.weather_service.get_available_days(selected_location)

                min_day = min(available_days)
                max_historical_day = max(available_days)
                max_day = max_historical_day + timedelta(days=self.future_days)

                selected_day = st.date_input(
                    "Date",
                    value=None,
                    min_value=min_day,
                    max_value=max_day,
                    format="DD/MM/YYYY",
                    label_visibility="collapsed"
                )

        return selected_location, selected_day
