import streamlit as st


class WeatherSummaryCards:
    def __init__(self, weather_service):
        # Save the service used to get weather data
        self.weather_service = weather_service

    def render(self, location: str, selected_day):
        # Get the weather data for the chosen district and day
        day_data = self.weather_service.get_data_by_location_and_day(location, selected_day)

        # If there is no data, show a warning and stop
        if day_data.empty:
            st.warning("No historical summary available for this date.")
            return

        # Compute summary values for the selected day
        avg_temperature = day_data["temperature_2m"].mean()
        total_rain = day_data["rain"].sum()
        avg_humidity = day_data["relative_humidity_2m"].mean()
        avg_wind = day_data["wind_speed_10m"].mean()
        avg_pressure = day_data["pressure_msl"].mean()

        # Section title
        st.markdown("<h3 style='color:#0f172a; font-weight:700;'>Daily Summary</h3>", unsafe_allow_html=True)
        
        # Create five cards in one row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #dbe6f3;
                    border-radius: 18px;
                    padding: 18px 14px;
                    text-align: center;
                    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
                    min-height: 120px;
                ">
                    <div style="font-size:14px; font-weight:600; color:black; margin-bottom:10px;">
                        Average Temperature
                    </div>
                    <div style="font-size:30px; font-weight:700; color:black; margin-bottom:8px;">
                        {avg_temperature:.1f}
                    </div>
                    <div style="font-size:13px; color:black;">
                        °C
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #dbe6f3;
                    border-radius: 18px;
                    padding: 18px 14px;
                    text-align: center;
                    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
                    min-height: 120px;
                ">
                    <div style="font-size:14px; font-weight:600; color:black; margin-bottom:10px;">
                        Total Rain
                    </div>
                    <div style="font-size:30px; font-weight:700; color:black; margin-bottom:8px;">
                        {total_rain:.1f}
                    </div>
                    <div style="font-size:13px; color:black;">
                        mm
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #dbe6f3;
                    border-radius: 18px;
                    padding: 18px 14px;
                    text-align: center;
                    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
                    min-height: 120px;
                ">
                    <div style="font-size:14px; font-weight:600; color:black; margin-bottom:10px;">
                        Average Humidity
                    </div>
                    <div style="font-size:30px; font-weight:700; color:black; margin-bottom:8px;">
                        {avg_humidity:.0f}
                    </div>
                    <div style="font-size:13px; color:black;">
                        %
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #dbe6f3;
                    border-radius: 18px;
                    padding: 18px 14px;
                    text-align: center;
                    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
                    min-height: 120px;
                ">
                    <div style="font-size:14px; font-weight:600; color:black; margin-bottom:10px;">
                        Average Wind
                    </div>
                    <div style="font-size:30px; font-weight:700; color:black; margin-bottom:8px;">
                        {avg_wind:.1f}
                    </div>
                    <div style="font-size:13px; color:black;">
                        km/h
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
                <div style="
                    background: white;
                    border: 1px solid #dbe6f3;
                    border-radius: 18px;
                    padding: 18px 14px;
                    text-align: center;
                    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
                    min-height: 120px;
                ">
                    <div style="font-size:14px; font-weight:600; color:black; margin-bottom:10px;">
                        Average Pressure
                    </div>
                    <div style="font-size:30px; font-weight:700; color:black; margin-bottom:8px;">
                        {avg_pressure:.1f}
                    </div>
                    <div style="font-size:13px; color:black;">
                        hPa
                    </div>
                </div>
            """, unsafe_allow_html=True)
