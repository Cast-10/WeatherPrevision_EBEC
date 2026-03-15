import streamlit as st


class FuturePredictionPanel:
    def __init__(self):
        # This class only displays future prediction results
        pass

    def render(self, ml_result):
        # Only show this panel for future days
        if ml_result is None or not ml_result.is_future_day():
            return

        st.markdown(
            "<h3 style='color:#0f172a; font-weight:700;'>Future Forecast (ML Prediction)</h3>",
            unsafe_allow_html=True
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        def render_card(title, value, unit, col):
            shown_value = 0.0 if value is None else value

            with col:
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
                        <div style="font-size:14px; font-weight:600; color:#334155; margin-bottom:10px;">
                            {title}
                        </div>
                        <div style="font-size:30px; font-weight:700; color:#0f172a; margin-bottom:8px;">
                            {shown_value:.1f}
                        </div>
                        <div style="font-size:13px; color:#475569;">
                            {unit}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        render_card("Predicted Temp", ml_result.get_temperature(), "°C", col1)
        render_card("Predicted Rain", ml_result.get_rain(), "mm", col2)
        render_card("Predicted Humidity", ml_result.get_humidity(), "%", col3)
        render_card("Predicted Wind", ml_result.get_wind(), "km/h", col4)
        render_card("Predicted Pressure", ml_result.get_pressure(), "hPa", col5)
