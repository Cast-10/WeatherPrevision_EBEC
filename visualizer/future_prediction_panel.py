import streamlit as st
import pandas as pd


class FuturePredictionPanel:
    def __init__(self, df: pd.DataFrame):
        # Save the dataset to know the last historical day available
        self.df = df.copy()

    def get_last_available_day(self):
        # Return the last day that exists in the historical dataset
        return self.df["time"].dt.date.max()

    def is_future_day(self, selected_day) -> bool:
        # Check if the selected day is after the last historical day
        return selected_day > self.get_last_available_day()

    def predict(self, location: str, selected_day):
        # This method will later be replaced by the real ML prediction logic
        pass

    def get_placeholder_prediction(self) -> dict:
        # Temporary values until the ML model is connected
        return {
            "temperature": 0.0,
            "rain": 0.0,
            "humidity": 0.0,
            "wind": 0.0,
            "pressure": 0.0
        }

    def render(self, location: str, selected_day):
        # Show a prediction panel only for future days
        if not self.is_future_day(selected_day):
            return

        prediction = self.get_placeholder_prediction()

        st.markdown(
            "<h3 style='color:#0f172a; font-weight:700;'>Future Forecast (ML Prediction)</h3>",
            unsafe_allow_html=True
        )

        st.info(
            "The selected date is in the future. For now, this panel shows placeholder values until the ML model is connected."
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        def render_card(title, value, unit, col):
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
                            {value:.1f}
                        </div>
                        <div style="font-size:13px; color:#475569;">
                            {unit}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        render_card("Predicted Temp", prediction["temperature"], "°C", col1)
        render_card("Predicted Rain", prediction["rain"], "mm", col2)
        render_card("Predicted Humidity", prediction["humidity"], "%", col3)
        render_card("Predicted Wind", prediction["wind"], "km/h", col4)
        render_card("Predicted Pressure", prediction["pressure"], "hPa", col5)
