import streamlit as st

from visualizer.data_loader import DataLoader
from visualizer.weather_service import WeatherService
from visualizer.weather_selector import WeatherSelector
from visualizer.weather_summary_cards import WeatherSummaryCards
from visualizer.hourly_weather_timeline import HourlyWeatherTimeline
from visualizer.app_styler import AppStyler
from visualizer.main_page import MainPage
from visualizer.future_prediction_panel import FuturePredictionPanel

st.set_page_config(page_title="Weather Interface", layout="wide")

# Load the dataset
loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

# Create the weather service
weather_service = WeatherService(df)

# Create the future prediction panel
future_panel = FuturePredictionPanel(df)

# Create and apply the visual style
styler = AppStyler(
    title="Weather Forecast Interface",
    subtitle="Explore hourly weather conditions and future forecasts by district across Portugal"
)
styler.apply_styles()
styler.render_header()

# Create the main page structure
main_page = MainPage()
main_page.render_intro()

# Show the selector section
styler.open_selector_box()
selector = WeatherSelector(weather_service, future_days=7)
selected_location, selected_day = selector.render()
styler.close_box()

# Only continue after both fields are selected
if selected_location is not None and selected_day is not None:
    if future_panel.is_future_day(selected_day):
        future_panel.render(selected_location, selected_day)
    else:
        summary = WeatherSummaryCards(weather_service)
        summary.render(selected_location, selected_day)

        styler.open_timeline_box("Hourly weather timeline")
        timeline = HourlyWeatherTimeline(weather_service)
        timeline.render(selected_location, selected_day)
        styler.close_box()
else:
    main_page.render_empty_state()
