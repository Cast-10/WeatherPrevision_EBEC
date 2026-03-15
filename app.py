import streamlit as st

from visualizer.data_loader import DataLoader
from visualizer.weather_service import WeatherService
from visualizer.weather_selector import WeatherSelector
from visualizer.weather_summary_cards import WeatherSummaryCards
from visualizer.hourly_weather_timeline import HourlyWeatherTimeline
from visualizer.app_styler import AppStyler
from visualizer.main_page import MainPage
from visualizer.future_prediction_panel import FuturePredictionPanel
from visualizer.ml_service import MLService

st.set_page_config(page_title="Weather Interface", layout="wide")

# Load the dataset
loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

# Create services
weather_service = WeatherService(df)
ml_service = MLService(df)

# Create UI helpers
future_panel = FuturePredictionPanel()
styler = AppStyler(
    title="Weather Forecast Interface",
    subtitle="Explore hourly weather conditions and future forecasts by district across Portugal"
)
main_page = MainPage()

# Apply visual style
styler.apply_styles()
styler.render_header()

# Render main page intro
main_page.render_intro()

# Render selector section
styler.open_selector_box()
selector = WeatherSelector(weather_service, future_days=7)
selected_location, selected_day = selector.render()
styler.close_box()

# Build ML result only after both values are selected
ml_result = None
if selected_location is not None and selected_day is not None:
    ml_result = ml_service.build_result(selected_location, selected_day)

# Render content only after full selection
if selected_location is not None and selected_day is not None:
    if ml_result is not None and ml_result.is_future_day():
        future_panel.render(ml_result)
    else:
        summary = WeatherSummaryCards(weather_service)
        summary.render(selected_location, selected_day)

        styler.open_timeline_box("Hourly weather timeline")
        timeline = HourlyWeatherTimeline(weather_service)
        timeline.render(selected_location, selected_day)
        styler.close_box()
else:
    main_page.render_empty_state()
