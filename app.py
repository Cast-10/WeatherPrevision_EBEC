import streamlit as st

from visualizer.data_loader import DataLoader
from visualizer.weather_service import WeatherService
from visualizer.weather_selector import WeatherSelector
from visualizer.weather_summary_cards import WeatherSummaryCards
from visualizer.hourly_weather_timeline import HourlyWeatherTimeline
from visualizer.app_styler import AppStyler

st.set_page_config(page_title="Weather Interface", layout="wide")

# Load the dataset
loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

# Create the weather service
weather_service = WeatherService(df)

# Create and apply the visual style
styler = AppStyler(
    title="Weather Forecast Interface",
    subtitle="Select a district and a day to explore historical data or future predictions"
)
styler.apply_styles()
styler.render_header()

# Show the selector section inside a styled box
styler.open_selector_box()
selector = WeatherSelector(weather_service, future_days=7)
selected_location, selected_day = selector.render()
styler.close_box()

# Show the daily summary
summary = WeatherSummaryCards(weather_service)
summary.render(selected_location, selected_day)

# Show the hourly timeline inside a styled lower box
styler.open_timeline_box("Hourly weather timeline")
timeline = HourlyWeatherTimeline(weather_service)
timeline.render(selected_location, selected_day)
styler.close_box()
