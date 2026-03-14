import streamlit as st

from visualizer.data_loader import DataLoader
from visualizer.weather_service import WeatherService
from visualizer.weather_selector import WeatherSelector
from visualizer.hourly_weather_timeline import HourlyWeatherTimeline

st.set_page_config(page_title="Weather Interface", layout="wide")

# Load the dataset
loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

# Create the service
weather_service = WeatherService(df)

# Create the selector
selector = WeatherSelector(weather_service, future_days=7)
selected_location, selected_day = selector.render()

# Create the hourly timeline
timeline = HourlyWeatherTimeline(weather_service)
timeline.render(selected_location, selected_day)
