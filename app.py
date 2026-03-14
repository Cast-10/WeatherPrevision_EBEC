import streamlit as st
from visualizer.data_loader import DataLoader
from visualizer.weather_service import WeatherService
from visualizer.weather_selector import WeatherSelector

st.set_page_config(page_title="Weather Interface", layout="wide")

# Load the dataset
loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

# Create the weather filtering service
weather_service = WeatherService(df)

# Create and show the selector component
selector = WeatherSelector(weather_service, future_days=7)
selected_location, selected_day = selector.render()

# Show what the user selected
st.write("Selected district:", selected_location)
st.write("Selected day:", selected_day)
