import streamlit as st
import joblib
import pandas as pd

from visualizer.data_loader import DataLoader
from visualizer.weather_service import WeatherService
from visualizer.weather_selector import WeatherSelector
from visualizer.weather_summary_cards import WeatherSummaryCards
from visualizer.hourly_weather_timeline import HourlyWeatherTimeline
from visualizer.app_styler import AppStyler
from visualizer.main_page import MainPage
from visualizer.future_prediction_panel import FuturePredictionPanel
from visualizer.accident_forecast_panel import AccidentForecastPanel
from visualizer.meteorology_forecast_panel import MeteorologyForecastPanel
from visualizer.ml_service import MLService

st.set_page_config(page_title="Weather Interface", layout="wide")

# --- LOAD DATA ---
loader = DataLoader("metherology_dataset.csv")
df = loader.load_data()

accidents_df = pd.read_csv("accidents_dataset.csv")

# --- LOAD MODELS ---
temperature_model = joblib.load("finalModelLevel2.pkl")
level4_model = joblib.load("finalModelLevel4.pkl")
level5_model = joblib.load("finalModelLevel5.pkl")
level5_labels = joblib.load("labelsLevel5.pkl")
level5_features = joblib.load("featuresLevel5.pkl")

# --- SERVICES ---
weather_service = WeatherService(df)
ml_service = MLService(
    df,
    accidents_df=accidents_df,
    temperature_model=temperature_model,
    level4_model=level4_model,
    level5_model=level5_model,
    level5_labels=level5_labels,
    level5_features=level5_features
)

# --- PANELS ---
future_panel = FuturePredictionPanel()
accident_panel = AccidentForecastPanel()
meteorology_panel = MeteorologyForecastPanel()

# --- UI SETUP ---
styler = AppStyler(
    title="Weather Forecast Interface",
    subtitle="Explore hourly weather conditions and future forecasts by district across Portugal"
)
main_page = MainPage()

styler.apply_styles()
styler.render_header()
main_page.render_intro()

# --- SELECTOR ---
styler.open_selector_box()
selector = WeatherSelector(weather_service, future_days=7)
selected_location, selected_day = selector.render()
styler.close_box()

# --- BUILD RESULT ---
ml_result = None
if selected_location is not None and selected_day is not None:
    ml_result = ml_service.build_result(selected_location, selected_day)

# --- RENDER PAGE ---
if selected_location is not None and selected_day is not None and ml_result is not None:
    accident_panel.render(ml_result)

    if ml_result.is_future_day():
        meteorology_panel.render(ml_result)
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