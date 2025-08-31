# app.py
import streamlit as st
import requests
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Set page config
st.set_page_config(page_title="🌧️ CloudCastML", page_icon="🌦️", layout="centered")


load_dotenv()
WEATHER_API_KEY = os.getenv("WeatherAPI_Key")




# Load trained model
model = joblib.load("raincast_model.pkl")

# Description to category mapping
desc_map = {
    "clear sky": 0, "few clouds": 0, "scattered clouds": 0, "broken clouds": 0,
    "overcast clouds": 1, "shower rain": 1, "rain": 1, "light rain": 1,
    "moderate rain": 1, "thunderstorm": 1, "snow": 1, "mist": 1
}

# Fetch weather data
def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if "main" not in data or "sys" not in data:
        return None, None

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    pressure = data["main"]["pressure"]
    wind = data["wind"]["speed"]
    clouds = data["clouds"]["all"]
    desc = data["weather"][0]["description"]
    desc_cat = desc_map.get(desc.lower(), 0)
    sunrise = data["sys"].get("sunrise", 0)
    sunset = data["sys"].get("sunset", 0)

    # ✅ Pass all 8 features
    features = [temp, humidity, pressure, wind, clouds, sunrise, sunset, desc_cat]
    return features, data

# App title
st.markdown("<h1 style='text-align: center;'>🌦️ CloudCastML</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Forecast rainfall in real-time using live Weather Data and ML algorithms</h4>", unsafe_allow_html=True)

# User input
city = st.text_input("🔍 Enter a city name:")

if city:
    features, raw = fetch_weather(city)
    if features:
        df = pd.DataFrame([features], columns=[
            "Temperature", "Humidity", "Pressure", "Wind Speed", "Clouds",
            "Sunrise", "Sunset", "DescriptionCategory"
        ])
        prediction = model.predict(df)[0]

        # Background and result message
        bg_color = "#0c8fa0" if prediction == 0 else "#0a3553"
        result_msg = "☀️ No Rain Expected" if prediction == 0 else "🌧️ Rain Expected"
        st.markdown(f"<div style='background-color:{bg_color}; padding: 20px; border-radius: 10px;'>"
                    f"<h2 style='text-align:center;'>{result_msg}</h2></div>", unsafe_allow_html=True)

        # Weather metrics
        st.markdown("### 📊 Weather Details")
        col1, col2 = st.columns(2)
        col1.metric("🌡️ Temperature", f"{features[0]} °C")
        col2.metric("💧 Humidity", f"{features[1]} %")
        col1.metric("📈 Pressure", f"{features[2]} hPa")
        col2.metric("💨 Wind Speed", f"{features[3]} m/s")
        col1.metric("☁️ Cloud Cover", f"{features[4]} %")
        col2.metric("📝 Description", raw["weather"][0]["description"].title())

        # Optional chart
        chart_data = pd.DataFrame({
            'Feature': ['Temp (°C)', 'Humidity (%)', 'Pressure (hPa)', 'Wind (m/s)', 'Clouds (%)'],
            'Value': features[:5]
        })
        st.markdown("### 📈 Visual Summary")
        st.bar_chart(chart_data.set_index("Feature"))

    else:
        st.error("⚠️ City not found or API issue. Try again.")
