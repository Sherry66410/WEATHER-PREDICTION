import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Kerala Weather Prediction",
    page_icon="🌤️",
    layout="wide"
)

# Exact 13 features from your notebook
FEATURES = [
    'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR',
    'temp_lag1', 'temp_lag2', 'temp_lag3',
    'temp_roll3', 'temp_roll7',
    'month', 'day', 'day_of_week', 'week_of_year'
]

@st.cache_resource(show_spinner="Loading model...")
def load_model_artifacts():
    base = 'model_artifacts'
    if not os.path.exists(base):
        st.error(f"'{base}' folder not found!")
        return None
    try:
        model    = joblib.load(f'{base}/mlp_model.pkl')
        scaler_X = joblib.load(f'{base}/scaler_X.pkl')
        scaler_y = joblib.load(f'{base}/scaler_y.pkl')
        df       = pd.read_csv(f'{base}/processed_weather_data.csv',
                               index_col=0, parse_dates=True)
        return model, scaler_X, scaler_y, df
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

# Exact classify_weather from your notebook
def classify_weather(temp, rain):
    if rain > 5:
        return "Rainy", "☔️", "Bring umbrella! 🌂"
    elif rain > 1:
        return "Cloudy", "☁️", "Might need a jacket 🧥"
    elif temp > 32:
        return "Hot", "☀️", "Stay hydrated! 💧"
    else:
        return "Pleasant", "😊", "Perfect day! 🌟"

# Exact predict_tomorrow logic from your notebook
def predict_tomorrow(today_input, model, scaler_X, scaler_y, df):
    try:
        new_date = df.index[-1] + pd.Timedelta(days=1)
        today_df = pd.DataFrame([today_input], index=[new_date])

        today_df['temp_lag1']    = df['T2M'].iloc[-1]
        today_df['temp_lag2']    = df['T2M'].iloc[-2]
        today_df['temp_lag3']    = df['T2M'].iloc[-3]
        today_df['temp_roll3']   = df['T2M'].iloc[-3:].mean()
        today_df['temp_roll7']   = df['T2M'].iloc[-7:].mean()
        today_df['month']        = new_date.month
        today_df['day']          = new_date.day
        today_df['day_of_week']  = new_date.dayofweek
        today_df['week_of_year'] = int(new_date.isocalendar()[1])

        today_df    = today_df[FEATURES]
        scaled      = scaler_X.transform(today_df)
        pred_scaled = model.predict(scaled).reshape(-1, 1)
        pred_temp   = scaler_y.inverse_transform(pred_scaled)
        return float(pred_temp[0][0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ── Load ──────────────────────────────────────────────────────────────────────
result = load_model_artifacts()
if result is None:
    st.stop()
best_model, scaler_X, scaler_y, df = result

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌤️ Kerala Next-Day Weather Prediction")
st.write("Enter today's weather to predict tomorrow's temperature and conditions.")

st.sidebar.header("🌡️ Today's Weather Parameters")
t2m         = st.sidebar.slider("Temperature (°C)",  10.0,  45.0, 27.0, 0.1)
rh2m        = st.sidebar.slider("Humidity (%)",       0.0, 100.0, 75.0, 0.1)
ws2m        = st.sidebar.slider("Wind Speed (m/s)",   0.0,  20.0,  2.5, 0.1)
prectotcorr = st.sidebar.slider("Precipitation (mm)", 0.0, 200.0,  0.0, 0.1)

today_input = {
    'T2M': t2m, 'RH2M': rh2m,
    'WS2M': ws2m, 'PRECTOTCORR': prectotcorr
}

st.subheader("📝 Today's Input")
st.dataframe(pd.DataFrame([{
    "Temperature (°C)":   t2m,
    "Humidity (%)":       rh2m,
    "Wind Speed (m/s)":   ws2m,
    "Precipitation (mm)": prectotcorr,
}]), use_container_width=True)

if st.button("🔮 Predict Tomorrow's Weather", type="primary"):
    with st.spinner("Calculating..."):
        temp = predict_tomorrow(today_input, best_model, scaler_X, scaler_y, df)
    if temp is not None:
        label, icon, advice = classify_weather(temp, prectotcorr)
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ Predicted Temperature", f"{temp:.1f} °C")
        c2.metric("☁️ Weather",               f"{icon} {label}")
        c3.metric("💡 Advice",                advice)
        st.success("✅ Prediction complete!")

with st.expander("📈 Historical Temperature (last 90 days)"):
    st.line_chart(df['T2M'].iloc[-90:])

st.markdown("---")
st.markdown("📊 Model: LSTM(96) + Dropout(0.4) · R²: 0.918 · MAE: 0.368°C · Trained on NASA/POWER Kerala data")
