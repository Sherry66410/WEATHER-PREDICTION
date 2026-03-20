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

# Your exact 13 features from the notebook
FEATURES = [
    'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR',
    'temp_lag1', 'temp_lag2', 'temp_lag3',
    'temp_roll3', 'temp_roll7',
    'month', 'day', 'day_of_week', 'week_of_year'
]

# ── Pure numpy LSTM forward pass (no TensorFlow needed) ──────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lstm_predict(x_scaled, lk, lrk, lb, dk, db):
    """Run your exact LSTM(32) + Dense(1) forward pass using numpy"""
    units = lk.shape[1] // 4  # 32

    h = np.zeros((1, units))
    c = np.zeros((1, units))

    b_i, b_f, b_c, b_o = np.split(lb, 4)

    gates = x_scaled @ lk + h @ lrk
    i_g, f_g, c_g, o_g = np.split(gates, 4, axis=1)

    i_g = sigmoid(i_g + b_i)
    f_g = sigmoid(f_g + b_f)
    c_g = np.tanh(c_g  + b_c)
    o_g = sigmoid(o_g  + b_o)

    c = f_g * c + i_g * c_g
    h = o_g * np.tanh(c)

    return float(h @ dk + db)

# ── Load all artifacts ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    base = 'model_artifacts'
    if not os.path.exists(base):
        st.error(f"'{base}' folder not found!")
        return None
    try:
        scaler_X = joblib.load(f'{base}/scaler_X.pkl')
        scaler_y = joblib.load(f'{base}/scaler_y.pkl')
        df       = pd.read_csv(f'{base}/processed_weather_data.csv',
                               index_col=0, parse_dates=True)
        weights  = {
            'lk':  np.load(f'{base}/lstm_kernel.npy'),
            'lrk': np.load(f'{base}/lstm_recurrent_kernel.npy'),
            'lb':  np.load(f'{base}/lstm_bias.npy'),
            'dk':  np.load(f'{base}/dense_kernel.npy'),
            'db':  np.load(f'{base}/dense_bias.npy'),
        }
        return scaler_X, scaler_y, df, weights
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

# ── Your exact classify_weather from notebook ────────────────────────────────
def classify_weather(temp, rain):
    if rain > 5:
        return "Rainy", "☔️", "Bring umbrella! 🌂"
    elif rain > 1:
        return "Cloudy", "☁️", "Might need a jacket 🧥"
    elif temp > 32:
        return "Hot", "☀️", "Stay hydrated! 💧"
    else:
        return "Pleasant", "😊", "Perfect day! 🌟"

# ── Your exact predict_tomorrow logic from notebook ───────────────────────────
def predict_tomorrow(today_input, scaler_X, scaler_y, df, weights):
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

        today_df  = today_df[FEATURES]
        scaled    = scaler_X.transform(today_df)  # (1, 13)

        pred_scaled = lstm_predict(
            scaled,
            weights['lk'], weights['lrk'], weights['lb'],
            weights['dk'], weights['db']
        )
        pred_temp = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        return float(pred_temp)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ── Load ──────────────────────────────────────────────────────────────────────
result = load_artifacts()
if result is None:
    st.stop()
scaler_X, scaler_y, df, weights = result

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
        temp = predict_tomorrow(today_input, scaler_X, scaler_y, df, weights)
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
st.markdown("📊 Model: LSTM(32) + Dropout(0.4) + Dense(1) · Trained on NASA/POWER Kerala data · R²: 0.918 · MAE: 0.368°C")
