# 🌤️ Kerala Next-Day Weather Prediction

An LSTM-based machine learning app that predicts tomorrow's temperature and weather condition based on today's inputs.

## Features
- Predicts next-day temperature using an LSTM model
- Classifies weather as Rainy / Cloudy / Hot / Pleasant
- Trained on NASA/POWER historical weather data for Kerala (1981–2026)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## Model Artifacts

| File | Description |
|------|-------------|
| `model_artifacts/best_lstm_model.keras` | Trained LSTM model |
| `model_artifacts/scaler_X.pkl` | Feature scaler (MinMaxScaler) |
| `model_artifacts/scaler_y.pkl` | Target scaler (MinMaxScaler) |
| `model_artifacts/processed_weather_data.csv` | Processed historical data |

## Tech Stack
- TensorFlow 2.13 (CPU)
- Scikit-learn 1.2.2
- Streamlit
- NumPy 1.24.3 / Pandas
