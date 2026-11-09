import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import joblib

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Predictor", layout="wide")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Model"

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_DIR / "Stock_Prediction_Model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.save")

model = load_trained_model()
scaler = load_scaler()

st.header("Stock Market Predictor")

stock = st.text_input("Enter Stock Symbol", "GOOG").upper().strip()

if st.button("Predict"):

    data = yf.download(
        stock,
        start="2016-10-22",
        end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )

    if data.empty:
        st.error("No stock data found.")
    else:
        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        st.subheader("Raw Data")
        st.write(data)

        if len(close) < 100:
            st.error("Insufficient data. At least 100 trading days are required.")
        else:
            last_100_days = close.tail(100).values.reshape(-1, 1)

            scaled_data = scaler.transform(last_100_days)

            x = np.array([scaled_data])

            prediction = model.predict(x, verbose=0)

            predicted_price = scaler.inverse_transform(prediction)[0][0]

            st.subheader("Stock Price Prediction")

            st.metric(
                "Predicted next trading-day price",
                f"${predicted_price:.2f}"
            )

            st.metric(
                "Latest closing price",
                f"${float(close.iloc[-1]):.2f}"
            )