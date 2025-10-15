import os
import numpy as np
import pandas as pd
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- Firebase Initialization ---
# Ensure you have your 'serviceAccountKey.json' in the same directory
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"🔥 Firebase initialization failed: {e}")
    print("🚨 Please make sure 'serviceAccountKey.json' is present and valid.")
    db = None

# --- Flask App Initialization ---
app = Flask(__name__)
# Allowing all origins for development purposes.
# For production, you should restrict this to your frontend's domain.
CORS(app)

# --- Global Variables & Constants ---
MODEL_CACHE = {} # Cache for storing trained models to avoid retraining on every call
DATA_LOOKBACK = 90  # Number of past days to use for prediction
PREDICTION_DAYS = 60 # Number of days of historical data to feed into the model sequence
CRYPTO_IDS = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'sol': 'solana',
    'ada': 'cardano',
    'doge': 'dogecoin'
}

# --- Machine Learning Model ---
def create_lstm_model(input_shape):
    """
    Creates and compiles a new LSTM model.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_or_train_model(crypto_id, scaled_data):
    """
    Retrieves a model from cache or trains a new one if not present.
    In a real-world scenario, you would save/load trained models from disk.
    For this example, we train a new model for each new crypto requested
    and cache it in memory for the server's lifetime.
    """
    if crypto_id in MODEL_CACHE:
        return MODEL_CACHE[crypto_id]

    # --- Data Preparation for LSTM ---
    x_train, y_train = [], []
    for i in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[i-PREDICTION_DAYS:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape data for LSTM [samples, timesteps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # --- Model Creation and Training ---
    model = create_lstm_model(input_shape=(x_train.shape[1], 1))
    # Note: epochs=1 is for quick demonstration. For real results, use more epochs (e.g., 25-50)
    model.fit(x_train, y_train, batch_size=32, epochs=1)

    MODEL_CACHE[crypto_id] = model
    return model

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict the next day's price for a given cryptocurrency.
    """
    data = request.get_json()
    if not data or 'crypto_id' not in data:
        return jsonify({'error': 'Invalid request. "crypto_id" is required.'}), 400

    crypto_id = data['crypto_id']
    if crypto_id not in CRYPTO_IDS.values():
         return jsonify({'error': f'Invalid cryptocurrency ID: {crypto_id}'}), 400

    # --- 1. Fetch Historical Data from CoinGecko ---
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': DATA_LOOKBACK, 'interval': 'daily'}
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        history = response.json()
        prices = history['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        if df.empty:
             return jsonify({'error': f'No historical data found for {crypto_id}'}), 404

        price_data = df[['price']].values

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Failed to fetch data from CoinGecko API: {e}'}), 500

    # --- 2. Scale and Prepare Data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price_data = scaler.fit_transform(price_data)

    # --- 3. Get or Train Model ---
    try:
        model = get_or_train_model(crypto_id, scaled_price_data)
    except Exception as e:
         return jsonify({'error': f'Model training failed: {e}'}), 500


    # --- 4. Make Prediction ---
    # Prepare the last `PREDICTION_DAYS` days from historical data as input
    last_sequence = scaled_price_data[-PREDICTION_DAYS:].reshape(1, PREDICTION_DAYS, 1)

    predicted_price_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

    # --- 5. Save Prediction to Firestore ---
    if db:
        try:
            prediction_ref = db.collection('predictions').document()
            prediction_ref.set({
                'crypto_id': crypto_id,
                'predicted_price_usd': float(predicted_price),
                'prediction_timestamp_utc': datetime.utcnow()
            })
        except Exception as e:
            # Log the error but don't fail the request if Firestore write fails
            print(f"🔥 Warning: Failed to save prediction to Firestore: {e}")


    # --- 6. Format and Return Response ---
    # Prepare historical data for the chart
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
    historical_for_chart = df[['date', 'price']].to_dict(orient='records')

    response_data = {
        'crypto_id': crypto_id,
        'predicted_price': round(float(predicted_price), 2),
        'historical_data': historical_for_chart
    }
    return jsonify(response_data)

@app.route('/')
def index():
    return "<h1>Crypto Prediction Backend is Running!</h1>"

if __name__ == '__main__':
    # Use environment variable for port if available, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
