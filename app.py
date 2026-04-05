from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

model  = pickle.load(open('models/xgb_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Exact column order the scaler was fit on — do NOT change this
FEATURE_ORDER = [
    'year', 'month', 'day', 'hour', 'day_of_week',
    'is_weekend', 'season', 'city', 'station',
    'temperature', 'humidity', 'wind_speed'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data], columns=FEATURE_ORDER)
    df_scaled = scaler.transform(df)
    aqi = float(model.predict(df_scaled)[0])
    aqi = max(25, min(500, round(aqi)))
    return jsonify({'aqi': aqi})

if __name__ == '__main__':
    app.run(debug=True, port=5000)