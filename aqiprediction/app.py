from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'LSTM_model.keras'
model = load_model(MODEL_PATH)

scaler = joblib.load('scaler.pkl')
COLS = ['aqi', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'pressure_msl', 'precipitation', 'cloud_cover', 'pm2.5']
AQI_MIN, AQI_MAX = scaler.data_min_[0], scaler.data_max_[0]

LAT, LON = 37.348497, -121.894898

_CACHE = {}
_CACHE_TTL = 1800  # 30 minutes

def cached_get(url, params, timeout=15):
    key = (url, tuple(sorted(params.items())))
    now = time.time()
    if key in _CACHE:
        ts, data = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return data
    resp = requests.get(url, params=params, timeout=timeout).json()
    _CACHE[key] = (now, resp)
    return resp


def scale_value(col_name, value):
    idx = COLS.index(col_name)
    lo, hi = scaler.data_min_[idx], scaler.data_max_[idx]
    return (value - lo) / (hi - lo)


def daily_means(times, values):
    buckets = {}
    for t, v in zip(times, values):
        if v is None:
            continue
        d = t[:10]
        buckets.setdefault(d, []).append(v)
    return {d: float(np.mean(v)) for d, v in buckets.items()}


@app.route('/live-aqi', methods=['GET'])
def live_aqi():
    try:
        today = datetime.now(ZoneInfo('America/Los_Angeles')).date()
        yesterday = today - timedelta(days=1)
        day_before = today - timedelta(days=2)

        weather_resp = cached_get(
            'https://api.open-meteo.com/v1/forecast',
            {
                'latitude': LAT, 'longitude': LON,
                'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation,cloud_cover',
                'past_days': 3, 'forecast_days': 1, 'timezone': 'America/Los_Angeles',
            },
        )

        aq_resp = cached_get(
            'https://air-quality-api.open-meteo.com/v1/air-quality',
            {
                'latitude': LAT, 'longitude': LON,
                'hourly': 'pm2_5,us_aqi',
                'past_days': 3, 'forecast_days': 1, 'timezone': 'America/Los_Angeles',
            },
        )

        if 'hourly' not in weather_resp or 'hourly' not in aq_resp:
            return jsonify({'error': 'missing hourly key', 'weather_resp': weather_resp, 'aq_resp': aq_resp}), 500
        wh = weather_resp['hourly']
        aqh = aq_resp['hourly']

        temp_d = daily_means(wh['time'], wh['temperature_2m'])
        rh_d = daily_means(wh['time'], wh['relative_humidity_2m'])
        wind_d = daily_means(wh['time'], wh['wind_speed_10m'])
        pres_d = daily_means(wh['time'], wh['pressure_msl'])
        precip_d = daily_means(wh['time'], wh['precipitation'])
        cloud_d = daily_means(wh['time'], wh['cloud_cover'])
        pm25_d = daily_means(aqh['time'], aqh['pm2_5'])
        aqi_d = daily_means(aqh['time'], aqh['us_aqi'])

        t_str, y_str, db_str = str(today), str(yesterday), str(day_before)

        aqi_t1 = aqi_d.get(y_str)
        aqi_t2 = aqi_d.get(db_str)
        if aqi_t1 is None or aqi_t2 is None:
            return jsonify({'error': 'Not enough recent AQI history yet from the data source.'}), 503

        raw = {
            'temperature_2m': temp_d.get(t_str),
            'relative_humidity_2m': rh_d.get(t_str),
            'wind_speed_10m': wind_d.get(t_str),
            'pressure_msl': pres_d.get(t_str),
            'precipitation': precip_d.get(t_str),
            'cloud_cover': cloud_d.get(t_str),
            'pm2.5': pm25_d.get(t_str),
        }
        if any(v is None for v in raw.values()):
            return jsonify({'error': "Today's weather data isn't fully available yet."}), 503

        feature_order = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                          'pressure_msl', 'precipitation', 'cloud_cover', 'pm2.5']
        scaled = [scale_value(c, raw[c]) for c in feature_order]
        scaled.append(np.clip((aqi_t1 - AQI_MIN) / (AQI_MAX - AQI_MIN), 0, 1))
        scaled.append(np.clip((aqi_t2 - AQI_MIN) / (AQI_MAX - AQI_MIN), 0, 1))

        X = np.array(scaled).reshape((1, 1, 9))
        pred_scaled = model.predict(X)[0][0]
        pred_aqi = float(pred_scaled) * (AQI_MAX - AQI_MIN) + AQI_MIN

        return jsonify({
            'date': t_str,
            'location': 'San Jose - Jackson, CA',
            'live_aqi_estimate': aqi_d.get(t_str),
            'model_predicted_aqi': round(pred_aqi, 1),
            'pm2_5': round(raw['pm2.5'], 2),
            'temperature_c': round(raw['temperature_2m'], 1),
            'humidity_pct': round(raw['relative_humidity_2m'], 1),
            'wind_speed_ms': round(raw['wind_speed_10m'], 2),
            'pressure_hpa': round(raw['pressure_msl'], 1),
            'precipitation_mm': round(raw['precipitation'], 2),
            'cloud_cover_pct': round(raw['cloud_cover'], 1),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/forecast-7day', methods=['GET'])
def forecast_7day():
    try:
        tz = ZoneInfo('America/Los_Angeles')
        today = datetime.now(tz).date()

        weather_resp = cached_get(
            'https://api.open-meteo.com/v1/forecast',
            {
                'latitude': LAT, 'longitude': LON,
                'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation,cloud_cover',
                'past_days': 2, 'forecast_days': 7, 'timezone': 'America/Los_Angeles',
            },
        )

        aq_resp = cached_get(
            'https://air-quality-api.open-meteo.com/v1/air-quality',
            {
                'latitude': LAT, 'longitude': LON,
                'hourly': 'pm2_5,us_aqi',
                'past_days': 2, 'forecast_days': 7, 'timezone': 'America/Los_Angeles',
            },
        )

        if 'hourly' not in weather_resp or 'hourly' not in aq_resp:
            return jsonify({'error': 'missing hourly key', 'weather_resp': weather_resp, 'aq_resp': aq_resp}), 500
        wh = weather_resp['hourly']
        aqh = aq_resp['hourly']

        temp_d = daily_means(wh['time'], wh['temperature_2m'])
        rh_d = daily_means(wh['time'], wh['relative_humidity_2m'])
        wind_d = daily_means(wh['time'], wh['wind_speed_10m'])
        pres_d = daily_means(wh['time'], wh['pressure_msl'])
        precip_d = daily_means(wh['time'], wh['precipitation'])
        cloud_d = daily_means(wh['time'], wh['cloud_cover'])
        pm25_d = daily_means(aqh['time'], aqh['pm2_5'])
        aqi_d = daily_means(aqh['time'], aqh['us_aqi'])

        dates = [today + timedelta(days=i) for i in range(-2, 7)]
        date_strs = [str(d) for d in dates]

        def scaled_aqi(date_str):
            v = aqi_d.get(date_str)
            if v is None:
                return None
            return float(np.clip((v - AQI_MIN) / (AQI_MAX - AQI_MIN), 0, 1))

        aqi_lag2 = scaled_aqi(date_strs[0])
        aqi_lag1 = scaled_aqi(date_strs[1])
        if aqi_lag1 is None or aqi_lag2 is None:
            return jsonify({'error': 'Not enough recent AQI history yet from the data source.'}), 503

        feature_order = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                          'pressure_msl', 'precipitation', 'cloud_cover', 'pm2.5']

        results = []
        for i in range(9):
            if len(results) >= 7:
                break
            d_str = date_strs[i + 2]
            raw = {
                'temperature_2m': temp_d.get(d_str),
                'relative_humidity_2m': rh_d.get(d_str),
                'wind_speed_10m': wind_d.get(d_str),
                'pressure_msl': pres_d.get(d_str),
                'precipitation': precip_d.get(d_str),
                'cloud_cover': cloud_d.get(d_str),
                'pm2.5': pm25_d.get(d_str),
            }
            if any(v is None for v in raw.values()):
                break

            scaled = [scale_value(c, raw[c]) for c in feature_order]
            scaled.append(aqi_lag1)
            scaled.append(aqi_lag2)

            X = np.array(scaled).reshape((1, 1, 9))
            pred_scaled = float(model.predict(X, verbose=0)[0][0])
            pred_aqi = pred_scaled * (AQI_MAX - AQI_MIN) + AQI_MIN

            results.append({'date': d_str, 'predicted_aqi': round(pred_aqi, 1)})

            aqi_lag2 = aqi_lag1
            aqi_lag1 = float(np.clip(pred_scaled, 0, 1))

        return jsonify({'location': 'San Jose - Jackson, CA', 'forecast': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return "Welcome to the AQI Prediction API!"


if __name__ == '__main__':
    import os
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
