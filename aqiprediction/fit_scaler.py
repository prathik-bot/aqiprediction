import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

cols_to_drop = ['latitude', 'longitude', 'city']
cols = ['aqi', 'temperature_2m', 'relative_humidity_2m',
        'wind_speed_10m', 'pressure_msl', 'precipitation', 'cloud_cover', 'pm2.5']

df = pd.read_csv('Data/Final/San_Jose___Jackson.csv')
df.drop(cols_to_drop, axis=1, inplace=True)
df.set_index(['date'], inplace=True)
df = df[cols]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[cols])

joblib.dump(scaler, 'scaler.pkl')
print("Saved scaler.pkl")
print("Column order:", cols)
print("data_min_:", scaler.data_min_.tolist())
print("data_max_:", scaler.data_max_.tolist())
