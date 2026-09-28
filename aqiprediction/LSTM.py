#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
# from keras import optimizers
from tensorflow.keras import optimizers 

from sklearn.model_selection import train_test_split


# In[ ]:


cols_to_drop = ['latitude', 'longitude', 'city']
cols = ['aqi', 'temperature_2m', 'relative_humidity_2m', 
       'wind_speed_10m', 'pressure_msl', 'precipitation', 'cloud_cover', 'pm2.5']


# In[ ]:


df = pd.read_csv('Data/Final/San_Jose___Jackson.csv')
df.drop(cols_to_drop, axis=1, inplace=True)
df.set_index(['date'], inplace=True)
df = df[cols]
df.head()


# ### Normalize the dataset

# In[ ]:


minimum = []
maximum = []
minimum.append(df['aqi'].min())
maximum.append(df['aqi'].max())

cols = df.columns.values.tolist()

scaler = MinMaxScaler(feature_range=(0, 1))
df[cols] = scaler.fit_transform(df[cols])

df.head()


# In[ ]:


def create_lagged_features(df, lag_steps, target_col='pm2.5'):
    for lag in range(1, lag_steps + 1):
        df[f'{target_col}_(t-{lag})'] = df[target_col].shift(lag)
    return df


lag_steps = 2

df = create_lagged_features(df, lag_steps, target_col='aqi')

df = df.dropna()
df.head()


# In[ ]:


correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


X = df.drop(columns=['aqi']).values
y = df['aqi'].values

temp1 = int(0.8*len(X))
temp2 = int(0.1*len(X))
X_train = X[:temp1]
X_val = X[temp1:temp1+temp2]
X_test = X[temp1+temp2:]

y_train = y[:temp1]
y_val = y[temp1:temp1+temp2]
y_test = y[temp1+temp2:]

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


# In[ ]:


look_back = 1
length_of_col = X_train.shape[1]

X_train = X_train.reshape((X_train.shape[0], look_back, length_of_col))
X_test = X_test.reshape((X_test.shape[0], look_back, length_of_col))
X_val = X_val.reshape((X_val.shape[0], look_back, length_of_col))

print(X_train.shape, X_val.shape, X_test.shape)


# In[ ]:


epoch = 100
model = Sequential()


model.add((LSTM(units=64, return_sequences=True, input_shape=(look_back, X_train.shape[2]), activation='relu')))
model.add(Dropout(0.2))
model.add((LSTM(units=32, return_sequences=True, activation='relu')))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 1, activation='sigmoid'))

# learning rate with decay
adam = optimizers.Adam(learning_rate=0.001)

model.compile(loss = 'mean_squared_error', optimizer = adam)

history = model.fit(X_train[:-1], y_train[1:], validation_data=(X_val[:-1], y_val[1:]), epochs=epoch, batch_size=50,verbose=1, shuffle = False)
model.save("LSTM_model.keras")
model.summary()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.figure(figsize=(8, 5))  # Set figure size for better readability

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='solid')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed')

# Titles and labels
plt.title('Model Loss Over Epochs - One Day', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# Add a legend and grid for clarity
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()


# In[ ]:


# Get the final loss values from history
final_train_loss_1day = history.history['loss'][-1]
final_val_loss_1day = history.history['val_loss'][-1]

print("1-Day Prediction Model:")
print(f"Final Training Loss: {final_train_loss_1day:.4f}")
print(f"Final Validation Loss: {final_val_loss_1day:.4f}")


# In[ ]:


#attempted 7 day forecast with recursive apporach, ran in to errors and hence moved to different model for 7 day prediction
# def recursive_forecast_for_all(model, X_data, steps=7):
#     """
#     Perform 7-step ahead forecast for each row in X_test or X_train with updates to future rows.
    
#     Parameters:
#         model: Trained LSTM model.
#         X_data: Input dataset (X_test or X_train) as a 3D NumPy array (samples, timesteps, features).
#         steps: Number of steps (days) to forecast.
    
#     Returns:
#         List of predictions for each row in X_data.
#     """
#     all_predictions = []  # Store predictions for all rows

#     for row in range(len(X_data)-steps+1):
#         predictions = []  # Store predictions for the current row
#         X_data_copy = X_data.copy()  # Copy the original dataset to preserve it

#         for i in range(steps):
#             # Ensure we do not exceed dataset bounds
#             if row + i >= len(X_data):
#                 break

#             # Extract current row input
#             temp_input = X_data_copy[row + i].copy()  # Shape (1, 9)

#             # Reshape for LSTM input: (samples=1, timesteps=1, features=9)
#             temp_input_reshaped = temp_input.reshape(1, 1, -1)

#             # Predict the next AQI value
#             pred = model.predict(temp_input_reshaped, verbose=0)
#             predictions.append(pred[0][0])  # Store the prediction

#             # Update lagged AQI values in the next row (if within bounds)
#             if row + i + 1 < len(X_data_copy):
#                 X_data_copy[row + i + 1][0][-2:] = [pred[0][0], temp_input[0][-2]]  # Update aqi_t-1 and aqi_t-2

#         all_predictions.append(predictions)  # Store the 7-step forecast for this row

#     return all_predictions



# seven_day_forecast = recursive_forecast_for_all(model, X_test, steps=7)
# # print("7-Day Forecasted AQI:", seven_day_forecast)


# In[ ]:


def recursive_forecast_for_all(model, X_data, steps=2):
    all_predictions = []  # Store predictions for all rows

    for row in range(len(X_data)):
        predictions = [np.nan] * steps
        X_data_copy = X_data.copy()  

        for i in range(steps):
            if row + i >= len(X_data):
                break

            temp_input = X_data_copy[row + i].copy()
            temp_input_reshaped = temp_input.reshape(1, 1, -1)
            pred = model.predict(temp_input_reshaped, verbose=0)
            predictions[i] = pred[0][0] 
            if row + i + 1 < len(X_data_copy):
                X_data_copy[row + i + 1][0][-2:] = [pred[0][0], temp_input[0][-2]]  

        all_predictions.append(predictions)

    return all_predictions

one_day_forecast = recursive_forecast_for_all(model, X_test, steps=7)


# In[ ]:


forecast_length = len(one_day_forecast)
test_df = df[-forecast_length:].copy()

columns = [f"pred_aqi_(t+{i})" if i > 0 else "pred_aqi" for i in range(1,len(one_day_forecast[0])+1)]


forecast_df = pd.DataFrame(one_day_forecast, columns=columns)
forecast_df.index = test_df.index
forecast_df['aqi_(t+1)'] = test_df['aqi'].shift(-1)

forecast_df = forecast_df * (maximum[0] - minimum[0]) + minimum[0]
forecast_df.tail(10)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

results = []

# Assuming 'aqi_(t+1)' is the observed AQI column
for col in forecast_df.columns:
    if col.startswith("pred_aqi_"):
        # Extract lag from the column name, for example 'pred_aqi_(t+1)', 'pred_aqi_(t+2)'
        lag = int(col.split("_(t+")[1].strip(")")) if "(t+" in col else 0

        # Adjust 'aqi_(t+1)' based on its lag relationship with predictions
        actual_column = 'aqi_(t+1)'  # Adjust this if the observed AQI column has a different name
        actual = forecast_df[actual_column][lag:].reset_index(drop=True)
        predicted = forecast_df[col][:-lag].reset_index(drop=True)

        # Ensure lengths match
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        # Drop NaN values
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]

        # Check if non-empty after removing NaNs
        if len(actual) == 0 or len(predicted) == 0:
            print(f"Skipping column {col} due to insufficient data.")
            continue

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)

        results.append({
            "Prediction Horizon": col,
            "RMSE": rmse,
            "R²": r2
        })

# Convert results to DataFrame
metrics_df = pd.DataFrame(results)
print(metrics_df)



# In[ ]:


import matplotlib.pyplot as plt

# plt.figure(figsize=(15, 8))  # Set figure size
# ax = plt.gca()  # Get the current axis

# # Replace 'aqi' with the actual column name for the observed AQI
# forecast_df['aqi_(t+1)'].plot(ax=ax, label="Actual AQI", linestyle='-', linewidth=2, color='blue')

# # Replace 'pred_aqi' with the actual column name for the predicted AQI
# forecast_df['pred_aqi_(t+1)'].plot(ax=ax, label="Predicted AQI", linestyle='-', linewidth=2, color='orange')

# ax.set_title("Actual AQI vs Predicted AQI", fontsize=16)
# ax.set_xlabel("Date", fontsize=14)
# ax.set_ylabel("AQI", fontsize=14)

# ax.legend(loc="upper left", fontsize=12)

# # Rotate x-axis ticks for better visibility
# plt.xticks(rotation=45)

# # Add grid
# ax.grid(True, linestyle='--', alpha=0.6)

# # Show the plot
# plt.tight_layout()
# plt.show()
#Fixed plot code 
plt.figure(figsize=(15, 8))
ax = plt.gca()

if "aqi_(t+1)" in forecast_df and "pred_aqi_(t+1)" in forecast_df:
    forecast_df["aqi_(t+1)"].plot(ax=ax, label="Actual AQI", linestyle='-', linewidth=2, color='blue')
    forecast_df["pred_aqi_(t+1)"].plot(ax=ax, label="Predicted AQI", linestyle='-', linewidth=2, color='orange')

    ax.set_title("Actual AQI vs Predicted AQI", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("AQI", fontsize=14)
    ax.legend(loc="upper left", fontsize=12)

    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Column names do not match! Check `forecast_df.columns` output.")


# In[ ]:


# Ensure the index is a DatetimeIndex
forecast_df.index = pd.to_datetime(forecast_df.index)

# Filter the data from 2024 onwards
forecast_df_2025 = forecast_df[forecast_df.index.year >= 2025]

# Plot the data
plt.figure(figsize=(15, 8))  # Set figure size
ax = plt.gca()  # Get the current axis

# Plot Actual AQI and Predicted AQI from 2024 onward
forecast_df_2025['aqi_(t+1)'].plot(ax=ax, label="Actual AQI", linestyle='-', linewidth=2, color='blue')
forecast_df_2025['pred_aqi_(t+1)'].plot(ax=ax, label="Predicted AQI", linestyle='-', linewidth=2, color='orange')

# Titles and labels
ax.set_title("Actual AQI vs Predicted AQI (2025 For One Day)", fontsize=16)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("AQI", fontsize=14)

ax.legend(loc="upper left", fontsize=12)

# Rotate x-axis ticks for better visibility
plt.xticks(rotation=45)

# Add grid
ax.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()





# In[ ]:




