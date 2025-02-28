# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# import traceback
#
# # Initialize the Flask application
# app = Flask(__name__)
#
# # Load the trained LSTM model
# model = load_model('LSTM_model.keras')
#
#
# @app.route('/')
# def home():
#     return "Welcome to the AQI Prediction API!"
#
#
# # Define a function to process the CSV file and prepare the data for prediction
# def preprocess_data(csv_file_path, required_features=9):
#     df = pd.read_csv(csv_file_path)
#
#     # Convert 'date' to datetime
#     df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce')
#
#     # Drop unwanted columns
#     cols_to_drop = ['latitude', 'longitude', 'city']
#     df.drop(columns=cols_to_drop, inplace=True)
#
#     # Relevant columns for the model
#     cols = ['aqi', 'temperature_2m', 'relative_humidity_2m',
#             'wind_speed_10m', 'pressure_msl', 'precipitation',
#             'cloud_cover', 'pm2.5']
#     df = df[cols]
#
#     # Drop rows with missing data
#     df.dropna(subset=cols, inplace=True)
#
#     # Prepare data for LSTM
#     time_steps = 3
#     X = []
#     for i in range(len(df) - time_steps):
#         X.append(df.iloc[i:i + time_steps].values)
#     X = np.array(X)
#
#     # Ensure it has the required features
#     current_features = X.shape[-1]
#     if current_features < required_features:
#         # Add padding with zeros
#         padding = np.zeros((X.shape[0], X.shape[1], required_features - current_features))
#         X = np.concatenate([X, padding], axis=-1)
#     elif current_features > required_features:
#         # Trim excess features
#         X = X[:, :, :required_features]
#
#     return X
#
# # Define a function to make predictions
# def make_prediction(input_data):
#     input_data = np.array(input_data)  # Ensure it's a numpy array
#     print(f"Input data shape before reshaping: {input_data.shape}")
#
#     # Check if input_data has the correct shape for LSTM model
#     if input_data.shape[0] == 1:  # If we have only 1 sample
#         input_data = input_data.reshape((1, 3, len(input_data[0])))  # Reshape for LSTM with correct features
#     else:
#         # If multiple sequences, no reshaping is needed
#         input_data = input_data.reshape(
#             (input_data.shape[0], 3, len(input_data[0])))  # Ensure (samples, time_steps, features)
#
#     print(f"Reshaped input data shape: {input_data.shape}")
#
#     prediction = model.predict(input_data)
#     return prediction[0][0]  # Assuming it's a single prediction value
#
#
# # Define the API endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the JSON request body
#         data = request.get_json(force=True)
#
#         # Check if the necessary data is in the request
#         if 'csv_file_path' not in data:
#             return jsonify({'error': 'Missing csv_file_path in the request'}), 400
#
#         csv_file_path = data['csv_file_path']
#
#         # Preprocess the data from the CSV file
#         input_data = preprocess_data(csv_file_path)
#
#         # Make the prediction using the model
#         prediction = make_prediction(input_data)
#
#         # Return the prediction as a JSON response
#         return jsonify({'prediction': prediction})
#
#     except Exception as e:
#         # Log the error and stack trace for debugging
#         error_message = str(e)
#         stack_trace = traceback.format_exc()
#         print(f"Error occurred: {error_message}")
#         print(f"Stack trace: {stack_trace}")
#
#         # Return the error message as a JSON response
#         return jsonify({'error': error_message, 'stack_trace': stack_trace}), 500
#
#
# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)

#Code to load data from csv file
# from flask import Flask, jsonify
# import pandas as pd
#
# # Initialize the Flask app
# app = Flask(__name__)
#
# # Path to the hard-coded CSV file
# CSV_FILE_PATH = 'Data/Final/San_Jose___Jackson.csv'
#
#
# def load_data():
#     """
#     Load and process the AQI data from the CSV file.
#     """
#     try:
#         # Read the CSV file
#         df = pd.read_csv(CSV_FILE_PATH, parse_dates=['date'], index_col='date')
#
#         # Check for duplicate index
#         if not df.index.is_unique:
#             # Separate numeric and non-numeric columns
#             numeric_cols = df.select_dtypes(include=['number']).columns
#             non_numeric_cols = df.select_dtypes(exclude=['number']).columns
#
#             # Aggregate numeric columns using mean
#             df_numeric = df[numeric_cols].groupby(df.index).mean()
#
#             # Aggregate non-numeric columns by taking the first non-null value
#             df_non_numeric = df[non_numeric_cols].groupby(df.index).first()
#
#             # Combine the numeric and non-numeric DataFrames
#             df = pd.concat([df_numeric, df_non_numeric], axis=1)
#
#         # Convert the index to string for JSON serialization
#         df.index = df.index.strftime('%Y-%m-%d')
#
#         # Convert the DataFrame to a dictionary for JSON response
#         data_dict = df.to_dict(orient='index')  # Use 'index' to keep the dates as keys
#         return data_dict
#
#     except FileNotFoundError:
#         raise Exception(f"CSV file not found at path: {CSV_FILE_PATH}")
#     except pd.errors.ParserError as e:
#         raise Exception(f"Error parsing CSV file: {str(e)}")
#     except Exception as e:
#         raise Exception(f"Unexpected error: {str(e)}")
#
#
#
#
#
# @app.route('/aqi-data', methods=['GET'])
# def get_aqi_data():
#     """
#     API endpoint to fetch the AQI data.
#     """
#     try:
#         # Load the data
#         data = load_data()
#
#         # Return the data as a JSON response
#         return jsonify(data)
#
#     except Exception as e:
#         # Handle errors and return a response
#         return jsonify({'error': str(e)}), 500
#
#
# @app.route('/')
# def home():
#     return "Welcome to the AQI Data API!"
#
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)
from flask import Flask, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Path to the CSV file
CSV_FILE_PATH = 'predictions_and_actuals.csv'

# Load the trained LSTM model
MODEL_PATH = 'LSTM_model.keras'
model = load_model(MODEL_PATH)


def preprocess_data_for_model(csv_file_path, time_steps=3, required_features=9):
    """
    Preprocess data from the CSV file for the LSTM model.
    Args:
        csv_file_path (str): Path to the CSV file.
        time_steps (int): Number of time steps for the LSTM input.
        required_features (int): Number of features expected by the model.
    Returns:
        np.array: Preprocessed data ready for prediction.
    """
    df = pd.read_csv(csv_file_path, parse_dates=['date'])

    # Select relevant columns for prediction
    features = ['aqi', 'temperature_2m', 'relative_humidity_2m',
                'wind_speed_10m', 'pressure_msl', 'precipitation',
                'cloud_cover', 'pm2.5']

    # Ensure the DataFrame has the correct number of features
    if len(features) < required_features:
        # Pad with zeros if there are fewer features than expected
        padding = np.zeros((len(df), required_features - len(features)))
        df = pd.concat([df[features], pd.DataFrame(padding)], axis=1)
    elif len(features) > required_features:
        # If there are more features, truncate to the required number
        df = df[features[:required_features]]

    df = df.dropna()  # Drop any rows with missing data

    # Prepare data for LSTM
    X = []
    for i in range(len(df) - time_steps):
        X.append(df.iloc[i:i + time_steps].values)

    return np.array(X)


@app.route('/predict', methods=['GET'])
def predict_aqi():
    """
    API endpoint to predict AQI for the next 30 days using the model.
    """
    try:
        # Preprocess the data
        input_data = preprocess_data_for_model(CSV_FILE_PATH)

        # Make predictions
        predictions = model.predict(input_data)
        predictions = predictions.flatten().tolist()

        # Construct a response with the prediction results
        response = {"predictions": predictions}
        return jsonify(response)

    except Exception as e:
        # Handle errors and return a response
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return "Welcome to the AQI Prediction API!"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

