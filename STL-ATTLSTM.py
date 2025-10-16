import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.seasonal import STL
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Layer
from keras import backend as K
from keras.callbacks import Callback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue

# Parameter configuration
CONFIG = {
    'look_back': 12,
    'epochs': 120,
    'batch_size': 32,
    'lstm_units': 64,
    'period': 12,
}

# Custom callback class: display training metrics after each epoch
class TrainingLogger(Callback):
    def __init__(self, dataX, dataY, scaler, n_features):
        super(TrainingLogger, self).__init__()
        self.dataX = dataX
        self.dataY = dataY
        self.scaler = scaler
        self.n_features = n_features

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        dataPredict = self.model.predict(self.dataX, batch_size=CONFIG['batch_size'], verbose=0)
        dataPredict_inv = self.inverse_transform(dataPredict)
        dataY_inv = self.inverse_transform(self.dataY.reshape(-1, 1))
        rmse = math.sqrt(mean_squared_error(dataY_inv, dataPredict_inv))
        mae = mean_absolute_error(dataY_inv, dataPredict_inv)
        mape = mean_absolute_percentage_error(dataY_inv, dataPredict_inv) * 100
        r2 = r2_score(dataY_inv, dataPredict_inv)
        logging.info(f"Epoch {epoch + 1:3d} | Loss: {train_loss:.4f} | RMSE: {rmse:.4f} | "
                     f"MAE: {mae:.4f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")

    def inverse_transform(self, predictions):
        extended = np.zeros((len(predictions), self.n_features))
        extended[:, 0] = predictions.ravel()
        return self.scaler.inverse_transform(extended)[:, 0]

# Data loading and preprocessing
def load_and_preprocess_data(file_path, target_column='value'):
    try:
        data = pd.read_excel(file_path)
        df = pd.DataFrame(data)
        if 'date' not in df.columns:
            raise ValueError("Missing 'date' column in data")
        df['date'] = pd.to_datetime(df['date'])
        if not df['date'].is_unique:
            logging.warning("Duplicate dates found, will remove duplicates")
            df = df.drop_duplicates(subset='date')
        if not df['date'].is_monotonic_increasing:
            logging.warning("Dates not in order, will sort")
            df = df.sort_values('date')
        df.set_index('date', inplace=True)
        df = df.loc['2001-01-01':'2015-12-31']
        expected_length = 180
        if len(df) != expected_length:
            logging.error(f"Dataset length {len(df)} does not equal expected {expected_length} months (2001-01 to 2015-12)")
            raise ValueError(f"Dataset length {len(df)} does not equal expected {expected_length} months")
        if df.isnull().sum().sum() > 0:
            df = df.fillna(method='ffill')
            logging.info("Missing values detected, forward filled")
        logging.info(f"Total dataset length: {len(df)}, date range: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        raise

# STL decomposition
def apply_stl_decomposition(data, column, period):
    stl = STL(data[column], period=period)
    result = stl.fit()
    data['trend'] = result.trend
    data['seasonal'] = result.seasonal
    data['residual'] = result.resid
    return data

# Data preparation
def prepare_lstm_data(normalized_data, look_back):
    dataX, dataY = [], []
    for i in range(len(normalized_data) - look_back):
        dataX.append(normalized_data[i:(i + look_back), :])
        dataY.append(normalized_data[i + look_back, 0])
    return np.array(dataX).reshape(-1, look_back, normalized_data.shape[1]), np.array(dataY)

# Custom attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        return K.sum(x * at, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Build model
def build_model(look_back, n_features, lstm_units):
    input_layer = Input(shape=(look_back, n_features))
    lstm_layer = LSTM(lstm_units, return_sequences=True)(input_layer)
    attention_output = AttentionLayer()(lstm_layer)
    output_layer = Dense(1, activation='linear')(attention_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Inverse normalization
def inverse_transform_predictions(predictions, n_features, scaler, column=0):
    extended = np.zeros((len(predictions), n_features))
    extended[:, column] = predictions.ravel()
    return scaler.inverse_transform(extended)[:, column]

# Plotting function
def plot_and_save(data_dict, title, xlabel, ylabel, filename, dpi=120, legend_loc='upper left', figsize=(12, 6)):
    plt.figure(figsize=figsize, dpi=dpi)
    for label, (x, y) in data_dict.items():
        plt.plot(x, y, label=label, linewidth=1.5)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10, loc=legend_loc, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main process
def main():
    # Data loading
    df = load_and_preprocess_data("YY1-2.xlsx", target_column='value')

    # STL decomposition
    df = apply_stl_decomposition(df, 'value', CONFIG['period'])

    # Output STL decomposition results
    logging.info("\nSTL Decomposition Statistics:")
    logging.info(f"Trend - Mean: {df['trend'].mean():.2f}, Std: {df['trend'].std():.2f}")
    logging.info(f"Seasonal - Mean: {df['seasonal'].mean():.2f}, Std: {df['seasonal'].std():.2f}")
    logging.info(f"Residual - Mean: {df['residual'].mean():.2f}, Std: {df['residual'].std():.2f}")

    # Plot STL decomposition
    stl_data = {
        'Original Data': (df.index, df['value']),
        'Trend': (df.index, df['trend']),
        'Seasonal': (df.index, df['seasonal']),
        'Residual': (df.index, df['residual'])
    }
    plt.figure(figsize=(10, 8), dpi=300)
    for i, (label, (x, y)) in enumerate(stl_data.items(), 1):
        plt.subplot(4, 1, i)
        plt.plot(x, y, label=label, linewidth=1.5)
        plt.title(label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        if i < 4:
            plt.xticks([])
    plt.tight_layout()
    plt.savefig('stl_decomposition.png')
    plt.close()

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(df)

    # Prepare LSTM data
    dataX, dataY = prepare_lstm_data(data_normalized, CONFIG['look_back'])

    # Validate data length
    expected_dataY_len = len(df) - CONFIG['look_back']  # 180 - 12 = 168
    logging.info(f"dataX shape: {dataX.shape}, dataY length: {len(dataY)}, expected: {expected_dataY_len}")
    if len(dataY) != expected_dataY_len:
        raise ValueError(f"Prediction data length error: dataY={len(dataY)}")

    # Build and train model
    model = build_model(CONFIG['look_back'], dataX.shape[2], CONFIG['lstm_units'])
    training_logger = TrainingLogger(dataX, dataY, scaler, df.shape[1])
    history = model.fit(dataX, dataY, epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                        verbose=0, callbacks=[training_logger])

    # Prediction
    dataPredict = model.predict(dataX, batch_size=CONFIG['batch_size'])

    # Inverse normalization
    dataPredict = inverse_transform_predictions(dataPredict, df.shape[1], scaler)
    dataY = inverse_transform_predictions(dataY, df.shape[1], scaler)

    # Calculate error
    dataError = dataY - dataPredict

    # STL decomposition on predictions
    dataPredict_series = pd.Series(dataPredict, index=df.index[CONFIG['look_back']:])
    dataPredict_df = pd.DataFrame({'value': dataPredict_series})
    dataPredict_df = apply_stl_decomposition(dataPredict_df, 'value', CONFIG['period'])

    # Validate prediction index
    logging.info(f"dataPredict_series index: {dataPredict_series.index.tolist()[:5]} ... {dataPredict_series.index.tolist()[-5:]}")

    # Visualization
    plot_and_save({'Training Loss': (range(len(history.history['loss'])), history.history['loss'])},
                  'Training Loss Curve', 'Epoch', 'Loss (MSE)', 'loss_curve.png', dpi=300, figsize=(10, 6))
    plot_and_save({
        'Actual': (df.index[CONFIG['look_back']:], dataY),
        'Predicted': (df.index[CONFIG['look_back']:], dataPredict)
    }, 'STL-Attention-LSTM Prediction Results', 'Time', 'Water Level (m)', 'prediction_result.png', figsize=(12, 6))
    plot_and_save({
        'Prediction Error': (df.index[CONFIG['look_back']:], dataError)
    }, 'Prediction Error', 'Time', 'Error (m)', 'error_plot.png', figsize=(12, 6))
    plt.figure(figsize=(10, 6), dpi=120)
    plt.hist(dataError, bins=30, edgecolor='black', alpha=0.7)
    plt.title("Prediction Error Distribution", fontsize=14)
    plt.xlabel("Error (m)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()

    # Evaluation metrics
    def calculate_metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R2': r2_score(y_true, y_pred)
        }

    metrics = calculate_metrics(dataY, dataPredict)
    logging.info("\nModel Evaluation Metrics:")
    logging.info(f"Scores: MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%, R2: {metrics['R2']:.2f}")

    # Save results to Excel
    total_length = len(dataY)  # 168
    expected_total_length = len(df) - CONFIG['look_back']
    logging.info(f"total_length: {total_length}, expected: {expected_total_length}")
    if total_length != expected_total_length:
        logging.error(f"total_length {total_length} does not equal expected {expected_total_length}")
        raise ValueError(f"total_length error: {total_length}")
    index = df.index[CONFIG['look_back']:]  # From 2002-01-01 to 2015-12-01
    if len(index) != total_length:
        logging.error(f"Index length {len(index)} does not equal total_length {total_length}")
        raise ValueError(f"Index length error")

    # Validate STL decomposition result length
    logging.info(f"dataPredict_df['trend'] length: {len(dataPredict_df['trend'])}, expected: {len(dataY)}")
    if len(dataPredict_df['trend']) != len(dataY):
        raise ValueError("STL decomposition result length does not match prediction length")

    # Validate all input array lengths
    arrays = {
        'Real': dataY,
        'Predict': dataPredict,
        'Error': dataError,
        'Predict_Trend': dataPredict_df['trend'].values,
        'Predict_Seasonal': dataPredict_df['seasonal'].values,
        'Predict_Residual': dataPredict_df['residual'].values,
        'True_Trend': df['trend'][CONFIG['look_back']:].values,
        'True_Seasonal': df['seasonal'][CONFIG['look_back']:].values,
        'True_Residual': df['residual'][CONFIG['look_back']:].values
    }
    for name, arr in arrays.items():
        logging.info(f"{name} length before padding: {len(arr)}")

    # Create results_df and populate data
    results_df = pd.DataFrame(index=index)
    for column, values in arrays.items():
        results_df[column] = values  # Direct assignment, no padding needed as lengths match

    # Validate results
    logging.info(f"results_df index: {results_df.index.tolist()[:5]} ... {results_df.index.tolist()[-5:]}")
    logging.info(f"Predict non-null count: {results_df['Predict'].notna().sum()}, expected: {len(dataPredict)}")
    results_df.to_excel('prediction_results.xlsx')

    # Data prediction analysis
    logging.info("\nData Prediction Analysis:")
    logging.info(f"Predicted value mean: {np.mean(dataPredict):.2f} m")
    logging.info(f"Actual value mean: {np.mean(dataY):.2f} m")
    logging.info(f"Error mean: {np.mean(dataError):.2f} m")
    logging.info(f"Error std: {np.std(dataError):.2f} m")
    logging.info(f"Error min: {np.min(dataError):.2f} m")
    logging.info(f"Error max: {np.max(dataError):.2f} m")

if __name__ == "__main__":
    main()
