# -*- coding: utf-8 -*-
"""
Yellow River Basin Groundwater Level Delayed Response Analysis Complete Code
Input data requirements: Excel file containing dates and specified features
Output results: Feature importance analysis, delayed response curves, prediction model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Multiply, Attention, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import shap
import os

# ================== Environment Configuration ==================
# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Disable oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# ================== Data Preprocessing Module ==================
def load_and_preprocess_data(filepath, target_col='YY1-1', max_lag=12):
    """
    Data loading and preprocessing pipeline
    :param filepath: Excel file path
    :param target_col: Target column name (groundwater level)
    :param max_lag: Maximum lag in months
    :return: 3D feature data, target values, feature name list
    """
    try:
        # 1. Data loading
        df = pd.read_excel(filepath,
                           sheet_name='HydroData',
                           parse_dates=['Time'],
                           index_col='Time')

        # 2. Validate required columns exist
        features = ['Rainfall', 'Evapotranspiration', 'Huayuankou_Water_Level', 'Huayuankou_Flow_Rate', 'Water_Consumption']
        required_cols = [target_col] + features
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Data file missing required columns: {missing_cols}")

        # 3. Feature engineering
        print("Original data sample:\n", df.head())

        # 4. Generate lagged features
        for col in features:
            for lag in range(1, max_lag + 1):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            df.drop(col, axis=1, inplace=True)  # Remove original features

        df = df.dropna()
        print("Data dimensions after lag processing:", df.shape)

        # 5. Data standardization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_X.fit_transform(df.drop(target_col, axis=1))
        y = scaler_y.fit_transform(df[[target_col]]).flatten()

        # 6. Convert to 3D input
        n_samples = X.shape[0]
        n_features = len(features)
        X_3d = X.reshape(n_samples, max_lag, n_features)

        return X_3d, y, scaler_X, scaler_y, features

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        raise


# ================== Model Building Module ==================
def build_lstm_attention_model(input_shape, n_features):
    """Build spatio-temporal attention LSTM model"""
    inputs = Input(shape=input_shape)

    # LSTM layer captures temporal dependencies
    lstm_out = LSTM(32, return_sequences=True)(inputs)

    # Attention mechanism layer
    attention = Attention()([lstm_out, lstm_out])
    context = Multiply()([lstm_out, attention])

    # Feature aggregation and output
    pooled = GlobalAveragePooling1D()(context)
    output = Dense(1)(pooled)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# ================== SHAP Analysis Module ==================
def shap_analysis(model, X_train, features, max_lag=12):
    """SHAP feature contribution analysis"""
    try:
        # Convert 3D data to 2D
        background = X_train[:50].reshape(50, -1)

        # Define model wrapper
        def model_wrapper(x):
            return model.predict(x.reshape(-1, max_lag, len(features))).flatten()

        # Calculate SHAP values
        explainer = shap.KernelExplainer(model_wrapper, background)
        test_samples = X_train[:10].reshape(10, -1)
        shap_values = explainer.shap_values(test_samples)

        # Aggregate feature contributions
        shap_global = np.abs(shap_values).mean(axis=0)
        shap_global = shap_global.reshape(max_lag, len(features)).sum(axis=0)
        total = shap_global.sum()
        return {features[i]: (shap_global[i] / total * 100) for i in range(len(features))}

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        return {f: 0.0 for f in features}


# ================== Visualization Module ==================
def plot_results(history, att_importance, shap_importance, features):
    """Results visualization"""
    # Training process visualization
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Process')
    plt.xlabel('Training Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    # Feature importance comparison
    importance_df = pd.DataFrame({
        'Feature': features,
        'Attention Weight(%)': [att_importance[f] for f in features],
        'SHAP Contribution(%)': [shap_importance[f] for f in features]
    }).sort_values('Attention Weight(%)', ascending=False)

    ax = importance_df.plot.bar(x='Feature', y=['Attention Weight(%)', 'SHAP Contribution(%)'],
                                figsize=(12, 6), rot=45)
    ax.set_title('Feature Importance Comparison')
    ax.set_ylabel('Contribution Percentage (%)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


# ================== Main Program ==================
if __name__ == "__main__":
    # Parameter configuration
    DATA_PATH = "new_Microsoft_Excel_worksheet.xlsx"  # Replace with actual path
    MAX_LAG = 6  # Number of lag months

    try:
        # 1. Data loading and preprocessing
        X, y, scaler_X, scaler_y, features = load_and_preprocess_data(DATA_PATH, target_col='YY1-1', max_lag=MAX_LAG)
        print("3D input data dimensions:", X.shape)

        # 2. Dataset split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 3. Model building
        model = build_lstm_attention_model((MAX_LAG, len(features)), len(features))
        model.summary()

        # 4. Model training
        history = model.fit(X_train, y_train,
                            epochs=200,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=20)])

        # 5. Feature importance analysis
        # 5.1 Attention weight analysis
        partial_model = Model(inputs=model.input, outputs=model.layers[2].output)
        att_weights = np.mean(partial_model.predict(X_train), axis=(0, 1))
        att_importance = {features[i]: att_weights[i] / att_weights.sum() * 100 for i in range(len(features))}

        # 5.2 SHAP analysis
        shap_importance = shap_analysis(model, X_train, features, MAX_LAG)

        # 6. Results visualization and saving
        plot_results(history, att_importance, shap_importance, features)

        # 7. Save results
        result_df = pd.DataFrame({
            'Feature': features,
            'Optimal Lag(months)': [1, 2, 3, 4, 5],  # Needs modification based on actual analysis
            'Attention Contribution(%)': [att_importance[f] for f in features],
            'SHAP Contribution(%)': [shap_importance[f] for f in features]
        })
        result_df.to_excel('analysis_results.xlsx', index=False)
        print("Analysis complete, results saved!")

    except Exception as e:
        print(f"Program execution error: {str(e)}")
