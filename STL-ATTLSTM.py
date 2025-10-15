import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.seasonal import STL
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Layer
from keras import backend as K
from keras.callbacks import Callback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 参数配置
CONFIG = {
    'look_back': 12,
    'epochs': 120,
    'batch_size': 32,
    'lstm_units': 64,
    'period': 12,
}

# 自定义回调类：每个 epoch 结束后显示训练指标
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

# 数据加载与预处理
def load_and_preprocess_data(file_path, target_column='value'):
    try:
        data = pd.read_excel(file_path)
        df = pd.DataFrame(data)
        if 'date' not in df.columns:
            raise ValueError("数据中缺少'date'列")
        df['date'] = pd.to_datetime(df['date'])
        if not df['date'].is_unique:
            logging.warning("发现重复日期，将删除重复项")
            df = df.drop_duplicates(subset='date')
        if not df['date'].is_monotonic_increasing:
            logging.warning("日期未按顺序排列，将排序")
            df = df.sort_values('date')
        df.set_index('date', inplace=True)
        df = df.loc['2001-01-01':'2015-12-31']
        expected_length = 180
        if len(df) != expected_length:
            logging.error(f"数据集长度 {len(df)} 不等于预期 {expected_length} 个月（2001-01到2015-12）")
            raise ValueError(f"数据集长度 {len(df)} 不等于预期 {expected_length} 个月")
        if df.isnull().sum().sum() > 0:
            df = df.fillna(method='ffill')
            logging.info("检测到缺失值，已使用前向填充处理")
        logging.info(f"数据集总长度: {len(df)}, 日期范围: {df.index[0]} 到 {df.index[-1]}")
        return df
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise

# STL分解
def apply_stl_decomposition(data, column, period):
    stl = STL(data[column], period=period)
    result = stl.fit()
    data['trend'] = result.trend
    data['seasonal'] = result.seasonal
    data['residual'] = result.resid
    return data

# 数据准备
def prepare_lstm_data(normalized_data, look_back):
    dataX, dataY = [], []
    for i in range(len(normalized_data) - look_back):
        dataX.append(normalized_data[i:(i + look_back), :])
        dataY.append(normalized_data[i + look_back, 0])
    return np.array(dataX).reshape(-1, look_back, normalized_data.shape[1]), np.array(dataY)

# 自定义注意力层
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

# 构建模型
def build_model(look_back, n_features, lstm_units):
    input_layer = Input(shape=(look_back, n_features))
    lstm_layer = LSTM(lstm_units, return_sequences=True)(input_layer)
    attention_output = AttentionLayer()(lstm_layer)
    output_layer = Dense(1, activation='linear')(attention_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# 反归一化
def inverse_transform_predictions(predictions, n_features, scaler, column=0):
    extended = np.zeros((len(predictions), n_features))
    extended[:, column] = predictions.ravel()
    return scaler.inverse_transform(extended)[:, column]

# 绘图函数
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

# 主流程
def main():
    # 数据加载
    df = load_and_preprocess_data("YY1-2.xlsx", target_column='value')

    # STL分解
    df = apply_stl_decomposition(df, 'value', CONFIG['period'])

    # 输出 STL 分解结果
    logging.info("\n数据 STL 分解结果统计：")
    logging.info(f"趋势 (Trend) - 均值: {df['trend'].mean():.2f}, 标准差: {df['trend'].std():.2f}")
    logging.info(f"季节性 (Seasonal) - 均值: {df['seasonal'].mean():.2f}, 标准差: {df['seasonal'].std():.2f}")
    logging.info(f"残差 (Residual) - 均值: {df['residual'].mean():.2f}, 标准差: {df['residual'].std():.2f}")

    # 绘制 STL 分解图
    stl_data = {
        '原始数据': (df.index, df['value']),
        '趋势': (df.index, df['trend']),
        '季节性': (df.index, df['seasonal']),
        '残差': (df.index, df['residual'])
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

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(df)

    # 准备LSTM数据
    dataX, dataY = prepare_lstm_data(data_normalized, CONFIG['look_back'])

    # 验证数据长度
    expected_dataY_len = len(df) - CONFIG['look_back']  # 180 - 12 = 168
    logging.info(f"dataX shape: {dataX.shape}, dataY length: {len(dataY)}, 预期: {expected_dataY_len}")
    if len(dataY) != expected_dataY_len:
        raise ValueError(f"预测数据长度错误: dataY={len(dataY)}")

    # 构建并训练模型
    model = build_model(CONFIG['look_back'], dataX.shape[2], CONFIG['lstm_units'])
    training_logger = TrainingLogger(dataX, dataY, scaler, df.shape[1])
    history = model.fit(dataX, dataY, epochs=CONFIG['epochs'], batch_size=CONFIG['batch_size'],
                        verbose=0, callbacks=[training_logger])

    # 预测
    dataPredict = model.predict(dataX, batch_size=CONFIG['batch_size'])

    # 反归一化
    dataPredict = inverse_transform_predictions(dataPredict, df.shape[1], scaler)
    dataY = inverse_transform_predictions(dataY, df.shape[1], scaler)

    # 计算误差
    dataError = dataY - dataPredict

    # 对预测值进行STL分解
    dataPredict_series = pd.Series(dataPredict, index=df.index[CONFIG['look_back']:])
    dataPredict_df = pd.DataFrame({'value': dataPredict_series})
    dataPredict_df = apply_stl_decomposition(dataPredict_df, 'value', CONFIG['period'])

    # 验证预测索引
    logging.info(f"dataPredict_series 索引: {dataPredict_series.index.tolist()[:5]} ... {dataPredict_series.index.tolist()[-5:]}")

    # 可视化
    plot_and_save({'训练损失': (range(len(history.history['loss'])), history.history['loss'])},
                  '训练损失曲线', '轮次', '损失 (MSE)', 'loss_curve.png', dpi=300, figsize=(10, 6))
    plot_and_save({
        '真实值': (df.index[CONFIG['look_back']:], dataY),
        '预测值': (df.index[CONFIG['look_back']:], dataPredict)
    }, 'STL-注意力-LSTM预测结果', '时间', '水位 (m)', 'prediction_result.png', figsize=(12, 6))
    plot_and_save({
        '预测误差': (df.index[CONFIG['look_back']:], dataError)
    }, '预测误差', '时间', '误差 (m)', 'error_plot.png', figsize=(12, 6))
    plt.figure(figsize=(10, 6), dpi=120)
    plt.hist(dataError, bins=30, edgecolor='black', alpha=0.7)
    plt.title("预测误差分布", fontsize=14)
    plt.xlabel("误差 (m)", fontsize=12)
    plt.ylabel("频率", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()

    # 评估指标
    def calculate_metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R2': r2_score(y_true, y_pred)
        }

    metrics = calculate_metrics(dataY, dataPredict)
    logging.info("\n模型评估指标：")
    logging.info(f"得分: MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%, R2: {metrics['R2']:.2f}")

    # 保存结果到Excel
    total_length = len(dataY)  # 168
    expected_total_length = len(df) - CONFIG['look_back']
    logging.info(f"total_length: {total_length}, 预期: {expected_total_length}")
    if total_length != expected_total_length:
        logging.error(f"total_length {total_length} 不等于预期 {expected_total_length}")
        raise ValueError(f"total_length 错误: {total_length}")
    index = df.index[CONFIG['look_back']:]  # 从 2002-01-01 到 2015-12-01
    if len(index) != total_length:
        logging.error(f"索引长度 {len(index)} 不等于 total_length {total_length}")
        raise ValueError(f"索引长度错误")

    # 验证 STL 分解结果长度
    logging.info(f"dataPredict_df['trend'] length: {len(dataPredict_df['trend'])}, 预期: {len(dataY)}")
    if len(dataPredict_df['trend']) != len(dataY):
        raise ValueError("STL分解结果长度与预测长度不匹配")

    # 验证所有输入数组的长度
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

    # 创建 results_df 并填充数据
    results_df = pd.DataFrame(index=index)
    for column, values in arrays.items():
        results_df[column] = values  # 直接赋值，无需填充，因为长度已匹配

    # 验证填充结果
    logging.info(f"results_df 索引: {results_df.index.tolist()[:5]} ... {results_df.index.tolist()[-5:]}")
    logging.info(f"Predict 非空值数量: {results_df['Predict'].notna().sum()}, 预期: {len(dataPredict)}")
    results_df.to_excel('prediction_results.xlsx')

    # 数据预测分析
    logging.info("\n数据预测分析：")
    logging.info(f"预测值均值: {np.mean(dataPredict):.2f} m")
    logging.info(f"真实值均值: {np.mean(dataY):.2f} m")
    logging.info(f"误差均值: {np.mean(dataError):.2f} m")
    logging.info(f"误差标准差: {np.std(dataError):.2f} m")
    logging.info(f"误差最小值: {np.min(dataError):.2f} m")
    logging.info(f"误差最大值: {np.max(dataError):.2f} m")

if __name__ == "__main__":
    main()