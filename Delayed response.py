# -*- coding: utf-8 -*-
"""
黄河流域地下水位滞后响应分析完整代码
输入数据要求：包含日期和指定特征的Excel文件
输出结果：特征重要性分析、滞后响应曲线、预测模型
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

# ================== 环境配置 ==================
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 禁用oneDNN优化警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# ================== 数据预处理模块 ==================
def load_and_preprocess_data(filepath, target_col='YY1-1', max_lag=12):
    """
    数据加载和预处理流程
    :param filepath: Excel文件路径
    :param target_col: 目标列名称(地下水位)
    :param max_lag: 最大滞后月份
    :return: 3D特征数据, 目标值, 特征名称列表
    """
    try:
        # 1. 数据加载
        df = pd.read_excel(filepath,
                           sheet_name='HydroData',
                           parse_dates=['Time'],
                           index_col='Time')

        # 2. 验证必要列存在
        features = ['Rainfall', '蒸散发', 'Huayuankou水位', 'Huayuankou流量', '耗水量']
        required_cols = [target_col] + features
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少必要列：{missing_cols}")

        # 3. 特征工程
        print("原始数据样例：\n", df.head())

        # 4. 生成滞后特征
        for col in features:
            for lag in range(1, max_lag + 1):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            df.drop(col, axis=1, inplace=True)  # 删除原始特征

        df = df.dropna()
        print("滞后处理后数据维度：", df.shape)

        # 5. 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = scaler_X.fit_transform(df.drop(target_col, axis=1))
        y = scaler_y.fit_transform(df[[target_col]]).flatten()

        # 6. 转换为3D输入
        n_samples = X.shape[0]
        n_features = len(features)
        X_3d = X.reshape(n_samples, max_lag, n_features)

        return X_3d, y, scaler_X, scaler_y, features

    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        raise


# ================== 模型构建模块 ==================
def build_lstm_attention_model(input_shape, n_features):
    """构建时空注意力LSTM模型"""
    inputs = Input(shape=input_shape)

    # LSTM层捕捉时间依赖性
    lstm_out = LSTM(32, return_sequences=True)(inputs)

    # 注意力机制层
    attention = Attention()([lstm_out, lstm_out])
    context = Multiply()([lstm_out, attention])

    # 特征聚合与输出
    pooled = GlobalAveragePooling1D()(context)
    output = Dense(1)(pooled)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


# ================== SHAP分析模块 ==================
def shap_analysis(model, X_train, features, max_lag=12):
    """SHAP特征贡献分析"""
    try:
        # 将3D数据转换为2D
        background = X_train[:50].reshape(50, -1)

        # 定义模型包装器
        def model_wrapper(x):
            return model.predict(x.reshape(-1, max_lag, len(features))).flatten()

        # 计算SHAP值
        explainer = shap.KernelExplainer(model_wrapper, background)
        test_samples = X_train[:10].reshape(10, -1)
        shap_values = explainer.shap_values(test_samples)

        # 聚合特征贡献
        shap_global = np.abs(shap_values).mean(axis=0)
        shap_global = shap_global.reshape(max_lag, len(features)).sum(axis=0)
        total = shap_global.sum()
        return {features[i]: (shap_global[i] / total * 100) for i in range(len(features))}

    except Exception as e:
        print(f"SHAP分析失败：{str(e)}")
        return {f: 0.0 for f in features}


# ================== 可视化模块 ==================
def plot_results(history, att_importance, shap_importance, features):
    """结果可视化"""
    # 训练过程可视化
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练过程')
    plt.xlabel('训练轮次')
    plt.ylabel('均方误差')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    # 特征重要性对比
    importance_df = pd.DataFrame({
        '特征': features,
        '注意力权重(%)': [att_importance[f] for f in features],
        'SHAP贡献(%)': [shap_importance[f] for f in features]
    }).sort_values('注意力权重(%)', ascending=False)

    ax = importance_df.plot.bar(x='特征', y=['注意力权重(%)', 'SHAP贡献(%)'],
                                figsize=(12, 6), rot=45)
    ax.set_title('特征重要性对比')
    ax.set_ylabel('贡献度百分比(%)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()


# ================== 主程序 ==================
if __name__ == "__main__":
    # 参数配置
    DATA_PATH = "新建 Microsoft Excel 工作表.xlsx"  # 替换为实际路径
    MAX_LAG = 6  # 滞后月份数

    try:
        # 1. 数据加载与预处理
        X, y, scaler_X, scaler_y, features = load_and_preprocess_data(DATA_PATH, target_col='YY1-1', max_lag=MAX_LAG)
        print("3D输入数据维度：", X.shape)

        # 2. 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 3. 模型构建
        model = build_lstm_attention_model((MAX_LAG, len(features)), len(features))
        model.summary()

        # 4. 模型训练
        history = model.fit(X_train, y_train,
                            epochs=200,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=20)])

        # 5. 特征重要性分析
        # 5.1 注意力权重分析
        partial_model = Model(inputs=model.input, outputs=model.layers[2].output)
        att_weights = np.mean(partial_model.predict(X_train), axis=(0, 1))
        att_importance = {features[i]: att_weights[i] / att_weights.sum() * 100 for i in range(len(features))}

        # 5.2 SHAP分析
        shap_importance = shap_analysis(model, X_train, features, MAX_LAG)

        # 6. 结果可视化与保存
        plot_results(history, att_importance, shap_importance, features)

        # 7. 保存结果
        result_df = pd.DataFrame({
            '特征': features,
            '最佳滞后(月)': [1, 2, 3, 4, 5],  # 需根据实际分析修改
            '注意力贡献(%)': [att_importance[f] for f in features],
            'SHAP贡献(%)': [shap_importance[f] for f in features]
        })
        result_df.to_excel('分析结果.xlsx', index=False)
        print("分析完成，结果已保存！")

    except Exception as e:
        print(f"程序运行出错：{str(e)}")