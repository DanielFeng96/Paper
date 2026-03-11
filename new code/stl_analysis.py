import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据读取与预处理 ====================
file_path = '月均值.xlsx'
data = pd.read_excel(file_path)

# 设置时间索引
data['Time'] = pd.to_datetime(data['Time'])
time_col = data.columns[0]
well_columns = data.columns[1:]

print("="*80)
print("STL季节性趋势分解分析")
print("="*80)
print(f"\n数据范围：{data['Time'].min().date()} 到 {data['Time'].max().date()}")
print(f"总共{len(data)}个月度观测值")
print(f"监测井数量：{len(well_columns)}")
print(f"监测井列表：{', '.join(well_columns)}\n")

# ==================== STL分解与统计计算 ====================
results_list = []
components_original  = {'Time': data['Time']}
components_trend     = {'Time': data['Time']}
components_seasonal  = {'Time': data['Time']}
components_remainder = {'Time': data['Time']}

for well in well_columns:
    # 提取该监测井的时间序列
    ts = data[well].values

    try:
        # 使用STL进行分解
        # period: 季节周期为12（月数据的年度周期）
        # seasonal: 季节性平滑参数（必须是奇数）
        # trend: 趋势窗口大小（必须是奇数且>period）
        trend_window = 13 * 12 + 1  # 保证是奇数且>12
        stl = STL(ts, seasonal=13, trend=trend_window, period=12)
        result = stl.fit()

        # 提取各分量
        trend = result.trend
        seasonal = result.seasonal
        remainder = result.resid
        original = ts

        # ========== 计算方差 ==========
        var_original = np.var(original, ddof=1)
        var_trend = np.var(trend, ddof=1)
        var_seasonal = np.var(seasonal, ddof=1)
        var_remainder = np.var(remainder, ddof=1)

        # ========== 计算方差比 ==========
        ratio_trend = var_trend / var_original * 100 if var_original != 0 else 0
        ratio_seasonal = var_seasonal / var_original * 100 if var_original != 0 else 0
        ratio_remainder = var_remainder / var_original * 100 if var_original != 0 else 0

        # ========== 基本统计量 ==========
        mean_original = np.mean(original)
        min_original = np.min(original)
        max_original = np.max(original)
        std_original = np.std(original, ddof=1)

        # 保存各分量时序
        components_original[well]  = original
        components_trend[well]     = trend
        components_seasonal[well]  = seasonal
        components_remainder[well] = remainder

        # 保存结果
        results_list.append({
            '监测井': well,
            '平均值(m)': mean_original,
            '最小值(m)': min_original,
            '最大值(m)': max_original,
            '标准差': std_original,
            '原始序列方差': var_original,
            '趋势分量方差': var_trend,
            '季节性分量方差': var_seasonal,
            '残差分量方差': var_remainder,
            '趋势方差比(%)': ratio_trend,
            '季节性方差比(%)': ratio_seasonal,
            '残差方差比(%)': ratio_remainder,
        })

        print(f"✓ {well} 分解成功")

    except Exception as e:
        print(f"✗ {well} 分解失败: {str(e)}")
        continue

# ==================== 生成结果表格 ====================
results_df = pd.DataFrame(results_list)

# 表1：基本统计量
table_basic = results_df[['监测井', '平均值(m)', '最小值(m)', '最大值(m)', '标准差']].copy()

# 表2：方差统计
table_var = results_df[['监测井', '原始序列方差', '趋势分量方差', '季节性分量方差', '残差分量方差']].copy()
table_var.columns = ['监测井', 'Original', 'Trend', 'Season', 'Remainder']

# 表3：方差比统计
table_ratio = results_df[['监测井', '趋势方差比(%)', '季节性方差比(%)', '残差方差比(%)']].copy()
table_ratio.columns = ['监测井', 'Trend (%)', 'Season (%)', 'Remainder (%)']

# ==================== 控制台输出 ====================
print("\n" + "="*80)
print("表1：地下水位基本统计量")
print("="*80)
print(table_basic.to_string(index=False))

print("\n" + "="*80)
print("表2：各分量的方差（Variance）")
print("="*80)
print(table_var.round(6).to_string(index=False))

print("\n" + "="*80)
print("表3：各分量方差比 / var. ratio（相对于原始序列方差）")
print("="*80)
print(table_ratio.round(4).to_string(index=False))

# ==================== 构建分量时序表 ====================
time_str = data['Time'].dt.strftime('%Y-%m')

df_original  = pd.DataFrame(components_original)
df_trend     = pd.DataFrame(components_trend)
df_seasonal  = pd.DataFrame(components_seasonal)
df_remainder = pd.DataFrame(components_remainder)

df_original['Time']  = time_str
df_trend['Time']     = time_str
df_seasonal['Time']  = time_str
df_remainder['Time'] = time_str

# 每口井四分量合并对照表（宽表：行=时间，列=监测井_分量）
combined_cols = [time_str]
for well in well_columns:
    combined_cols.append(df_original[well].rename(f'{well}_Original'))
    combined_cols.append(df_trend[well].rename(f'{well}_Trend'))
    combined_cols.append(df_seasonal[well].rename(f'{well}_Season'))
    combined_cols.append(df_remainder[well].rename(f'{well}_Remainder'))
df_combined = pd.concat([time_str] + [df_original[w].rename(f'{w}_Original') for w in well_columns]
                        + [df_trend[w].rename(f'{w}_Trend') for w in well_columns]
                        + [df_seasonal[w].rename(f'{w}_Season') for w in well_columns]
                        + [df_remainder[w].rename(f'{w}_Remainder') for w in well_columns], axis=1)
df_combined.columns = ['Time'] + [f'{w}_Original' for w in well_columns] \
                                + [f'{w}_Trend'    for w in well_columns] \
                                + [f'{w}_Season'   for w in well_columns] \
                                + [f'{w}_Remainder' for w in well_columns]

# ==================== 输出到单一Excel ====================
with pd.ExcelWriter('STL分解分析结果.xlsx', engine='openpyxl') as writer:
    table_basic.to_excel(writer, sheet_name='基本统计量', index=False)
    table_var.to_excel(writer, sheet_name='方差统计 Variance', index=False)
    table_ratio.to_excel(writer, sheet_name='方差比 Var.Ratio', index=False)
    results_df.to_excel(writer, sheet_name='完整统计', index=False)
    df_original.to_excel(writer, sheet_name='原始序列 Original', index=False)
    df_trend.to_excel(writer, sheet_name='趋势分量 Trend', index=False)
    df_seasonal.to_excel(writer, sheet_name='季节性分量 Season', index=False)
    df_remainder.to_excel(writer, sheet_name='残差分量 Remainder', index=False)
    df_combined.to_excel(writer, sheet_name='四分量合并对照', index=False)

print("\n✓ 结果已保存到 'STL分解分析结果.xlsx'")
print("  工作表：基本统计量 / 方差统计 / 方差比 / 完整统计")
print("         原始序列 / 趋势分量 / 季节性分量 / 残差分量 / 四分量合并对照")
print("="*80)
