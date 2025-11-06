"""
Step 1.4 · 多因子模型框架设计

任务：
1. 因子IC值计算（与未来收益的相关性）
2. 因子选择与降维
3. 因子标准化方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
stocks = ["00005HK", "00939HK", "00700HK", "00941HK"]
output_dir = os.path.dirname(os.path.abspath(__file__)) or "."
os.chdir(output_dir)

# ==================== 数据加载 ====================

def load_factor_data():
    """加载所有股票的技术因子数据"""
    factor_data = {}
    for stock in stocks:
        file_path = f"{stock}_tech_factors.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['t'] = pd.to_datetime(df['t'])
            df = df.sort_values('t').reset_index(drop=True)
            factor_data[stock] = df
            print(f"✅ 加载 {stock}: {len(df)} 条记录")
        else:
            print(f"❌ 文件不存在: {file_path}")
    return factor_data

# ==================== 因子计算扩展 ====================

def calculate_additional_factors(df):
    """计算额外的技术因子"""
    df = df.copy()
    
    # 1. 收益率因子
    df['return_1d'] = df['c'].pct_change(1)  # 1日收益率
    df['return_5d'] = df['c'].pct_change(5)   # 5日收益率
    df['return_20d'] = df['c'].pct_change(20) # 20日收益率
    
    # 2. 动量因子
    df['momentum_5d'] = df['c'] / df['c'].shift(5) - 1  # 5日动量
    df['momentum_20d'] = df['c'] / df['c'].shift(20) - 1  # 20日动量
    
    # 3. 波动率因子
    df['volatility_5d'] = df['return_1d'].rolling(5).std()  # 5日波动率
    df['volatility_20d'] = df['return_1d'].rolling(20).std()  # 20日波动率
    
    # 4. 均值回归因子
    df['price_to_ma5'] = df['c'] / df['MA5'] - 1  # 价格相对MA5偏离度
    df['price_to_ma20'] = df['c'] / df['MA20'] - 1  # 价格相对MA20偏离度
    df['ma5_to_ma20'] = df['MA5'] / df['MA20'] - 1  # MA5相对MA20偏离度
    
    # 5. 成交量因子
    df['volume_ma5'] = df['v'].rolling(5).mean()
    df['volume_ratio'] = df['v'] / df['volume_ma5']  # 成交量比率
    
    # 6. RSI相关因子
    df['rsi_deviation'] = df['RSI14'] - 50  # RSI偏离中性
    
    return df

# ==================== IC值计算 ====================

def calculate_ic(factor_values, future_returns, method='pearson'):
    """
    计算因子IC值（Information Coefficient）
    
    参数:
        factor_values: 因子值序列
        future_returns: 未来收益率序列
        method: 'pearson' 或 'spearman'
    
    返回:
        IC值
    """
    # 对齐数据，去除NaN
    valid_idx = ~(pd.isna(factor_values) | pd.isna(future_returns))
    if valid_idx.sum() < 10:  # 至少需要10个有效数据点
        return np.nan
    
    factor_clean = factor_values[valid_idx]
    return_clean = future_returns[valid_idx]
    
    if method == 'pearson':
        ic, p_value = pearsonr(factor_clean, return_clean)
    else:
        ic, p_value = spearmanr(factor_clean, return_clean)
    
    return ic

def calculate_ic_analysis(df, forward_periods=[1, 3, 5, 10]):
    """
    计算不同时间窗口的IC值
    
    参数:
        df: 包含因子和价格的数据框
        forward_periods: 未来收益率的时间窗口列表
    
    返回:
        IC分析结果字典
    """
    ic_results = {}
    
    # 计算未来收益率
    for period in forward_periods:
        df[f'future_return_{period}d'] = df['c'].shift(-period) / df['c'] - 1
    
    # 定义所有因子列
    factor_columns = [
        'MA5', 'MA20', 'RSI14',
        'return_1d', 'return_5d', 'return_20d',
        'momentum_5d', 'momentum_20d',
        'volatility_5d', 'volatility_20d',
        'price_to_ma5', 'price_to_ma20', 'ma5_to_ma20',
        'volume_ratio', 'rsi_deviation'
    ]
    
    # 计算每个因子的IC值
    for factor in factor_columns:
        if factor not in df.columns:
            continue
        
        ic_dict = {}
        for period in forward_periods:
            future_return_col = f'future_return_{period}d'
            if future_return_col in df.columns:
                ic = calculate_ic(df[factor], df[future_return_col])
                ic_dict[f'IC_{period}d'] = ic
        
        if ic_dict:
            ic_results[factor] = ic_dict
    
    return ic_results

# ==================== 因子标准化 ====================

def standardize_factor(series, method='zscore', window=None):
    """
    因子标准化
    
    参数:
        series: 因子值序列
        method: 'zscore' (Z-score标准化) 或 'minmax' (Min-Max标准化)
        window: 滚动窗口大小（None表示使用全样本）
    
    返回:
        标准化后的序列
    """
    if method == 'zscore':
        if window is None:
            # 全样本标准化
            mean = series.mean()
            std = series.std()
            return (series - mean) / (std + 1e-8)
        else:
            # 滚动窗口标准化
            mean = series.rolling(window).mean()
            std = series.rolling(window).std()
            return (series - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        if window is None:
            min_val = series.min()
            max_val = series.max()
            return (series - min_val) / (max_val - min_val + 1e-8)
        else:
            min_val = series.rolling(window).min()
            max_val = series.rolling(window).max()
            return (series - min_val) / (max_val - min_val + 1e-8)
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ==================== 因子选择与降维 ====================

def analyze_factor_correlation(factor_data):
    """分析因子间相关性，识别冗余因子"""
    all_factors = []
    factor_names = []
    
    # 收集所有股票的因子数据
    for stock, df in factor_data.items():
        # 选择数值型因子列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        factor_cols = [col for col in numeric_cols if col not in ['c', 'v', 't']]
        
        for factor in factor_cols:
            if factor not in factor_names:
                factor_names.append(factor)
                all_factors.append(df[factor].values)
    
    # 创建因子矩阵
    factor_matrix = np.array(all_factors).T
    
    # 计算相关性矩阵
    factor_df = pd.DataFrame(factor_matrix, columns=factor_names)
    correlation_matrix = factor_df.corr()
    
    return correlation_matrix, factor_names

def select_factors_by_ic(ic_results, threshold=0.05, top_n=20):
    """
    基于IC值选择因子
    
    参数:
        ic_results: IC分析结果字典
        threshold: IC绝对值阈值
        top_n: 选择前N个因子
    
    返回:
        选中的因子列表
    """
    # 计算平均IC值（跨不同时间窗口）
    factor_ic_scores = {}
    
    for factor, ic_dict in ic_results.items():
        ic_values = [abs(v) for v in ic_dict.values() if not np.isnan(v)]
        if ic_values:
            factor_ic_scores[factor] = {
                'mean_ic': np.mean(ic_values),
                'max_ic': np.max(ic_values),
                'ic_dict': ic_dict
            }
    
    # 按平均IC值排序
    sorted_factors = sorted(factor_ic_scores.items(), 
                          key=lambda x: x[1]['mean_ic'], 
                          reverse=True)
    
    # 选择因子
    selected_factors = []
    for factor, scores in sorted_factors:
        if scores['mean_ic'] >= threshold:
            selected_factors.append({
                'factor': factor,
                'mean_ic': scores['mean_ic'],
                'max_ic': scores['max_ic'],
                'ic_details': scores['ic_dict']
            })
        if len(selected_factors) >= top_n:
            break
    
    return selected_factors

def remove_redundant_factors(correlation_matrix, selected_factors, threshold=0.8):
    """
    去除高度相关的冗余因子
    
    参数:
        correlation_matrix: 因子相关性矩阵
        selected_factors: 已选择的因子列表
        threshold: 相关性阈值（超过此值视为冗余）
    
    返回:
        去冗余后的因子列表
    """
    factor_names = [f['factor'] for f in selected_factors]
    final_factors = []
    removed_factors = []
    
    for i, factor1 in enumerate(factor_names):
        if factor1 not in correlation_matrix.columns:
            continue
        
        is_redundant = False
        for factor2 in final_factors:
            if factor2 in correlation_matrix.columns:
                corr = abs(correlation_matrix.loc[factor1, factor2])
                if corr > threshold:
                    # 保留IC值更高的因子
                    ic1 = selected_factors[i]['mean_ic']
                    ic2 = next(f['mean_ic'] for f in selected_factors if f['factor'] == factor2)
                    if ic1 <= ic2:
                        is_redundant = True
                        removed_factors.append(factor1)
                        break
        
        if not is_redundant:
            final_factors.append(factor1)
    
    return final_factors, removed_factors

# ==================== 可视化 ====================

def plot_ic_analysis(ic_results, stock_name, save_path=None):
    """绘制IC分析结果"""
    factors = list(ic_results.keys())
    periods = ['IC_1d', 'IC_3d', 'IC_5d', 'IC_10d']
    
    # 准备数据
    ic_data = []
    for factor in factors:
        for period in periods:
            if period in ic_results[factor]:
                ic_value = ic_results[factor][period]
                if not np.isnan(ic_value):
                    ic_data.append({
                        'Factor': factor,
                        'Period': period.replace('IC_', ''),
                        'IC': ic_value
                    })
    
    if not ic_data:
        print(f"  ⚠️ {stock_name}: 无有效IC数据")
        return
    
    ic_df = pd.DataFrame(ic_data)
    
    # 绘制热力图
    pivot_df = ic_df.pivot(index='Factor', columns='Period', values='IC')
    
    plt.figure(figsize=(10, max(6, len(factors) * 0.3)))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0, vmin=-0.3, vmax=0.3, cbar_kws={'label': 'IC值'})
    plt.title(f'{stock_name} - 因子IC值分析（不同时间窗口）', fontsize=14, fontweight='bold')
    plt.xlabel('预测时间窗口', fontsize=12)
    plt.ylabel('因子', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 保存IC分析图: {save_path}")
    plt.close()

def plot_factor_correlation(correlation_matrix, save_path=None):
    """绘制因子相关性矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={'label': '相关系数'})
    plt.title('因子相关性矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 保存相关性矩阵图: {save_path}")
    plt.close()

# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("Step 1.4 · 多因子模型框架设计")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n【1. 加载因子数据】")
    factor_data = load_factor_data()
    
    if not factor_data:
        print("❌ 未找到任何因子数据文件")
        return
    
    # 2. 计算扩展因子
    print("\n【2. 计算扩展技术因子】")
    extended_factors = {}
    for stock, df in factor_data.items():
        df_extended = calculate_additional_factors(df)
        extended_factors[stock] = df_extended
        print(f"  ✅ {stock}: 计算了 {len([c for c in df_extended.columns if c not in df.columns])} 个新因子")
    
    # 3. 计算IC值
    print("\n【3. 计算因子IC值（与未来收益的相关性）】")
    all_ic_results = {}
    for stock, df in extended_factors.items():
        print(f"\n  分析 {stock}...")
        ic_results = calculate_ic_analysis(df, forward_periods=[1, 3, 5, 10])
        all_ic_results[stock] = ic_results
        
        # 打印IC值摘要
        print(f"    计算了 {len(ic_results)} 个因子的IC值")
        for factor, ic_dict in list(ic_results.items())[:5]:  # 显示前5个
            ic_str = ", ".join([f"{k}={v:.3f}" for k, v in ic_dict.items() if not np.isnan(v)])
            if ic_str:
                print(f"    {factor}: {ic_str}")
    
    # 4. 绘制IC分析图
    print("\n【4. 生成IC分析可视化】")
    for stock, ic_results in all_ic_results.items():
        plot_ic_analysis(ic_results, stock, f"{stock}_IC_analysis.png")
    
    # 5. 因子选择
    print("\n【5. 因子选择（基于IC值）】")
    # 合并所有股票的IC结果（取平均）
    combined_ic = {}
    for stock, ic_results in all_ic_results.items():
        for factor, ic_dict in ic_results.items():
            if factor not in combined_ic:
                combined_ic[factor] = {}
            for period, ic_value in ic_dict.items():
                if period not in combined_ic[factor]:
                    combined_ic[factor][period] = []
                if not np.isnan(ic_value):
                    combined_ic[factor][period].append(ic_value)
    
    # 计算平均IC
    avg_ic_results = {}
    for factor, period_dict in combined_ic.items():
        avg_ic_results[factor] = {
            period: np.mean(values) if values else np.nan
            for period, values in period_dict.items()
        }
    
    selected_factors = select_factors_by_ic(avg_ic_results, threshold=0.02, top_n=20)
    print(f"\n  基于IC值选择了 {len(selected_factors)} 个因子:")
    for i, f in enumerate(selected_factors[:10], 1):  # 显示前10个
        print(f"    {i}. {f['factor']}: 平均IC={f['mean_ic']:.4f}, 最大IC={f['max_ic']:.4f}")
    
    # 6. 因子相关性分析
    print("\n【6. 因子相关性分析与降维】")
    correlation_matrix, factor_names = analyze_factor_correlation(extended_factors)
    
    # 绘制相关性矩阵
    plot_factor_correlation(correlation_matrix, "factor_correlation_matrix.png")
    
    # 去除冗余因子
    final_factors, removed_factors = remove_redundant_factors(
        correlation_matrix, selected_factors, threshold=0.8
    )
    
    print(f"\n  最终选择 {len(final_factors)} 个因子（去除 {len(removed_factors)} 个冗余因子）:")
    for i, factor in enumerate(final_factors[:15], 1):
        print(f"    {i}. {factor}")
    
    if removed_factors:
        print(f"\n  去除的冗余因子: {', '.join(removed_factors[:10])}")
    
    # 7. 因子标准化示例
    print("\n【7. 因子标准化方法】")
    print("  实现了两种标准化方法:")
    print("    - Z-score标准化: (x - mean) / std")
    print("    - Min-Max标准化: (x - min) / (max - min)")
    print("  支持全样本标准化和滚动窗口标准化")
    
    # 保存结果
    print("\n【8. 保存结果】")
    
    # 保存IC分析结果
    ic_summary = []
    for factor_info in selected_factors:
        row = {
            'factor': factor_info['factor'],
            'mean_ic': factor_info['mean_ic'],
            'max_ic': factor_info['max_ic']
        }
        row.update(factor_info['ic_details'])
        ic_summary.append(row)
    
    ic_df = pd.DataFrame(ic_summary)
    ic_df.to_csv('factor_ic_summary.csv', index=False, encoding='utf-8-sig')
    print(f"  ✅ 保存IC分析结果: factor_ic_summary.csv")
    
    # 保存最终选择的因子列表
    final_factors_df = pd.DataFrame({
        'factor': final_factors,
        'rank': range(1, len(final_factors) + 1)
    })
    final_factors_df.to_csv('selected_factors.csv', index=False, encoding='utf-8-sig')
    print(f"  ✅ 保存因子列表: selected_factors.csv")
    
    # 保存相关性矩阵
    correlation_matrix.to_csv('factor_correlation_matrix.csv', encoding='utf-8-sig')
    print(f"  ✅ 保存相关性矩阵: factor_correlation_matrix.csv")
    
    print("\n" + "=" * 60)
    print("Step 1.4 完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - factor_ic_summary.csv: IC分析结果")
    print(f"  - selected_factors.csv: 最终选择的因子列表")
    print(f"  - factor_correlation_matrix.csv: 因子相关性矩阵")
    print(f"  - *_IC_analysis.png: 各股票IC分析热力图")
    print(f"  - factor_correlation_matrix.png: 因子相关性热力图")
    print("=" * 60)

if __name__ == "__main__":
    main()

