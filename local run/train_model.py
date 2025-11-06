"""
多因子模型训练脚本

在执行前，请先配置 model_training_config.md 中的信息
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
# 使用默认配置（与step1.py一致）

CONFIG = {
    # 数据配置
    "stock_code": "00005HK",  # 默认股票代码
    "train_start": "2020-01-01",
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end": "2024-06-30",
    "test_start": "2024-07-01",
    "test_end": "2024-12-31",
    
    # 新闻配置
    "news_lang": "en",
    "news_categories": ["ECONOMY", "MARKET", "STOCK"],
    
    # 经济数据配置（使用step1.py中的默认配置）
    "econ_series": {
        "GDP_Growth": "A193RC1A027NBEA",
        "CPI": "CPIAUCSL"
    },
    
    # 模型配置
    "prediction_window": 5,  # 预测未来5天收益率
    "model_type": "LinearRegression",  # "LinearRegression" / "RandomForestRegressor" / "XGBoost"
    
    # 因子配置（已从step1.4确定）
    "selected_factors": [
        "price_to_ma20",
        "MA5",
        "price_to_ma5",
        "return_1d",
        "volatility_20d",
        "volume_ratio"
    ],
    
    # 标准化
    "standardize_method": "zscore",
    
    # API配置
    "user": "2d72ec3382",
    "api_key": "ff4bfb0c2d2fe8d64d0f523550567237617689c2cea999568bcf9f2f28d0c70e"
}

# ==================== API客户端 ====================

class AlgoGeneClient:
    """ALGOGENE REST API 客户端"""
    
    def __init__(self, user: str, api_key: str):
        self.user = user
        self.api_key = api_key
        self.base_url = "https://algogene.com/rest/v1"
        self.headers = {'Content-Type': 'application/json'}
    
    def _request(self, endpoint: str, params: dict):
        """发送API请求"""
        import requests
        url = f"{self.base_url}/{endpoint}"
        params['user'] = self.user
        params['api_key'] = self.api_key
        try:
            res = requests.get(url, params=params, headers=self.headers, timeout=120)
            res.raise_for_status()
            data = res.json()
            if isinstance(data, dict) and 'res' in data:
                data = data['res']
            return data
        except Exception as e:
            print(f"  API请求错误 ({endpoint}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  响应状态码: {e.response.status_code}")
                print(f"  响应内容: {e.response.text[:200]}")
            return {}
    
    def get_historical_price(self, instrument: str, count: int, interval: str, timestamp: str):
        """获取历史价格数据"""
        params = {
            "instrument": instrument,
            'count': count,
            'interval': interval,
            'timestamp': timestamp
        }
        data = self._request("history_price", params)
        
        if data and isinstance(data, dict):
            rows = list(data.values())
            df = pd.DataFrame(rows)
            if "t" in df.columns:
                df["t"] = pd.to_datetime(df["t"])
                df = df.sort_values("t").reset_index(drop=True)
            return df
        elif data and isinstance(data, list):
            # 如果返回的是列表格式
            df = pd.DataFrame(data)
            if "t" in df.columns:
                df["t"] = pd.to_datetime(df["t"])
                df = df.sort_values("t").reset_index(drop=True)
            return df
        return pd.DataFrame()
    
    def get_historical_news(self, lang: str, count: int, starttime: str, endtime: str = None):
        """获取历史新闻数据"""
        params = {
            'lang': lang,
            'count': count,
            'starttime': starttime
        }
        if endtime:
            params['endtime'] = endtime
        data = self._request("history_news", params)
        if data:
            return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
        return pd.DataFrame()
    
    def get_economic_data(self, series_id: str, start_date: str, end_date: str):
        """
        获取经济指标数据
        
        根据 API 文档：
        - 端点: /history_econs_stat (注意没有空格)
        - 参数: starttime, endtime (不是 start_date, end_date)
        - 响应: {'count': int, 'res': [{'date': str, 'value': str, 'series_id': str}, ...]}
        """
        params = {
            'series_id': series_id,
            'starttime': start_date,  # API 文档要求使用 starttime
            'endtime': end_date        # API 文档要求使用 endtime
        }
        # 注意：_request 方法已经处理了 {'res': {...}} 嵌套结构，直接返回 res 的内容
        # 所以这里 data 可能是列表（如果 res 是列表）或字典（如果 res 是字典）
        data = self._request("history_econs_stat", params)
        
        if data:
            # 如果返回的是列表（res 字段的内容）
            if isinstance(data, list):
                if len(data) > 0:
                    df = pd.DataFrame(data)
                    if not df.empty and "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        # value 可能是字符串，转换为数值
                        if "value" in df.columns:
                            df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    return df
            # 如果返回的是字典（可能是错误信息或其他格式）
            elif isinstance(data, dict):
                if 'res' in data:
                    if isinstance(data['res'], list):
                        df = pd.DataFrame(data['res'])
                        if not df.empty and "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            if "value" in df.columns:
                                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                        return df
                    elif isinstance(data['res'], str):
                        print(f"  API 返回消息: {data['res']}")
        return pd.DataFrame()

# ==================== 因子计算 ====================

def calculate_price_factors(df):
    """计算价格因子"""
    factors_df = df.copy()
    
    # 1. price_to_ma20
    factors_df['MA20'] = factors_df['c'].rolling(20).mean()
    factors_df['price_to_ma20'] = factors_df['c'] / factors_df['MA20'] - 1
    
    # 2. MA5
    factors_df['MA5'] = factors_df['c'].rolling(5).mean()
    
    # 3. price_to_ma5
    factors_df['price_to_ma5'] = factors_df['c'] / factors_df['MA5'] - 1
    
    # 4. return_1d
    factors_df['return_1d'] = factors_df['c'].pct_change(1)
    
    # 5. volatility_20d
    factors_df['volatility_20d'] = factors_df['return_1d'].rolling(20).std()
    
    # 6. volume_ratio
    factors_df['volume_ma5'] = factors_df['v'].rolling(5).mean()
    factors_df['volume_ratio'] = factors_df['v'] / factors_df['volume_ma5']
    
    return factors_df

def calculate_news_factors(news_df, price_df):
    """计算新闻因子（按日期聚合）"""
    if news_df.empty:
        return pd.DataFrame()
    
    # 提取日期
    if 'published' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['published']).dt.date
    elif 'date' in news_df.columns:
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    else:
        return pd.DataFrame()
    
    # 简单情绪分析
    positive_keywords = ["涨", "利好", "增长", "盈利", "突破", "up", "gain", "beat", "rise", "increase"]
    negative_keywords = ["跌", "利空", "下降", "亏损", "危机", "down", "loss", "miss", "fall", "decrease"]
    
    def simple_sentiment(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        score = 0
        for word in positive_keywords:
            if word.lower() in text_lower:
                score += 1
        for word in negative_keywords:
            if word.lower() in text_lower:
                score -= 1
        return score
    
    # 计算情绪得分
    if 'title' in news_df.columns:
        news_df['sentiment'] = news_df['title'].apply(simple_sentiment)
    elif 'text' in news_df.columns:
        news_df['sentiment'] = news_df['text'].apply(simple_sentiment)
    else:
        news_df['sentiment'] = 0
    
    # 按日期聚合
    daily_news = news_df.groupby('date').agg(
        news_count=('sentiment', 'size'),
        news_sentiment=('sentiment', 'sum')
    ).reset_index()
    
    # 对齐到价格数据日期
    price_df['date'] = pd.to_datetime(price_df['t']).dt.date
    merged = price_df.merge(daily_news, on='date', how='left')
    merged['news_count'] = merged['news_count'].fillna(0)
    merged['news_sentiment'] = merged['news_sentiment'].fillna(0)
    
    return merged[['date', 'news_count', 'news_sentiment']]

def calculate_econ_factors(econ_data_dict, price_df):
    """计算经济因子（前向填充）"""
    price_df = price_df.copy()
    price_df['date'] = pd.to_datetime(price_df['t']).dt.date
    
    for name, econ_df in econ_data_dict.items():
        if econ_df.empty:
            price_df[f'econ_{name}'] = 0.0
            continue
        
        if 'date' in econ_df.columns:
            econ_df['date'] = pd.to_datetime(econ_df['date']).dt.date
            # 前向填充
            econ_series = econ_df.set_index('date')['value']
            price_df[f'econ_{name}'] = price_df['date'].map(econ_series).fillna(method='ffill').fillna(0.0)
        else:
            price_df[f'econ_{name}'] = 0.0
    
    return price_df

# ==================== 数据准备 ====================

def prepare_training_data(config, client):
    """准备训练数据"""
    print("=" * 60)
    print("准备训练数据")
    print("=" * 60)
    
    stock_code = config["stock_code"]
    
    # 1. 获取价格数据
    print(f"\n【1. 获取价格数据: {stock_code}】")
    try:
        price_df = client.get_historical_price(
            stock_code,
            count=2000,
            interval="D",
            timestamp=config["test_end"]
        )
        if price_df.empty:
            print(f"  ⚠️ 警告: API返回空数据，尝试检查API连接和参数")
            print(f"  参数: instrument={stock_code}, count=2000, interval=D, timestamp={config['test_end']}")
            raise ValueError(f"无法获取价格数据: {stock_code}")
        print(f"  获取到 {len(price_df)} 条价格数据")
        print(f"  数据列: {list(price_df.columns)}")
        print(f"  时间范围: {price_df['t'].min()} 至 {price_df['t'].max()}")
    except Exception as e:
        print(f"  ❌ 获取价格数据失败: {e}")
        raise
    
    # 2. 计算价格因子
    print(f"\n【2. 计算价格因子】")
    price_df = calculate_price_factors(price_df)
    print(f"  计算了 {len(config['selected_factors'])} 个价格因子")
    
    # 3. 新闻因子（暂时跳过）
    print(f"\n【3. 新闻因子（暂时跳过）】")
    price_df['news_count'] = 0
    price_df['news_sentiment'] = 0
    print("  新闻因子已设置为0（暂时跳过）")
    
    # 4. 获取经济数据
    print(f"\n【4. 获取经济数据】")
    econ_data_dict = {}
    # API 限制：历史数据只能从 2019-12-08 之后访问
    econ_start_date = max(config["train_start"], "2019-12-09")
    
    for name, series_id in config["econ_series"].items():
        if series_id:
            try:
                econ_df = client.get_economic_data(
                    series_id,
                    econ_start_date,
                    config["test_end"]
                )
                if not econ_df.empty:
                    print(f"  {name}: 获取到 {len(econ_df)} 条数据")
                    econ_data_dict[name] = econ_df
                else:
                    print(f"  {name}: 未获取到数据")
            except Exception as e:
                print(f"  {name}: 获取失败 - {e}")
        else:
            print(f"  {name}: 未配置series_id，跳过")
    
    if econ_data_dict:
        price_df = calculate_econ_factors(econ_data_dict, price_df)
    
    # 5. 计算未来收益率（预测目标）
    print(f"\n【5. 计算预测目标（未来{config['prediction_window']}天收益率）】")
    window = config["prediction_window"]
    price_df['future_return'] = price_df['c'].shift(-window) / price_df['c'] - 1
    
    # 6. 划分数据集
    print(f"\n【6. 划分数据集】")
    price_df['date'] = pd.to_datetime(price_df['t'])
    
    train_df = price_df[
        (price_df['date'] >= config["train_start"]) &
        (price_df['date'] <= config["train_end"])
    ].copy()
    
    val_df = price_df[
        (price_df['date'] >= config["val_start"]) &
        (price_df['date'] <= config["val_end"])
    ].copy()
    
    test_df = price_df[
        (price_df['date'] >= config["test_start"]) &
        (price_df['date'] <= config["test_end"])
    ].copy()
    
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    # 7. 准备特征和标签
    factor_cols = config["selected_factors"].copy()
    
    # 新闻因子暂时跳过，不加入特征列表
    # if 'news_count' in price_df.columns:
    #     factor_cols = factor_cols + ['news_count', 'news_sentiment']
    
    # 添加经济因子（如果有）
    econ_cols = [col for col in price_df.columns if col.startswith('econ_')]
    factor_cols = factor_cols + econ_cols
    
    # 去除缺失值
    train_df = train_df.dropna(subset=factor_cols + ['future_return'])
    val_df = val_df.dropna(subset=factor_cols + ['future_return'])
    test_df = test_df.dropna(subset=factor_cols + ['future_return'])
    
    X_train = train_df[factor_cols].values
    y_train = train_df['future_return'].values
    
    X_val = val_df[factor_cols].values
    y_val = val_df['future_return'].values
    
    X_test = test_df[factor_cols].values
    y_test = test_df['future_return'].values
    
    print(f"\n  最终特征数: {len(factor_cols)}")
    print(f"  特征列表: {factor_cols}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'factor_names': factor_cols,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }

# ==================== 模型训练 ====================

def train_model(X_train, y_train, X_val, y_val, model_type, factor_names):
    """训练模型"""
    print("\n" + "=" * 60)
    print("训练模型")
    print("=" * 60)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 选择模型
    if model_type == "LinearRegression":
        model = LinearRegression()
        print(f"\n使用模型: {model_type}")
    elif model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        print(f"\n使用模型: {model_type} (n_estimators=200, max_depth=10)")
    elif model_type == "XGBoost":
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        print(f"\n使用模型: {model_type} (n_estimators=200, max_depth=6)")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练
    print(f"\n训练中...")
    model.fit(X_train_scaled, y_train)
    
    # 评估
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n训练集表现:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\n验证集表现:")
    print(f"  MSE: {val_mse:.6f}")
    print(f"  MAE: {val_mae:.6f}")
    print(f"  R²: {val_r2:.4f}")
    
    # 特征重要性（随机森林或XGBoost）
    if model_type in ["RandomForestRegressor", "XGBoost"]:
        print(f"\n特征重要性:")
        importances = model.feature_importances_
        for i, (name, imp) in enumerate(zip(factor_names, importances)):
            print(f"  {i+1}. {name}: {imp:.4f}")
    
    return model, scaler

# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("多因子模型训练")
    print("=" * 60)
    print("\n使用默认配置（与step1.py一致）\n")
    
    # 检查配置
    econ_series = CONFIG["econ_series"]
    if not econ_series or all(v is None for v in econ_series.values()):
        print("⚠️ 警告: 经济数据series_id未配置，将跳过经济因子")
    
    # 创建输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 初始化API客户端
    client = AlgoGeneClient(CONFIG["user"], CONFIG["api_key"])
    
    # 准备数据
    data = prepare_training_data(CONFIG, client)
    
    # 训练模型
    model, scaler = train_model(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        CONFIG["model_type"],
        data['factor_names']
    )
    
    # 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)
    X_test_scaled = scaler.transform(data['X_test'])
    y_test_pred = model.predict(X_test_scaled)
    
    test_mse = mean_squared_error(data['y_test'], y_test_pred)
    test_mae = mean_absolute_error(data['y_test'], y_test_pred)
    test_r2 = r2_score(data['y_test'], y_test_pred)
    
    print(f"\n测试集表现:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²: {test_r2:.4f}")
    
    # 保存模型
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)
    
    model_file = os.path.join(model_dir, f"model_{CONFIG['stock_code']}_{CONFIG['model_type']}.pkl")
    scaler_file = os.path.join(model_dir, f"scaler_{CONFIG['stock_code']}.pkl")
    config_file = os.path.join(model_dir, f"config_{CONFIG['stock_code']}.json")
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ 模型已保存: {model_file}")
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✅ 标准化器已保存: {scaler_file}")
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'factor_names': data['factor_names'],
            'train_size': len(data['X_train']),
            'val_size': len(data['X_val']),
            'test_size': len(data['X_test']),
            'train_mse': float(mean_squared_error(data['y_train'], model.predict(scaler.transform(data['X_train'])))),
            'val_mse': float(mean_squared_error(data['y_val'], model.predict(scaler.transform(data['X_val'])))),
            'test_mse': float(test_mse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2)
        }, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 配置已保存: {config_file}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

