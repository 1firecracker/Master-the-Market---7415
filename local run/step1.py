"""
Step 1 · 多因子模型数据探索

目标：获取股票价格、新闻、经济指标数据，计算技术因子和情绪因子。
使用 REST API 在本地运行，不依赖平台服务器。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import os
from typing import Dict, List, Optional

# ==================== 配置 ====================
# 从环境变量或直接配置（请替换为你的实际值）
USER = "2d72ec3382"
API_KEY = "ff4bfb0c2d2fe8d64d0f523550567237617689c2cea999568bcf9f2f28d0c70e"
BASE_URL = "https://algogene.com/rest/v1"

stocks = ["00005HK", "00939HK", "00700HK", "00941HK"]
END_DATE = "2024-12-31"

# ==================== API 客户端 ====================

class AlgoGeneClient:
    """ALGOGENE REST API 客户端"""
    
    def __init__(self, user: str, api_key: str):
        self.user = user
        self.api_key = api_key
        self.base_url = BASE_URL
        self.headers = {'Content-Type': 'application/json'}
    
    def _request(self, endpoint: str, params: dict) -> dict:
        """发送 API 请求"""
        url = f"{self.base_url}/{endpoint}"
        params['user'] = self.user
        params['api_key'] = self.api_key
        try:
            res = requests.get(url, params=params, headers=self.headers, timeout=60)
            res.raise_for_status()
            data = res.json()
            # 处理可能的嵌套结构（如 {'res': {...}}）
            if isinstance(data, dict) and 'res' in data:
                data = data['res']
            # 调试：打印响应状态
            if not data:
                print(f"  警告: {endpoint} 返回空数据")
            return data
        except Exception as e:
            print(f"  API 请求错误 ({endpoint}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  响应状态码: {e.response.status_code}")
                print(f"  响应内容: {e.response.text[:200]}")
            return {}
    
    def get_historical_price(self, instrument: str, count: int, interval: str, timestamp: str) -> pd.DataFrame:
        """获取历史价格数据"""
        # 注意：REST API 端点可能需要调整，如果失败则返回空 DataFrame
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
    
    def get_historical_news(self, lang: str = "en", count: int = 1000, 
                           starttime: str = "2020-01-01",
                           category: List[str] = None) -> pd.DataFrame:
        """获取历史新闻数据"""
        if category is None:
            category = ["ECONOMY", "MARKET", "STOCK"]
        
        params = {
            'lang': lang,
            'count': count,
            'starttime': starttime,
            'category': category
        }
        data = self._request("history_news", params)
        
        if data:
            return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
        return pd.DataFrame()
    
    def get_economic_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
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

# ==================== 数据获取函数 ====================

def fetch_stock_daily(client: AlgoGeneClient, instrument: str, 
                     count: int = 1000, end_ts: str = END_DATE) -> pd.DataFrame:
    """获取股票日线数据"""
    df = client.get_historical_price(instrument, count, "D", end_ts)
    return df

# ==================== 技术因子计算 ====================

def ta_ma(series: pd.Series, window: int) -> pd.Series:
    """计算移动平均线"""
    return series.rolling(window).mean()

def ta_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

# ==================== 新闻情绪分析 ====================

def simple_sentiment(text: str) -> int:
    """简单关键词情绪分析"""
    pos = ["涨", "利好", "增长", "盈利", "突破", "up", "gain", "beat"]
    neg = ["跌", "利空", "下降", "亏损", "危机", "down", "loss", "miss"]
    if not isinstance(text, str):
        return 0
    return sum(1 for w in pos if w in text) - sum(1 for w in neg if w in text)

# ==================== 主程序 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("Step 1 · 多因子模型数据探索")
    print("使用 REST API 在本地运行")
    print("=" * 60)
    print("\n注意：如果 API 调用失败，可能是：")
    print("1. API 端点需要平台内访问")
    print("2. 网络连接问题")
    print("3. API 密钥或用户信息不正确")
    print("=" * 60)
    
    # 确保输出目录存在（脚本在 local run 目录中运行，直接保存到当前目录）
    output_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    os.chdir(output_dir)  # 切换到脚本所在目录
    
    # 初始化 API 客户端
    client = AlgoGeneClient(USER, API_KEY)
    
    # ========== 1. 获取股票数据 ==========
    print("\n[1/4] 获取股票价格数据...")
    stock_frames = {}
    for s in stocks:
        print(f"  正在获取 {s} 的数据...")
        df = fetch_stock_daily(client, s)
        stock_frames[s] = df
        if not df.empty and "t" in df.columns and "c" in df.columns:
            print(f"  {s}: {df.shape[0]} 条数据, 时间范围: {df['t'].min()} 至 {df['t'].max()}")
            
            # 绘制价格图
            plt.figure(figsize=(8, 3))
            plt.plot(df["t"].values, df["c"].values, label=f"{s}")
            plt.title(f"{s} Closing Price")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(f"{s}_price.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print(f"  {s}: 数据为空")
    
    # ========== 2. 计算相关性 ==========
    print("\n[2/4] 计算股票收益率相关性...")
    date_sets = [set(f["t"]) for f in stock_frames.values() if not f.empty and "t" in f.columns]
    if len(date_sets) > 0:
        common_dates = set.intersection(*date_sets) if len(date_sets) > 1 else date_sets[0]
        ret_df = []
        for s, df in stock_frames.items():
            if df.empty or "t" not in df.columns:
                continue
            df2 = df[df["t"].isin(common_dates)].sort_values("t").copy()
            if len(df2) > 1:
                df2["ret"] = df2["c"].pct_change()
                ret_df.append(df2[["t", "ret"]].set_index("t").rename(columns={"ret": s}))
        if ret_df:
            ret_panel = pd.concat(ret_df, axis=1).dropna()
            if len(ret_panel) > 0:
                corr = ret_panel.corr()
                print("\n相关性矩阵:")
                print(corr)
                corr.to_csv("stock_correlation.csv")
    
    # ========== 3. 计算技术因子 ==========
    print("\n[3/4] 计算技术因子...")
    tech_factors = {}
    for s, df in stock_frames.items():
        if df.empty:
            print(f"  {s}: 跳过（数据为空）")
            continue
        # 检查必要的列是否存在
        required_cols = ["t", "c", "v"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  {s}: 跳过（缺少列: {missing_cols}，可用列: {list(df.columns)}）")
            continue
        tmp = df[["t", "c", "v"]].copy()
        tmp["MA5"] = ta_ma(tmp["c"], 5)
        tmp["MA20"] = ta_ma(tmp["c"], 20)
        tmp["RSI14"] = ta_rsi(tmp["c"], 14)
        tech_factors[s] = tmp
        print(f"\n{s} 技术因子 (最后3行):")
        print(tmp.tail(3))
        tmp.to_csv(f"{s}_tech_factors.csv", index=False)
    
    # ========== 4. 新闻情绪因子 ==========
    print("\n[4/4] 获取新闻数据并计算情绪因子...")
    news_df = client.get_historical_news(
        lang="en", 
        count=1000, 
        starttime="2020-01-01", 
        category=["ECONOMY", "MARKET", "STOCK"]
    )
    
    if not news_df.empty:
        print(f"  获取到 {len(news_df)} 条新闻")
        # 查找时间列
        time_col = None
        for col in ["published", "time", "timestamp", "date", "t"]:
            if col in news_df.columns:
                time_col = col
                break
        
        if time_col:
            news_df["date"] = pd.to_datetime(news_df[time_col]).dt.date
        
        # 计算情绪分数
        if "title" in news_df.columns:
            news_df["sentiment"] = news_df["title"].apply(simple_sentiment)
        elif "text" in news_df.columns:
            news_df["sentiment"] = news_df["text"].apply(simple_sentiment)
        elif "content" in news_df.columns:
            news_df["sentiment"] = news_df["content"].apply(simple_sentiment)
        else:
            news_df["sentiment"] = 0
        
        # 按日期聚合
        if "date" in news_df.columns:
            daily_news = news_df.groupby("date").agg(
                news_count=("sentiment", "size"),
                sentiment_sum=("sentiment", "sum")
            ).reset_index()
            print("\n每日新闻统计 (前5行):")
            print(daily_news.head())
            daily_news.to_csv("daily_news.csv", index=False)
            
            # 绘制新闻数量图
            plt.figure(figsize=(10, 3))
            plt.plot(pd.to_datetime(daily_news["date"]).values, daily_news["news_count"].values)
            plt.title("Daily News Count")
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig("daily_news_count.png", dpi=150, bbox_inches='tight')
            plt.close()
    else:
        print("  未获取到新闻数据")
    
    # ========== 5. 经济指标 ==========
    print("\n[5/5] 获取经济指标数据...")
    economic_indicators = {
        "CPI": "CPIAUCSL",
        "GDP_Growth": "A193RC1A027NBEA"
    }
    
    # API 限制：历史数据只能从 2019-12-08 之后访问
    econ_start_date = "2019-12-09"  # 使用 API 允许的最早日期
    
    econ_frames = {}
    for name, series_id in economic_indicators.items():
        try:
            print(f"  正在获取 {name} ({series_id})...")
            df = client.get_economic_data(series_id, econ_start_date, END_DATE)
            if not df.empty and "date" in df.columns:
                econ_frames[name] = df
                print(f"  {name}: {df.shape[0]} 条数据")
                print(f"  最后3行:")
                print(df.tail(3))
                df.to_csv(f"{name}_economic.csv", index=False)
                
                # 绘制经济指标图
                plt.figure(figsize=(8, 3))
                plt.plot(df["date"].values, pd.to_numeric(df["value"], errors="coerce").values)
                plt.title(name)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(f"{name}_economic.png", dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print(f"  {name}: 数据为空")
        except Exception as e:
            print(f"  {name} 获取失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"数据探索完成！所有结果已保存到: {os.getcwd()}")
    print("=" * 60)

if __name__ == "__main__":
    main()

