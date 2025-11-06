# 本地使用 ALGOGENE API 指南

## 概述

由于平台服务器可能较慢，可以通过 REST API 在本地 Python 环境中直接调用 ALGOGENE 的数据接口，无需依赖平台的 Jupyter Notebook 环境。

## 优势

- ✅ **速度快**：本地运行，不受平台服务器延迟影响
- ✅ **灵活性高**：可以使用本地 IDE（如 VS Code、PyCharm）
- ✅ **离线开发**：可以先下载数据，然后离线分析
- ✅ **批量处理**：可以编写脚本批量获取数据

## 准备工作

### 1. 获取 API 凭证

1. 登录 [ALGOGENE 平台](https://algogene.com/login)
2. 进入 `[Settings]` > `[User Profile]`
3. 获取 `USER`（用户名）和 `API_KEY`（API密钥）

### 2. 安装依赖

```bash
pip install requests pandas numpy matplotlib
```

## API 使用方法

### 1. 历史价格数据（Market Data）

```python
import requests
import pandas as pd
from datetime import datetime

USER = "your_username"
API_KEY = "your_api_key"

def get_historical_price(instrument, count, interval, timestamp):
    """
    获取历史价格数据
    
    参数:
        instrument: 标的代码，如 "00005HK", "EURUSD"
        count: 数据条数
        interval: 时间间隔，'D'=日线, 'H'=小时, 'M'=分钟
        timestamp: 截止时间，格式 "YYYY-MM-DD"
    
    返回:
        DataFrame，包含 t, o, h, l, c, v 等列
    """
    url = 'https://algogene.com/rest/v1/history_price'
    headers = {'Content-Type': 'application/json'}
    params = {
        "instrument": instrument,
        'user': USER,
        'api_key': API_KEY,
        'count': count,
        'interval': interval,
        'timestamp': timestamp
    }
    res = requests.get(url, params=params, headers=headers)
    data = res.json()
    
    # 转换为 DataFrame
    if data and isinstance(data, dict):
        rows = list(data.values())
        df = pd.DataFrame(rows)
        if "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"])
            df = df.sort_values("t").reset_index(drop=True)
        return df
    return pd.DataFrame()

# 使用示例
df = get_historical_price("00005HK", 1000, "D", "2024-12-31")
print(df.head())
```

### 2. 历史新闻数据（News Data）

```python
def get_historical_news(lang="en", count=1000, starttime="2020-01-01", 
                        category=["ECONOMY", "MARKET", "STOCK"]):
    """
    获取历史新闻数据
    
    参数:
        lang: 语言代码，'en', 'zh', 'ja' 等
        count: 获取数量
        starttime: 开始时间
        category: 新闻类别列表
    
    返回:
        DataFrame
    """
    url = "https://algogene.com/rest/v1/history_news"
    headers = {'Content-Type': 'application/json'}
    params = {
        'user': USER,
        'api_key': API_KEY,
        'lang': lang,
        'count': count,
        'starttime': starttime,
        'category': category
    }
    res = requests.get(url, params=params, headers=headers)
    data = res.json()
    
    if data:
        return pd.DataFrame(data)
    return pd.DataFrame()

# 使用示例
news_df = get_historical_news(lang="en", count=1000, starttime="2020-01-01",
                              category=["ECONOMY", "MARKET", "STOCK"])
print(news_df.head())
```

### 3. 实时新闻数据

```python
def get_realtime_news(lang="en"):
    """
    获取实时新闻数据
    
    参数:
        lang: 语言代码
    
    返回:
        DataFrame
    """
    url = "https://algogene.com/rest/v1/realtime_news"
    headers = {'Content-Type': 'application/json'}
    params = {
        'user': USER,
        'api_key': API_KEY,
        'lang': lang
    }
    res = requests.get(url, params=params, headers=headers)
    data = res.json()
    
    if data:
        return pd.DataFrame(data)
    return pd.DataFrame()
```

### 4. 经济指标数据（Economic Data）

```python
def get_economic_data(series_id, start_date, end_date):
    """
    获取经济指标数据
    
    参数:
        series_id: 经济指标系列ID，如 "CPIAUCSL"
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"
    
    返回:
        DataFrame，包含 date 和 value 列
    """
    url = "https://algogene.com/rest/v1/history_econstat"
    headers = {'Content-Type': 'application/json'}
    params = {
        'user': USER,
        'api_key': API_KEY,
        'series_id': series_id,
        'start_date': start_date,
        'end_date': end_date
    }
    res = requests.get(url, params=params, headers=headers)
    data = res.json()
    
    if data:
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

# 使用示例
cpi_df = get_economic_data("CPIAUCSL", "2019-01-01", "2024-12-31")
print(cpi_df.head())
```

## 完整示例：本地数据获取脚本

```python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 配置
USER = "your_username"
API_KEY = "your_api_key"
BASE_URL = "https://algogene.com/rest/v1"

class AlgoGeneClient:
    """ALGOGENE API 客户端"""
    
    def __init__(self, user, api_key):
        self.user = user
        self.api_key = api_key
        self.base_url = BASE_URL
        self.headers = {'Content-Type': 'application/json'}
    
    def _request(self, endpoint, params):
        """发送 API 请求"""
        url = f"{self.base_url}/{endpoint}"
        params['user'] = self.user
        params['api_key'] = self.api_key
        res = requests.get(url, params=params, headers=self.headers)
        return res.json()
    
    def get_historical_price(self, instrument, count, interval, timestamp):
        """获取历史价格"""
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
        return pd.DataFrame()
    
    def get_historical_news(self, lang="en", count=1000, starttime="2020-01-01",
                           category=["ECONOMY", "MARKET", "STOCK"]):
        """获取历史新闻"""
        params = {
            'lang': lang,
            'count': count,
            'starttime': starttime,
            'category': category
        }
        data = self._request("history_news", params)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()
    
    def get_economic_data(self, series_id, start_date, end_date):
        """获取经济指标"""
        params = {
            'series_id': series_id,
            'start_date': start_date,
            'end_date': end_date
        }
        data = self._request("history_econstat", params)
        if data:
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = AlgoGeneClient(USER, API_KEY)
    
    # 获取股票数据
    stocks = ["00005HK", "00939HK", "00700HK", "00941HK"]
    stock_frames = {}
    for stock in stocks:
        df = client.get_historical_price(stock, 1000, "D", "2024-12-31")
        stock_frames[stock] = df
        print(f"{stock}: {df.shape}")
    
    # 获取新闻数据
    news_df = client.get_historical_news(lang="en", count=1000, 
                                         starttime="2020-01-01",
                                         category=["ECONOMY", "MARKET", "STOCK"])
    print(f"新闻数据: {news_df.shape}")
    
    # 获取经济指标
    cpi_df = client.get_economic_data("CPIAUCSL", "2019-01-01", "2024-12-31")
    print(f"CPI数据: {cpi_df.shape}")
```

## 数据缓存策略

为了提高效率，可以缓存已下载的数据：

```python
import os
import pickle
from datetime import datetime, timedelta

CACHE_DIR = "./data_cache"

def get_cached_data(cache_key):
    """从缓存读取数据"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        # 检查缓存是否过期（例如24小时）
        mtime = os.path.getmtime(cache_file)
        if datetime.now() - datetime.fromtimestamp(mtime) < timedelta(hours=24):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    return None

def save_to_cache(cache_key, data):
    """保存数据到缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

# 使用缓存的数据获取函数
def get_historical_price_cached(instrument, count, interval, timestamp):
    cache_key = f"price_{instrument}_{count}_{interval}_{timestamp}"
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached
    
    # 从 API 获取
    df = get_historical_price(instrument, count, interval, timestamp)
    save_to_cache(cache_key, df)
    return df
```

## 注意事项

1. **API 限制**：注意 API 的调用频率限制，避免过于频繁的请求
2. **数据格式**：REST API 返回的 JSON 格式可能与平台内 API 略有不同，需要适配
3. **错误处理**：添加适当的错误处理和重试机制
4. **安全性**：不要将 API_KEY 提交到版本控制系统，使用环境变量或配置文件

## 环境变量配置

创建 `.env` 文件：

```env
ALGOGENE_USER=your_username
ALGOGENE_API_KEY=your_api_key
```

在代码中读取：

```python
import os
from dotenv import load_dotenv

load_dotenv()
USER = os.getenv("ALGOGENE_USER")
API_KEY = os.getenv("ALGOGENE_API_KEY")
```

## 与平台 API 的对比

| 特性 | 平台 Jupyter Notebook | 本地 REST API |
|------|----------------------|---------------|
| 速度 | 受服务器影响 | 本地运行，速度快 |
| 环境 | 平台提供 | 需要本地配置 |
| 数据访问 | 通过 AlgoAPI 模块 | 通过 HTTP 请求 |
| 离线使用 | 不支持 | 支持（缓存后） |
| 批量处理 | 受限 | 灵活 |

## 参考资源

- ALGOGENE 数据获取指南：`knowledge/数据获取指南.md`
- ALGOGENE 知识库：`knowledge/知识.md`

