"""
下载 0005.HK 历史数据脚本
从 Yahoo Finance 获取汇丰控股（0005.HK）2020年的历史数据
"""
import sys

# 检查并导入必要的库
try:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime
    import json
    import os
except ImportError as e:
    print(f"导入错误: {e}")
    print("请先安装必要的依赖库:")
    print("  pip install yfinance pandas numpy")
    sys.exit(1)

def download_hk_stock_data(ticker="0005.HK", start_date="2020-01-01", end_date="2020-12-31"):
    """
    下载港股历史数据
    
    Parameters:
    -----------
    ticker : str
        股票代码，格式为 "0005.HK"
    start_date : str
        开始日期，格式 "YYYY-MM-DD"
    end_date : str
        结束日期，格式 "YYYY-MM-DD"
    
    Returns:
    --------
    pd.DataFrame : 包含历史数据的DataFrame
    """
    print(f"正在下载 {ticker} 的历史数据...")
    print(f"日期范围: {start_date} 至 {end_date}")
    
    try:
        # 下载数据
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # 检查数据是否为空
        if data.empty:
            print(f"警告: 未获取到 {ticker} 的数据，请检查股票代码和日期范围")
            return None
        
        # 处理 MultiIndex 列（如果有多个股票代码）
        if isinstance(data.columns, pd.MultiIndex):
            # 取第一层作为列名，如果有第二层（股票代码），忽略它
            data.columns = data.columns.get_level_values(0)
        
        # 重置索引，使日期成为列
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            # 确保日期列名正确
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'date'}, inplace=True)
        
        # 重命名列名（统一格式，转为小写）
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        # 重命名列
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)
        
        # 确保日期列为字符串格式（如果是datetime，转换为字符串）
        if 'date' in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        
        print(f"成功下载 {len(data)} 条数据记录")
        print(f"数据列: {list(data.columns)}")
        print(f"\n前5条数据预览:")
        print(data.head())
        
        return data
    
    except Exception as e:
        print(f"下载数据时发生错误: {str(e)}")
        return None


def save_to_csv(data, filename="0005.HK.csv"):
    """
    保存数据到CSV文件
    按照 ALGOGENE 要求的列顺序：date, open, high, low, close, volume
    
    Parameters:
    -----------
    data : pd.DataFrame
        要保存的数据
    filename : str
        输出文件名
    """
    if data is not None:
        # 确保列顺序：date, open, high, low, close, volume
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        # 只保留需要的列，按顺序排列
        data_to_save = data[required_columns].copy()
        
        data_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {filename}")
        print(f"文件大小: {os.path.getsize(filename) / 1024:.2f} KB")
        print(f"列顺序: {list(data_to_save.columns)}")
    else:
        print("无法保存数据：数据为空")


def create_meta_json(ticker="0005.HK", start_date="2020-01-01", end_date="2020-12-31", 
                     filename="_meta_.json", contract_size=1):
    """
    创建 ALGOGENE 格式的元数据文件
    按照 ALGOGENE 要求：第一层key是股票代码，第二层包含所有配置
    
    Parameters:
    -----------
    ticker : str
        股票代码（必须与 ALGOGENE 现有代码不同）
    start_date : str
        开始日期，格式 YYYY-MM-DD
    end_date : str
        结束日期，格式 YYYY-MM-DD
    filename : str
        输出文件名
    contract_size : int
        每手合约的股数
    """
    # ALGOGENE 格式：第一层key是股票代码
    meta = {
        ticker: {
            "file": f"{ticker}.csv",
            "file_delimiter": ",",
            "period_start": start_date,
            "period_end": end_date,
            "settleCurrency": "HKD",
            "contractSize": contract_size,
            "fmt_time": "%Y-%m-%d",  # 日期格式：YYYY-MM-DD
            "col_time": 0,      # date列位置（第一列）
            "col_open": 1,      # open列位置
            "col_high": 2,      # high列位置
            "col_low": 3,       # low列位置
            "col_close": 4,     # close列位置
            "col_volume": 5     # volume列位置
        }
    }
    
    # 保存JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n元数据文件已创建: {filename}")
    print("\n元数据内容预览:")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 配置参数
    TICKER = "0939.HK"
    START_DATE = "2020-01-01"
    END_DATE = "2020-12-31"
    
    # 下载数据
    data = download_hk_stock_data(TICKER, START_DATE, END_DATE)
    
    # 保存为CSV
    if data is not None:
        csv_filename = f"{TICKER}.csv"
        save_to_csv(data, csv_filename)
        
        # 创建元数据文件
        create_meta_json(TICKER, START_DATE, END_DATE)
        
        print("\n" + "="*50)
        print("任务完成！")
        print(f"✓ CSV 文件: {csv_filename}")
        print(f"✓ 元数据文件: _meta_.json")
        print("="*50)
    else:
        print("\n下载失败，请检查网络连接和参数设置")

