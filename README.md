# 配对交易策略 (Pairs Trading)

## 1. 安装依赖

```bash
pip install yfinance pandas numpy statsmodels
```

## 2. 策略逻辑

**统计套利策略（Pairs Trading）**

- 使用两个标的进行配对交易（Y 和 X）
- 每日收集最近 5 个交易日的收盘价
- 使用 OLS 线性回归拟合关系：`Y = b × X`（无截距）
- 计算当前残差：`diff = Y当前价 - b × X当前价`
- 交易信号：
  - `diff > 0.1×MSE`：做空 Y，做多 X（数量 = b）
  - `diff < -0.1×MSE`：做多 Y，做空 X（数量 = b）
- 持仓时间：5 个交易日

## 3. 错误处理

### IndexError: list index out of range

**错误位置：**
```python
self.myinstrument_X = mEvt['subscribeList'][1]  # 索引 1 不存在
```

**原因：** 策略需要两个标的，但平台配置中只订阅了一个。

**解决方案：** 在 ALGOGENE 平台的 Settings 中，确保订阅列表包含两个标的：
- 第一个标的（Y）：`0005.HK`
- 第二个标的（X）：`0939.HK`

## 4. 数据文件

- `0005.HK.csv`：汇丰控股历史数据
- `0939.HK.csv`：建设银行历史数据
- `_meta_.json`：ALGOGENE 元数据配置
- `download_data.py`：数据下载脚本

