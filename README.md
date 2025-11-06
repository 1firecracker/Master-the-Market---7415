# COMP7415 量化交易策略项目

多因子模型股票交易策略，基于ALGOGENE平台开发。

## 项目结构

```
Algo/
├── baseline.py              # 单因子均值回归策略（baseline）
├── algo_sample.py           # 配对交易策略示例
├── download_data.py         # 数据下载脚本
├── local run/               # 本地训练和测试
│   ├── step1.py            # 数据探索与因子计算
│   ├── step1_4_factor_design.py  # 因子IC分析与选择
│   ├── train_model.py      # 模型训练脚本
│   └── results/            # 训练结果记录
└── knowledge/              # 项目文档
    └── 项目实施计划.md      # 详细实施计划
```

## 快速开始

### 1. 数据探索
```bash
cd "local run"
python step1.py
```

### 2. 因子分析
```bash
python step1_4_factor_design.py
```

### 3. 模型训练
```bash
python train_model.py
```

## 策略说明

### Baseline策略
- **类型**: 单因子均值回归
- **标的**: 单只股票
- **逻辑**: 价格偏离均值时反向交易

### 多因子模型策略
- **类型**: 多因子预测模型
- **因子**: 价格因子（6个）+ 经济因子（2个）
- **模型**: XGBoost / RandomForest / LinearRegression
- **预测目标**: 未来5天收益率

## 依赖

- Python 3.x
- pandas, numpy, scikit-learn
- xgboost
- matplotlib, seaborn
- requests

## 配置

在 `local run/train_model.py` 中修改配置：
- 股票代码
- 模型类型
- 数据时间范围

## 结果

训练结果记录在 `local run/results/record.md`
