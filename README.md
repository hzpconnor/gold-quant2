# Gold Quant — 黄金量化交易系统

基于 Python 的完整量化交易流程，以黄金 ETF（GLD）为标的，涵盖数据获取、技术指标、策略回测、参数优化、风险控制、实盘模拟六个环节。

---

## 项目结构

```
gold-quant/
├── step1_data.py          # 数据获取
├── step2_indicators.py    # 技术指标计算
├── step3_backtest.py      # 策略回测
├── step4_optimize.py      # 参数网格搜索优化
├── step5_risk.py          # 风险控制回测
├── step6_live.py          # 实盘模拟（Alpaca Paper Trading）
├── backtest_result.png    # 回测收益图表
├── optimization_heatmap.png  # 参数优化热力图
└── risk_result.png        # 风控策略图表
```

---

## 依赖安装

```bash
pip install yfinance pandas numpy ta matplotlib
# 实盘模拟额外需要
pip install alpaca-trade-api
```

---

## 运行顺序

### Step 1 — 数据获取 (`step1_data.py`)

从 Yahoo Finance 下载 2020-01-01 至 2024-01-01 的历史数据：

| 代码 | 资产 |
|------|------|
| GLD | 黄金 ETF |
| GC=F | 黄金期货 |
| SLV | 白银 ETF |
| DX-Y.NYB | 美元指数（负相关参考） |

输出：合并价格表及基本统计。

```bash
python step1_data.py
```

---

### Step 2 — 技术指标 (`step2_indicators.py`)

对 GLD 收盘价计算以下指标：

| 指标 | 参数 |
|------|------|
| SMA | 20日、60日简单移动平均 |
| EMA | 20日指数移动平均 |
| RSI | 14日相对强弱指数 |
| MACD | 默认参数（12/26/9） |
| 布林带 | 20日，±2σ |

```bash
python step2_indicators.py
```

---

### Step 3 — 策略回测 (`step3_backtest.py`)

**信号逻辑（多条件组合，只做多）：**

- **买入**：SMA20 > SMA60（均线多头）AND RSI < 70 AND MACD 金叉
- **卖出**：SMA20 < SMA60 OR RSI > 75 OR MACD 死叉

**输出指标：**

- 总收益率 / 市场收益率
- 年化收益率
- 夏普比率（年化）
- 最大回撤
- 胜率

**输出文件：** `backtest_result.png`（收益对比 / RSI / 回撤三图）

```bash
python step3_backtest.py
```

---

### Step 4 — 参数优化 (`step4_optimize.py`)

对短期均线（5–35，步长5）和长期均线（20–110，步长10）进行网格搜索，约 56 种有效组合，按夏普比率排序输出 Top 10。

**输出文件：** `optimization_heatmap.png`（夏普比率热力图）

```bash
python step4_optimize.py
```

---

### Step 5 — 风险控制 (`step5_risk.py`)

在基础策略之上叠加三层风控：

| 机制 | 参数 |
|------|------|
| 止损 | 单笔亏损 ≥ 5% 平仓 |
| 止盈 | 单笔盈利 ≥ 10% 平仓 |
| 组合最大回撤保护 | 净值回撤 ≥ 15% 强制全平 |

另外使用**波动率倒数**动态调整仓位（目标日波动率 1%，波动大时自动降仓）。

**输出文件：** `risk_result.png`

```bash
python step5_risk.py
```

---

### Step 6 — 实盘模拟 (`step6_live.py`)

```bash
pip install yfinance pandas numpy ta matplotlib
# 实盘模拟额外需要
pip install alpaca-trade-api
pip install pytz
```

对接 **Alpaca Paper Trading**（免费，无需真实资金），每 60 秒循环检查：

1. 获取近 60 日历史 K 线
2. 计算 SMA30 / SMA40 / RSI 信号
3. 按信号 + 止损/止盈规则执行买卖
4. 打印账户净值和持仓状态

**使用前配置：**

```python
# step6_live.py 顶部配置区
API_KEY    = "YOUR_ALPACA_API_KEY"
API_SECRET = "YOUR_ALPACA_SECRET_KEY"
```

在 [alpaca.markets](https://alpaca.markets) 免费注册，选择 **Paper Trading** 模式获取密钥。

```bash
python step6_live.py
```

---

## 策略核心逻辑总结

```
信号 = 均线趋势（SMA 金/死叉）
     + 动量确认（RSI 超买/超卖）
     + 趋势加速（MACD 金/死叉）

仓位 = min(目标波动率 / 当前波动率, 1.0)   # 动态仓位

风控 = 止损 5% | 止盈 10% | 组合回撤保护 15%
```

---

## 注意事项

- 历史回测结果不代表未来收益，策略存在过拟合风险。
- Step 4 参数优化在训练集上择优，实盘需结合样本外验证。
- Step 6 使用 Paper Trading，不涉及真实资金。
- 数据来源为 Yahoo Finance，网络不稳定时下载可能失败。
