import yfinance as yf
yf.set_tz_cache_location(".yf_cache")
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# 获取数据
df = yf.download("GLD", start="2020-01-01", end="2024-01-01", progress=False, auto_adjust=True)
close = df["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
df = pd.DataFrame({"Close": close})

# 技术指标
df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
df["SMA_60"] = SMAIndicator(close=df["Close"], window=60).sma_indicator()
df["RSI"]    = RSIIndicator(close=df["Close"], window=14).rsi()
macd         = MACD(close=df["Close"])
df["MACD"]   = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
bb = BollingerBands(close=df["Close"], window=20)
df["BB_Upper"] = bb.bollinger_hband()
df["BB_Lower"] = bb.bollinger_lband()
df.dropna(inplace=True)

# 策略信号（多条件组合）
def generate_signal(row):
    buy  = row["SMA_20"] > row["SMA_60"]  # 均线多头
    buy &= row["RSI"] < 70                # RSI未超买
    buy &= row["MACD"] > row["MACD_Signal"]  # MACD金叉

    sell  = row["SMA_20"] < row["SMA_60"]  # 均线空头
    sell |= row["RSI"] > 75               # RSI超买离场
    sell |= row["MACD"] < row["MACD_Signal"]  # MACD死叉

    if buy:
        return 1
    elif sell:
        return -1
    return 0

df["Signal"] = df.apply(generate_signal, axis=1)
# 将0替换为NaN以便向前填充，然后将-1（卖出信号）替换为0（空仓）
df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)

# 计算收益
df["Return"]   = df["Close"].pct_change()
df["Strategy"] = df["Position"].shift(1) * df["Return"]
df["Cumulative_Market"]   = (1 + df["Return"]).cumprod()
df["Cumulative_Strategy"] = (1 + df["Strategy"]).cumprod()

# 绩效指标
total_return   = df["Cumulative_Strategy"].iloc[-1] - 1
market_return  = df["Cumulative_Market"].iloc[-1] - 1
annual_return  = (1 + total_return) ** (1/4) - 1
sharpe         = df["Strategy"].mean() / df["Strategy"].std() * np.sqrt(252)

# 最大回撤
rolling_max    = df["Cumulative_Strategy"].cummax()
drawdown       = (df["Cumulative_Strategy"] - rolling_max) / rolling_max
max_drawdown   = drawdown.min()

# 胜率
wins  = (df["Strategy"] > 0).sum()
total = (df["Strategy"] != 0).sum()
win_rate = wins / total if total > 0 else 0

print("=" * 40)
print("       策略绩效报告")
print("=" * 40)
print(f"策略总收益率:   {total_return:.2%}")
print(f"市场总收益率:   {market_return:.2%}")
print(f"年化收益率:     {annual_return:.2%}")
print(f"夏普比率:       {sharpe:.2f}")
print(f"最大回撤:       {max_drawdown:.2%}")
print(f"胜率:           {win_rate:.2%}")
print("=" * 40)

# 画图
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 图1：收益对比
axes[0].plot(df["Cumulative_Market"],   label="买入持有", linewidth=1.5)
axes[0].plot(df["Cumulative_Strategy"], label="组合策略", linewidth=1.5)
axes[0].set_title("策略收益 vs 买入持有")
axes[0].legend()
axes[0].grid(True)

# 图2：RSI
axes[1].plot(df["RSI"], color="purple", linewidth=1)
axes[1].axhline(70, color="red",   linestyle="--", alpha=0.7, label="超买70")
axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, label="超卖30")
axes[1].set_title("RSI指标")
axes[1].legend()
axes[1].grid(True)

# 图3：回撤
axes[2].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
axes[2].set_title("策略回撤")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("backtest_result.png", dpi=150)
plt.show()
print("\n图表已保存至 backtest_result.png")
