import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# 获取数据
df_raw = yf.download("GLD", start="2020-01-01", end="2024-01-01", progress=False, auto_adjust=True)
close = df_raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

df = pd.DataFrame({"Close": close.dropna()})
df["SMA_30"] = SMAIndicator(close=df["Close"], window=30).sma_indicator()
df["SMA_40"] = SMAIndicator(close=df["Close"], window=40).sma_indicator()
df["RSI"]    = RSIIndicator(close=df["Close"], window=14).rsi()
df["Return"] = df["Close"].pct_change()
df["Volatility"] = df["Return"].rolling(20).std()  # 20日波动率
df.dropna(inplace=True)

# 风险控制参数
STOP_LOSS     = 0.05   # 单笔最大亏损5%
TAKE_PROFIT   = 0.10   # 单笔止盈10%
MAX_DRAWDOWN  = 0.15   # 组合最大回撤15%，触发全平

def backtest_with_risk(df):
    capital    = 1.0    # 初始资金（归一化）
    position   = 0      # 当前仓位（0或1）
    entry_price= 0
    equity_curve = []
    trades     = []

    peak = 1.0  # 历史最高净值，用于回撤计算

    for i, (date, row) in enumerate(df.iterrows()):
        price = row["Close"]

        # 计算动态仓位（波动率倒数，波动大时减仓）
        vol = row["Volatility"]
        target_vol = 0.01  # 目标日波动率1%
        position_size = min(target_vol / vol, 1.0) if vol > 0 else 1.0

        # 信号
        buy_signal  = row["SMA_30"] > row["SMA_40"] and row["RSI"] < 70
        sell_signal = row["SMA_30"] < row["SMA_40"] or row["RSI"] > 75

        # 计算当前包含浮动盈亏的净值
        current_equity = capital
        if position == 1:
            pnl = (price - entry_price) / entry_price
            current_equity = capital * (1 + pnl * position_size)

        # 持仓中检查止损/止盈
        if position == 1:
            if pnl <= -STOP_LOSS:
                position = 0
                capital = current_equity
                trades.append({"date": date, "type": "止损", "pnl": pnl})
            elif pnl >= TAKE_PROFIT:
                position = 0
                capital = current_equity
                trades.append({"date": date, "type": "止盈", "pnl": pnl})
            elif sell_signal:
                position = 0
                capital = current_equity
                trades.append({"date": date, "type": "信号平仓", "pnl": pnl})

        # 组合最大回撤保护
        peak = max(peak, current_equity)
        portfolio_dd = (current_equity - peak) / peak
        if portfolio_dd <= -MAX_DRAWDOWN and position == 1:
            position = 0
            capital = current_equity
            trades.append({"date": date, "type": "回撤保护", "pnl": pnl})

        # 开仓
        if position == 0 and buy_signal:
            position   = 1
            entry_price= price

        equity_curve.append(current_equity if position == 1 else capital)

    return pd.Series(equity_curve, index=df.index), pd.DataFrame(trades)

equity, trades = backtest_with_risk(df)

# 基准：买入持有
benchmark = (1 + df["Return"]).cumprod()

# 绩效统计
total_return  = equity.iloc[-1] - 1
market_return = benchmark.iloc[-1] - 1
sharpe        = df["Return"].mean() / df["Return"].std() * np.sqrt(252)
rolling_max   = equity.cummax()
drawdown      = (equity - rolling_max) / rolling_max
max_dd        = drawdown.min()

print("=" * 45)
print("       风险控制策略绩效报告")
print("=" * 45)
print(f"策略总收益率:   {total_return:.2%}")
print(f"市场总收益率:   {market_return:.2%}")
print(f"最大回撤:       {max_dd:.2%}")
print(f"总交易次数:     {len(trades)}")
if len(trades) > 0:
    win_trades = trades[trades["pnl"] > 0]
    print(f"胜率:           {len(win_trades)/len(trades):.2%}")
    print(f"\n交易类型分布:")
    print(trades["type"].value_counts().to_string())
print("=" * 45)

# 画图
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(benchmark, label="买入持有", linewidth=1.5, alpha=0.7)
axes[0].plot(equity,    label="风控策略", linewidth=1.5)
axes[0].set_title("风控策略 vs 买入持有")
axes[0].legend()
axes[0].grid(True)

axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
axes[1].axhline(-MAX_DRAWDOWN, color="darkred", linestyle="--", label=f"最大回撤阈值 {-MAX_DRAWDOWN:.0%}")
axes[1].set_title("策略回撤")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("risk_result.png", dpi=150)
plt.show()
print("\n图表已保存至 risk_result.png")
