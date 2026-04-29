import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# 获取数据
df_raw = yf.download("GLD", start="2020-01-01", end="2024-01-01", progress=False, auto_adjust=True)
close = df_raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
close = close.dropna()

def backtest(close, short, long, rsi_buy=70, rsi_sell=75):
    df = pd.DataFrame({"Close": close})
    df["SMA_short"] = SMAIndicator(close=df["Close"], window=short).sma_indicator()
    df["SMA_long"]  = SMAIndicator(close=df["Close"], window=long).sma_indicator()
    df["RSI"]       = RSIIndicator(close=df["Close"], window=14).rsi()
    df.dropna(inplace=True)

    df["Signal"] = 0
    df.loc[(df["SMA_short"] > df["SMA_long"]) & (df["RSI"] < rsi_buy),  "Signal"] = 1
    df.loc[(df["SMA_short"] < df["SMA_long"]) | (df["RSI"] > rsi_sell), "Signal"] = 0

    df["Return"]   = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Return"]

    cum = (1 + df["Strategy"]).cumprod()
    total_return = cum.iloc[-1] - 1
    sharpe = df["Strategy"].mean() / df["Strategy"].std() * np.sqrt(252) if df["Strategy"].std() > 0 else 0
    rolling_max  = cum.cummax()
    max_drawdown = ((cum - rolling_max) / rolling_max).min()

    return total_return, sharpe, max_drawdown

# 参数网格搜索
short_range = range(5, 40, 5)
long_range  = range(20, 120, 10)

results = []
total = len(short_range) * len(long_range)
count = 0

print(f"开始优化，共测试 {total} 种参数组合...")

for short in short_range:
    for long in long_range:
        if short >= long:
            continue
        ret, sharpe, dd = backtest(close, short, long)
        results.append({
            "short": short, "long": long,
            "return": ret, "sharpe": sharpe, "max_drawdown": dd
        })
        count += 1
        if count % 20 == 0:
            print(f"进度: {count}/{total}")

results_df = pd.DataFrame(results)

# 按夏普比率排序
top10 = results_df.sort_values("sharpe", ascending=False).head(10)

print("\n========== Top 10 参数组合（按夏普比率）==========")
print(f"{'短期均线':>8} {'长期均线':>8} {'总收益':>10} {'夏普比率':>10} {'最大回撤':>10}")
print("-" * 52)
for _, row in top10.iterrows():
    print(f"{int(row['short']):>8} {int(row['long']):>8} {row['return']:>9.2%} {row['sharpe']:>10.2f} {row['max_drawdown']:>9.2%}")

best = top10.iloc[0]
print(f"\n最优参数: 短期={int(best['short'])}日, 长期={int(best['long'])}日")
print(f"最优夏普: {best['sharpe']:.2f}, 收益: {best['return']:.2%}, 最大回撤: {best['max_drawdown']:.2%}")

# 热力图可视化
pivot_sharpe = results_df.pivot(index="short", columns="long", values="sharpe")
plt.figure(figsize=(12, 6))
plt.imshow(pivot_sharpe, aspect="auto", cmap="RdYlGn")
plt.colorbar(label="夏普比率")
plt.xticks(range(len(pivot_sharpe.columns)), pivot_sharpe.columns)
plt.yticks(range(len(pivot_sharpe.index)), pivot_sharpe.index)
plt.xlabel("长期均线")
plt.ylabel("短期均线")
plt.title("参数优化热力图（夏普比率）")
plt.tight_layout()
plt.savefig("optimization_heatmap.png", dpi=150)
plt.show()
print("\n热力图已保存至 optimization_heatmap.png")
