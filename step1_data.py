import yfinance as yf
import pandas as pd

# 获取多个黄金相关资产
tickers = {
    "GLD":  "黄金ETF",
    "GC=F": "黄金期货",
    "SLV":  "白银ETF",
    "DX-Y.NYB": "美元指数",  # 黄金负相关
}

data = {}
for ticker, name in tickers.items():
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01", progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    data[ticker] = close
    print(f"{name} ({ticker}): {len(df)} 条数据")

# 合并成一张表
prices = pd.DataFrame(data)
prices.dropna(inplace=True)

print("\n数据预览：")
print(prices.tail())

print("\n基本统计：")
print(prices.describe())
