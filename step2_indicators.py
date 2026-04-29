import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# 获取黄金ETF数据
df = yf.download("GLD", start="2020-01-01", end="2024-01-01", progress=False, auto_adjust=True)
close = df["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
df = pd.DataFrame({"Close": close})

# 均线
df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
df["SMA_60"] = SMAIndicator(close=df["Close"], window=60).sma_indicator()
df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()

# RSI（相对强弱指数）
df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

# MACD
macd = MACD(close=df["Close"])
df["MACD"]        = macd.macd()
df["MACD_Signal"] = macd.macd_signal()
df["MACD_Hist"]   = macd.macd_diff()

# 布林带
bb = BollingerBands(close=df["Close"], window=20)
df["BB_Upper"] = bb.bollinger_hband()
df["BB_Middle"]= bb.bollinger_mavg()
df["BB_Lower"] = bb.bollinger_lband()

df.dropna(inplace=True)

print("技术指标添加完成，字段列表：")
print(df.columns.tolist())
print("\n最新数据：")
print(df.tail(3).to_string())
