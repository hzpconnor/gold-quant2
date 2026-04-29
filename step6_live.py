"""
实盘模拟对接 — 使用 Alpaca Paper Trading API
免费注册：https://alpaca.markets（选 Paper Trading，无需真实资金）

安装依赖：
    pip install alpaca-trade-api pandas ta pytz

优化点：
  1. 信号逻辑与 step3 回测一致（SMA20/SMA60/MACD/RSI）
  2. 市场时间判断，只在 NYSE 开市期间运行
  3. 收盘后单次触发，不再每60秒盲目轮询
  4. 波动率动态仓位（step5 逻辑）
  5. 三层风控：止损5% / 止盈10% / 组合回撤15%
  6. 日志持久化写入 live_trading.log
  7. 重复下单保护（检查 pending 订单）
  8. 分类错误处理（API / 数据 / 订单）
"""

import time
import logging
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import pytz

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

# ─── 配置区 ────────────────────────────────────────────
API_KEY       = "PKBNJSZCA5OSYU76TM6FJSLS5J"
API_SECRET    = "4E8kqgmJKY9gauEhMygBa3f8J54wvsspFrzWYZt19kxu"
BASE_URL      = "https://paper-api.alpaca.markets"

SYMBOL        = "GLD"
STOP_LOSS     = 0.05    # 单笔止损 5%
TAKE_PROFIT   = 0.10    # 单笔止盈 10%
MAX_DRAWDOWN  = 0.15    # 组合最大回撤 15%，触发全平
TARGET_VOL    = 0.01    # 目标日波动率 1%（动态仓位基准）
HISTORY_DAYS  = 90      # 拉取近90日日线（保证 SMA60 有足够数据）
# ────────────────────────────────────────────────────────

ET = pytz.timezone("America/New_York")
MARKET_OPEN  = {"hour": 9,  "minute": 30}
MARKET_CLOSE = {"hour": 16, "minute": 0}

# ─── 日志配置 ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── API 初始化 ──────────────────────────────────────────
def get_api():
    try:
        import alpaca_trade_api as tradeapi
        return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
    except ImportError:
        log.error("请先安装: pip install alpaca-trade-api")
        raise


# ─── 市场时间工具 ─────────────────────────────────────────
def now_et() -> datetime:
    return datetime.now(ET)


def is_market_open(api) -> bool:
    """通过 Alpaca clock 接口判断当前是否处于 NYSE 正市交易时段。"""
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        log.warning(f"获取市场时钟失败，默认视为关闭: {e}")
        return False


def seconds_until_next_close(api) -> float:
    """返回距下次收盘还有多少秒（用于收盘后触发逻辑）。"""
    try:
        clock = api.get_clock()
        next_close = clock.next_close.replace(tzinfo=timezone.utc).astimezone(ET)
        now = now_et()
        return max((next_close - now).total_seconds(), 0)
    except Exception as e:
        log.warning(f"获取收盘时间失败: {e}")
        # 兜底：等到今天 ET 16:05
        now = now_et()
        target = now.replace(hour=16, minute=5, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return (target - now).total_seconds()


def seconds_until_next_open(api) -> float:
    """返回距下次开盘还有多少秒。"""
    try:
        clock = api.get_clock()
        next_open = clock.next_open.replace(tzinfo=timezone.utc).astimezone(ET)
        return max((next_open - now_et()).total_seconds(), 0)
    except Exception as e:
        log.warning(f"获取开盘时间失败: {e}")
        return 3600.0


# ─── 数据获取 ─────────────────────────────────────────────
def get_historical_data(api, symbol: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """拉取近 N 日日线收盘价，返回 DataFrame[Close]。"""
    end   = datetime.now()
    start = end - timedelta(days=days)
    try:
        bars = api.get_bars(
            symbol,
            "1Day",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            feed="iex",
        ).df
        if bars.empty:
            raise ValueError(f"获取到的 {symbol} 数据为空")
        return bars[["close"]].rename(columns={"close": "Close"})
    except Exception as e:
        log.error(f"数据获取失败 [{symbol}]: {e}")
        raise


# ─── 技术指标 & 信号（与 step3 一致）────────────────────────
def compute_signal(df: pd.DataFrame) -> tuple[str, pd.Series, float]:
    """
    信号逻辑（同 step3_backtest.py）：
      买入：SMA20 > SMA60 AND RSI < 70 AND MACD > MACD_Signal
      卖出：SMA20 < SMA60 OR RSI > 75 OR MACD < MACD_Signal
    另外计算 20日波动率用于动态仓位。
    """
    df = df.copy()
    df["SMA_20"]      = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_60"]      = SMAIndicator(close=df["Close"], window=60).sma_indicator()
    df["RSI"]         = RSIIndicator(close=df["Close"], window=14).rsi()
    macd              = MACD(close=df["Close"])
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["Return"]      = df["Close"].pct_change()
    df["Volatility"]  = df["Return"].rolling(20).std()
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("计算指标后数据为空，历史数据不足")

    last = df.iloc[-1]
    buy  = (last["SMA_20"] > last["SMA_60"]
            and last["RSI"] < 70
            and last["MACD"] > last["MACD_Signal"])
    sell = (last["SMA_20"] < last["SMA_60"]
            or last["RSI"] > 75
            or last["MACD"] < last["MACD_Signal"])

    signal = "BUY" if buy else ("SELL" if sell else "HOLD")
    vol    = float(last["Volatility"]) if last["Volatility"] > 0 else TARGET_VOL
    return signal, last, vol


# ─── 动态仓位计算（step5 逻辑）──────────────────────────────
def calc_position_size(api, current_vol: float) -> int:
    """根据波动率和账户净值计算本次应买入股数。"""
    position_ratio = min(TARGET_VOL / current_vol, 1.0)
    try:
        acct  = api.get_account()
        equity = float(acct.equity)
    except Exception as e:
        log.warning(f"获取账户净值失败，默认买1股: {e}")
        return 1

    try:
        bars  = api.get_bars(SYMBOL, "1Day", limit=1, feed="iex").df
        price = float(bars["close"].iloc[-1])
    except Exception as e:
        log.warning(f"获取最新价格失败，默认买1股: {e}")
        return 1

    if price <= 0:
        return 1
    qty = int((equity * position_ratio) / price)
    return max(qty, 1)


# ─── 账户 & 持仓查询 ──────────────────────────────────────
def get_position(api, symbol: str) -> tuple[float, float]:
    try:
        pos = api.get_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except Exception:
        return 0.0, 0.0


def get_account_equity(api) -> tuple[float, float]:
    try:
        acct = api.get_account()
        return float(acct.equity), float(acct.last_equity)
    except Exception as e:
        log.error(f"获取账户信息失败: {e}")
        raise


def has_pending_order(api, symbol: str) -> bool:
    """检查是否有未成交的挂单，防止重复下单。"""
    try:
        orders = api.list_orders(status="open", symbols=[symbol])
        if orders:
            log.info(f"  存在 {len(orders)} 笔未成交订单，跳过本次下单")
            return True
        return False
    except Exception as e:
        log.warning(f"查询挂单失败: {e}")
        return False


# ─── 下单 ────────────────────────────────────────────────
def place_order(api, symbol: str, qty: int, side: str):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )
        log.info(f"  下单成功: {side.upper()} {qty}股 {symbol} | 订单ID: {order.id}")
        return order
    except Exception as e:
        log.error(f"  下单失败 [{side} {qty}股 {symbol}]: {e}")
        raise


# ─── 单次信号检查 & 执行 ───────────────────────────────────
def run_once(api, peak_equity: float) -> float:
    """
    执行一次完整的信号检查和交易逻辑。
    peak_equity: 历史最高账户净值，用于组合回撤保护。
    返回更新后的 peak_equity。
    """
    log.info("=" * 55)
    log.info(f"信号检查 @ {now_et():%Y-%m-%d %H:%M:%S ET}")

    # 账户状态
    equity, last_equity = get_account_equity(api)
    peak_equity = max(peak_equity, equity)
    portfolio_dd = (equity - peak_equity) / peak_equity
    log.info(f"  账户净值: ${equity:,.2f} | 今日盈亏: ${equity - last_equity:+,.2f} | "
             f"组合回撤: {portfolio_dd:.2%}")

    # 组合最大回撤保护（先于信号判断）
    qty, entry = get_position(api, SYMBOL)
    if qty > 0 and portfolio_dd <= -MAX_DRAWDOWN:
        log.warning(f"  组合回撤 {portfolio_dd:.2%} 超过 {-MAX_DRAWDOWN:.0%}，强制全平")
        if not has_pending_order(api, SYMBOL):
            place_order(api, SYMBOL, int(qty), "sell")
        return peak_equity

    # 获取数据 & 计算信号
    df = get_historical_data(api, SYMBOL)
    signal, last_row, vol = compute_signal(df)

    log.info(
        f"  收盘价: ${last_row['Close']:.2f} | "
        f"SMA20: {last_row['SMA_20']:.2f} | SMA60: {last_row['SMA_60']:.2f} | "
        f"RSI: {last_row['RSI']:.1f} | MACD差: {last_row['MACD'] - last_row['MACD_Signal']:.4f} | "
        f"波动率: {vol:.4f}"
    )
    log.info(f"  策略信号: {signal}")

    qty, entry = get_position(api, SYMBOL)
    log.info(f"  当前持仓: {qty:.0f}股 | 成本价: ${entry:.2f}")

    # 持仓中：检查止损/止盈/信号平仓
    if qty > 0:
        pnl_pct = (last_row["Close"] - entry) / entry
        log.info(f"  持仓盈亏: {pnl_pct:+.2%}")

        if pnl_pct <= -STOP_LOSS:
            log.info("  触发止损，平仓")
            if not has_pending_order(api, SYMBOL):
                place_order(api, SYMBOL, int(qty), "sell")
        elif pnl_pct >= TAKE_PROFIT:
            log.info("  触发止盈，平仓")
            if not has_pending_order(api, SYMBOL):
                place_order(api, SYMBOL, int(qty), "sell")
        elif signal == "SELL":
            log.info("  信号卖出，平仓")
            if not has_pending_order(api, SYMBOL):
                place_order(api, SYMBOL, int(qty), "sell")
        else:
            log.info("  持仓中，无操作")

    # 空仓：信号买入
    elif signal == "BUY":
        if has_pending_order(api, SYMBOL):
            return peak_equity
        trade_qty = calc_position_size(api, vol)
        log.info(f"  信号买入 {trade_qty}股（动态仓位，波动率={vol:.4f}）")
        place_order(api, SYMBOL, trade_qty, "buy")

    else:
        log.info("  空仓等待信号")

    return peak_equity


# ─── 主循环 ──────────────────────────────────────────────
def run_bot():
    api = get_api()

    log.info("=" * 55)
    log.info("   黄金量化交易机器人 v2 (Paper Trading)")
    log.info("=" * 55)

    # 验证 API 连接
    try:
        api.get_account()
        log.info("API 连接成功")
    except Exception as e:
        log.error(f"API 连接失败: {e}")
        return

    # 初始化历史最高净值（用于组合回撤保护）
    try:
        equity, _ = get_account_equity(api)
        peak_equity = equity
    except Exception:
        peak_equity = 0.0

    while True:
        try:
            if not is_market_open(api):
                # 等到下次开盘
                wait = seconds_until_next_open(api)
                log.info(f"市场已关闭，等待 {wait/3600:.1f} 小时至下次开盘...")
                time.sleep(min(wait, 1800))  # 最多睡30分钟，防止时钟漂移
                continue

            # 市场开着：等到收盘后再执行信号（日线策略）
            secs_to_close = seconds_until_next_close(api)
            if secs_to_close > 120:
                log.info(f"市场开盘中，距收盘还有 {secs_to_close/3600:.2f}h，等待收盘后执行...")
                # 每30分钟唤醒一次，重新判断（应对提前收盘等特殊情况）
                time.sleep(min(secs_to_close - 60, 1800))
                continue

            # 收盘后执行一次信号检查
            log.info("接近/达到收盘时间，执行今日信号检查...")
            peak_equity = run_once(api, peak_equity)

            # 等到下次开盘再循环
            wait = seconds_until_next_open(api)
            log.info(f"今日执行完毕，等待 {wait/3600:.1f}h 至下次开盘。")
            time.sleep(min(wait, 1800))

        except KeyboardInterrupt:
            log.info("用户中断，程序退出。")
            break
        except Exception:
            log.error(f"主循环发生未预期错误:\n{traceback.format_exc()}")
            log.info("60秒后重试...")
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
