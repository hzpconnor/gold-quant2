"""
黄金量化交易机器人 v3 — 1秒周期实盘模拟（多空双向）
使用 Alpaca Paper Trading API (WebSocket 实时流)

架构改造要点：
  1. 周期：日线 → 1秒（每秒评估一次信号）
  2. 数据：REST历史日线 → WebSocket实时成交流 + 1秒滚动窗口
  3. 指标：EMA20s / EMA60s / RSI14s / MACD(12,26,9)s（适配秒级）
  4. 循环：收盘单次触发 → 市场开盘期间每秒触发
  5. 风控：止损0.5% / 止盈1.0% / 组合回撤15% / 下单冷却10s
  6. 方向：多空双向，BUY 信号开多，SELL 信号开空

安装依赖：
    pip install alpaca-trade-api pandas ta numpy pytz
"""

import collections
import logging
import queue
import threading
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

# ─── 配置区 ────────────────────────────────────────────────
API_KEY       = "PKBNJSZCA5OSYU76TM6FJSLS5J"
API_SECRET    = "4E8kqgmJKY9gauEhMygBa3f8J54wvsspFrzWYZt19kxu"
BASE_URL      = "https://paper-api.alpaca.markets"
DATA_FEED     = "iex"       # iex 免费数据源；sip 需付费订阅

SYMBOL        = "GLD"
INTERVAL_SEC  = 1           # 信号评估周期（秒）
BUFFER_SIZE   = 300         # 滚动窗口：保留最近 300 秒收盘价
MIN_BARS      = 80          # 至少积累 N 秒数据后才开始交易
SEED_BARS     = 200         # 初始化时从历史 1 分钟 bars 预装数量

# 指标窗口（均以秒为单位）
EMA_SHORT     = 20
EMA_LONG      = 60
RSI_WINDOW    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIG      = 9

# 风控参数（日内交易，止损更小）
STOP_LOSS     = 0.005   # 单笔止损 0.5%
TAKE_PROFIT   = 0.010   # 单笔止盈 1.0%
MAX_DRAWDOWN  = 0.15    # 组合最大回撤 15%，触发全平
TARGET_VOL    = 0.0001  # 目标秒级波动率（动态仓位基准）
ORDER_COOLDOWN = 10     # 下单冷却秒数（防抖）
# ──────────────────────────────────────────────────────────

ET = pytz.timezone("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─── API 工具 ─────────────────────────────────────────────

def get_api():
    try:
        import alpaca_trade_api as tradeapi
        return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
    except ImportError:
        log.error("请先安装: pip install alpaca-trade-api")
        raise


def now_et() -> datetime:
    return datetime.now(ET)


def is_market_open(api) -> bool:
    try:
        return api.get_clock().is_open
    except Exception as e:
        log.warning(f"获取市场状态失败: {e}")
        return False


def get_account_equity(api) -> tuple[float, float]:
    acct = api.get_account()
    return float(acct.equity), float(acct.last_equity)


def get_position(api, symbol: str) -> tuple[float, float]:
    try:
        pos = api.get_position(symbol)
        return float(pos.qty), float(pos.avg_entry_price)
    except Exception:
        return 0.0, 0.0


def has_pending_order(api, symbol: str) -> bool:
    try:
        return len(api.list_orders(status="open", symbols=[symbol])) > 0
    except Exception:
        return False


def place_order(api, symbol: str, qty: int, side: str):
    """side='buy' 开多或平空；side='sell' 开空或平多。"""
    try:
        pos_qty = float(api.get_position(symbol).qty)
    except Exception:
        pos_qty = 0.0

    if side == "sell" and pos_qty > 0:
        # 平多：卖出不超过持仓量
        qty = min(qty, int(pos_qty))
    elif side == "buy" and pos_qty < 0:
        # 平空：买回不超过空仓量
        qty = min(qty, int(abs(pos_qty)))

    try:
        order = api.submit_order(
            symbol=symbol, qty=qty, side=side,
            type="market", time_in_force="day",
        )
        log.info(f"下单: {side.upper()} {qty}股 {symbol} | ID: {order.id}")
        return order
    except Exception as e:
        log.error(f"下单失败 [{side} {qty}股 {symbol}]: {e}")
        raise


def calc_position_size(equity: float, last_price: float, current_vol: float) -> int:
    ratio = min(TARGET_VOL / current_vol, 1.0) if current_vol > 0 else 0.1
    try:
        return max(int((equity * ratio) / last_price), 1)
    except Exception:
        return 1


# ─── 指标与信号（秒级） ────────────────────────────────────

def compute_signal(prices: list) -> tuple[str, float, float]:
    """
    基于秒级价格序列计算信号。
    返回: (signal "BUY"/"SELL"/"HOLD", last_price, volatility)
    """
    s = pd.Series(prices, dtype=float)

    ema_short = EMAIndicator(close=s, window=EMA_SHORT).ema_indicator()
    ema_long  = EMAIndicator(close=s, window=EMA_LONG).ema_indicator()
    rsi       = RSIIndicator(close=s, window=RSI_WINDOW).rsi()
    macd_obj  = MACD(close=s, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIG)
    macd_line = macd_obj.macd()
    macd_sig  = macd_obj.macd_signal()

    last_price = s.iloc[-1]
    ret_std    = s.pct_change().rolling(20).std().iloc[-1]
    vol        = float(ret_std) if (ret_std and not np.isnan(ret_std) and ret_std > 0) else TARGET_VOL

    es = float(ema_short.iloc[-1])
    el = float(ema_long.iloc[-1])
    rv = float(rsi.iloc[-1])
    mv = float(macd_line.iloc[-1])
    ms = float(macd_sig.iloc[-1])

    buy  = (es > el and rv < 70 and mv > ms)
    sell = (es < el and rv > 30 and mv < ms)   # 三条件同时成立才开空

    signal = "BUY" if buy else ("SELL" if sell else "HOLD")
    log.info(
        f"价格={last_price:.4f} | EMA{EMA_SHORT}={es:.4f} EMA{EMA_LONG}={el:.4f} "
        f"RSI={rv:.1f} MACD差={mv - ms:.6f} vol={vol:.6f} → {signal}"
    )
    return signal, last_price, vol


# ─── 交易机器人 ───────────────────────────────────────────

class TradingBot:
    def __init__(self):
        self.api = get_api()
        self.price_buffer: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._price_queue: queue.SimpleQueue = queue.SimpleQueue()  # lock-free, async-safe
        self.peak_equity: float = 0.0
        self._cooldown_until: float = 0.0

    # ── 初始化 ──────────────────────────────────────────────

    def seed_buffer(self):
        """用最近 N 根 1 分钟 bar 的收盘价初始化滚动缓冲区。"""
        try:
            bars = self.api.get_bars(
                SYMBOL, "1Min", limit=SEED_BARS, feed=DATA_FEED
            ).df
            if bars.empty:
                log.warning("历史数据为空，等待实时数据积累...")
                return
            for p in bars["close"].values:
                self.price_buffer.append(float(p))
            log.info(f"缓冲区预装完成：{len(self.price_buffer)} 条历史 1 分钟 bar")
        except Exception as e:
            log.warning(f"缓冲区初始化失败: {e}，将依赖实时数据积累")

    # ── WebSocket 数据流（运行于独立线程）──────────────────────

    def _stream_thread(self):
        """在后台线程中运行 Alpaca WebSocket 实时成交流。"""
        try:
            from alpaca_trade_api.stream import Stream
        except ImportError:
            log.error("请安装 alpaca-trade-api: pip install alpaca-trade-api")
            return

        async def on_trade(trade):
            # SimpleQueue.put() never blocks — safe to call from async context
            self._price_queue.put(float(trade.price))

        try:
            stream = Stream(
                API_KEY, API_SECRET,
                base_url=BASE_URL,
                data_feed=DATA_FEED,
            )
            stream.subscribe_trades(on_trade, SYMBOL)
            log.info(f"WebSocket 已连接，订阅 {SYMBOL} 实时成交流...")
            stream.run()
        except Exception:
            log.error(f"WebSocket 流中断:\n{traceback.format_exc()}")

    def start_stream(self):
        t = threading.Thread(target=self._stream_thread, daemon=True, name="ws-stream")
        t.start()
        return t

    # ── 1秒聚合 ─────────────────────────────────────────────

    def _flush_second_bar(self):
        """将当前秒内所有成交价聚合为 1 秒 bar 的收盘价（最后成交价），写入缓冲区。"""
        last = None
        while not self._price_queue.empty():
            last = self._price_queue.get_nowait()
        if last is not None:
            self.price_buffer.append(last)

    # ── 信号评估与交易执行 ─────────────────────────────────────

    def _evaluate(self):
        if len(self.price_buffer) < MIN_BARS:
            log.debug(f"数据积累中 ({len(self.price_buffer)}/{MIN_BARS})...")
            return

        prices = list(self.price_buffer)
        try:
            signal, last_price, vol = compute_signal(prices)
        except Exception as e:
            log.warning(f"指标计算失败: {e}")
            return

        # 账户净值
        try:
            equity, last_equity = get_account_equity(self.api)
        except Exception as e:
            log.error(f"获取账户失败: {e}")
            return
        self.peak_equity = max(self.peak_equity, equity)
        port_dd = (equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        log.info(
            f"净值=${equity:,.2f} 盈亏=${equity - last_equity:+,.2f} "
            f"回撤={port_dd:.3%} 缓冲区={len(self.price_buffer)}条"
        )

        qty, entry = get_position(self.api, SYMBOL)
        pending = has_pending_order(self.api, SYMBOL)  # 只查询一次

        # 组合最大回撤保护：强制全平所有仓位
        if qty != 0 and port_dd <= -MAX_DRAWDOWN:
            log.warning(f"组合回撤 {port_dd:.2%} 超限，强制全平")
            if not pending:
                close_side = "sell" if qty > 0 else "buy"
                place_order(self.api, SYMBOL, int(abs(qty)), close_side)
                self._cooldown_until = time.time() + ORDER_COOLDOWN
            return

        # ── 多头持仓：止损 / 止盈 / 信号反转 ─────────────────────
        if qty > 0 and entry > 0:
            pnl = (last_price - entry) / entry
            log.info(f"多头持仓 {qty:.0f}股 成本=${entry:.4f} 盈亏={pnl:+.3%}")
            reason = (
                f"止损触发 ({pnl:+.3%})，平多" if pnl <= -STOP_LOSS else
                f"止盈触发 ({pnl:+.3%})，平多" if pnl >= TAKE_PROFIT else
                "信号反转，平多" if signal == "SELL" else None
            )
            if reason:
                log.info(reason)
                if not pending:
                    place_order(self.api, SYMBOL, int(qty), "sell")
                    self._cooldown_until = time.time() + ORDER_COOLDOWN
            else:
                log.info("多头持仓中，无操作")

        # ── 空头持仓：止损 / 止盈 / 信号反转 ─────────────────────
        elif qty < 0 and entry > 0:
            pnl = (entry - last_price) / entry   # 价格下跌时盈利为正
            log.info(f"空头持仓 {abs(qty):.0f}股 成本=${entry:.4f} 盈亏={pnl:+.3%}")
            reason = (
                f"止损触发 ({pnl:+.3%})，平空" if pnl <= -STOP_LOSS else
                f"止盈触发 ({pnl:+.3%})，平空" if pnl >= TAKE_PROFIT else
                "信号反转，平空" if signal == "BUY" else None
            )
            if reason:
                log.info(reason)
                if not pending:
                    place_order(self.api, SYMBOL, int(abs(qty)), "buy")
                    self._cooldown_until = time.time() + ORDER_COOLDOWN
            else:
                log.info("空头持仓中，无操作")

        # ── 空仓：根据信号开多或开空 ───────────────────────────────
        else:
            if time.time() < self._cooldown_until:
                log.info(f"下单冷却中，剩余 {self._cooldown_until - time.time():.0f}s")
                return
            if pending:
                log.info("有挂单，等待成交")
                return

            if signal == "BUY":
                trade_qty = calc_position_size(equity, last_price, vol)
                log.info(f"信号买入，开多 {trade_qty}股（vol={vol:.6f}）")
                place_order(self.api, SYMBOL, trade_qty, "buy")
                self._cooldown_until = time.time() + ORDER_COOLDOWN
            elif signal == "SELL":
                trade_qty = calc_position_size(equity, last_price, vol)
                log.info(f"信号卖出，开空 {trade_qty}股（vol={vol:.6f}）")
                place_order(self.api, SYMBOL, trade_qty, "sell")
                self._cooldown_until = time.time() + ORDER_COOLDOWN
            else:
                log.info("空仓等待信号")

    # ── 主循环 ──────────────────────────────────────────────

    def run(self):
        log.info("=" * 55)
        log.info("   黄金量化交易机器人 v3 — 1秒周期 多空双向 (Paper Trading)")
        log.info("=" * 55)

        # API 连通性验证
        try:
            self.api.get_account()
            log.info("API 连接成功")
        except Exception as e:
            log.error(f"API 连接失败: {e}")
            return

        # 初始化峰值净值
        try:
            equity, _ = get_account_equity(self.api)
            self.peak_equity = equity
            log.info(f"初始账户净值: ${equity:,.2f}")
        except Exception as e:
            log.warning(f"获取初始净值失败: {e}")

        # 预热缓冲区（历史 bar）
        self.seed_buffer()

        # 启动 WebSocket 后台流
        self.start_stream()
        log.info(f"交易循环启动，周期={INTERVAL_SEC}秒，等待数据积累...")

        while True:
            loop_start = time.monotonic()
            try:
                self._flush_second_bar()

                if not is_market_open(self.api):
                    log.info("市场已关闭，暂停交易，每60秒检查一次...")
                    time.sleep(60)
                    continue

                self._evaluate()

            except KeyboardInterrupt:
                log.info("用户中断，程序退出。")
                break
            except Exception:
                log.error(f"主循环出错:\n{traceback.format_exc()}")

            # 精确控制周期为 INTERVAL_SEC 秒
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)


# ─── 入口 ─────────────────────────────────────────────────

def run_bot():
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    run_bot()
