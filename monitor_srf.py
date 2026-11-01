#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0050 / 台積電新引擎（2026-11-01 切換）：SHARED-A 日線（SRF＋QFF）＋均線 60m（SRF 專用）。
現股代理（yfinance），紙上模擬，非真單。
設計原則：
- 冪等單次檢查：排程每 15 分鐘觸發一次，引擎只處理「新收盤的 bar」，無新 bar 即靜默退出。
- 兩系統分槽管理：SRF_SHARED、QFF_SHARED、SRF_MA 各自獨立部位與損益，互不壓制，
  與回測（各系統獨立跑）口徑一致。
- SHARED-A：只做多、單一倉位、出場當根不重進、出場統一 v1。
- MA-60m：MA5/MA20 翻倉（多空都做）、停損 1.0xATR、停損判定優先於交叉（與回測 run_ma 一致，
  停損出場後若同根有交叉，依回測邏輯同根重進）。
- TG 只在進場／出場事件與手動強制回報時發送，不發心跳。
- 出場損益即時結算進帳戶（realized_pnl、總筆數、勝筆數）。
"""
import bisect
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pytz
    TAIPEI = pytz.timezone("Asia/Taipei")
except Exception:
    TAIPEI = None

MULT = {"SRF": 1000, "QFF": 100}
LOTS = {"SRF": 2, "QFF": 1}
FEE_PER_LOT = 50
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()
FORCE_REPORT = (os.environ.get("FORCE_REPORT") or "").strip().lower() == "true"
# ---- 推播顯示名（只改顯示，內部代號 RES／VWAP／VCP／W、MA-60m、state 格式全不動）----
SIGNAL_DISPLAY = {"RES": "共振多", "VWAP": "VWAP", "VCP": "VCP", "W": "W底"}
REASON_DISPLAY = {"SL": "停損", "TRAIL": "追蹤停損", "LOCK": "鎖定出場",
                  "CAP": "遠程出場", "HEDGE": "對鎖出場", "CROSS": "交叉翻倉"}


def signal_display(strategy):
    """strategy 可能是進場事件的代號（RES）或部位存的 SHARED-RES，統一轉中文顯示名。"""
    code = (strategy or "").replace("SHARED-", "")
    return SIGNAL_DISPLAY.get(code, code)


def reason_display(reason):
    return REASON_DISPLAY.get(reason, reason)


def dir_display(direction):
    return "做多" if direction == "BUY" else "做空"


def pos_display(direction):
    return "多單" if direction == "BUY" else "空單"
# 即時觸價（2026-11-01 使用者更正：SRF／QFF 觸價一律抓期貨價、走統一 API，
# 不用現股 MIS；以下 MIS 迴圈預設關閉，僅留作備援，正式觸價由本機 unitrade 推播程式負責）。
REALTIME_TOUCH = (os.environ.get("REALTIME_TOUCH", "0") or "0").strip() != "0"
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "25"))
TOUCH_LOOP_MINUTES = int(os.environ.get("TOUCH_LOOP_MINUTES", "12"))
MIS_URL = ("https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
           "?ex_ch=tse_0050.tw|tse_2330.tw&json=1&delay=0")
SPOT_TICKER = {"SRF": "0050", "QFF": "2330"}


def fetch_mis_spot():
    """回傳 ({'0050': price, '2330': price}, date_str)；失敗回傳 (None, None)。
    取 z（成交價）欄位；d 欄位為資料日期（YYYYMMDD），用來判斷是否為當日即時價。"""
    try:
        import requests
        r = requests.get(MIS_URL, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://mis.twse.com.tw/stock/fibest.jsp",
        }, timeout=15)
        data = r.json()
        out = {}
        date_str = None
        for item in data.get("msgArray", []):
            code = (item.get("c") or "").strip()
            z = item.get("z")
            if code and z not in (None, "", "-"):
                try:
                    out[code] = float(z)
                except (TypeError, ValueError):
                    pass
            if item.get("d"):
                date_str = item["d"]
        if not out:
            return None, None
        return out, date_str
    except Exception as e:
        print(f"[MIS_ERROR] {e}")
        return None, None


def touch_check_position(asset, pos, live_price):
    """即時觸價檢查。回傳 (new_pos, exit_event_or_None)。
    只管價位觸發的出場：SHARED 的 SL／追蹤／鎖定／遠程、MA 的 SL。
    對鎖與進場仍看收盤 bar（需要收盤價與指標確認），不在此處理。
    出場價一律取觸到的水準（與回測口徑一致），不取即時價。"""
    if pos is None:
        return None, None
    if pos["slot"] == "SHARED":
        a = pos["atr"]
        pos["peak"] = max(pos["peak"], live_price)
        trail = pos["peak"] - 2.0 * a
        level, reason = trail, "TRAIL"
        if (not pos["locked"]) and live_price >= pos["entry"] + 3 * a:
            pos["locked"] = True
            pos["floor"] = pos["entry"] + 1 * a
        if pos["locked"] and pos["floor"] >= trail:
            level, reason = pos["floor"], "LOCK"
        exit_price, reason_out = None, None
        if live_price <= pos["sl"]:
            exit_price, reason_out = pos["sl"], "SL"
        elif live_price <= level:
            exit_price, reason_out = level, reason
        elif live_price >= pos["entry"] + 5 * a:
            exit_price, reason_out = pos["entry"] + 5 * a, "CAP"
        if exit_price is not None:
            pnl = trade_pnl(asset, pos["entry"], exit_price, pos["lots"], "BUY")
            return None, {"reason": reason_out, "exit": exit_price,
                          "entry": pos["entry"], "lots": pos["lots"],
                          "pnl": pnl, "strategy": pos["strategy"]}
        return pos, None
    # MA 槽：只看停損（交叉翻倉仍看收盤 bar）
    sl = pos["sl"]
    hit = (live_price <= sl) if pos["dir"] == "BUY" else (live_price >= sl)
    if hit:
        pnl = trade_pnl(asset, pos["entry"], sl, pos["lots"], pos["dir"])
        return None, {"reason": "SL", "exit": sl,
                      "entry": pos["entry"], "lots": pos["lots"],
                      "pnl": pnl, "strategy": pos["strategy"],
                      "dir": pos["dir"]}
    return pos, None


def send_tg_message(message):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("[TG_SKIPPED] no token/chat id")
        print(message)
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message,
                                     "parse_mode": "HTML"}, timeout=15)
        if r.status_code == 200 and r.json().get("ok"):
            print("[TG_PUSH_OK]")
            return True
        r2 = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message},
                           timeout=15)
        ok = r2.status_code == 200 and r2.json().get("ok")
        print("[TG_PUSH_PLAIN_OK]" if ok else f"[TG_PUSH_FAIL] {r.status_code} {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"[TG_PUSH_ERROR] {e}")
        return False


def now_taipei():
    if TAIPEI is not None:
        return datetime.now(TAIPEI)
    return datetime.utcnow() + timedelta(hours=8)


def norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        for lvl in range(df.columns.nlevels):
            vals = [str(v) for v in df.columns.get_level_values(lvl)]
            if any(v.lower() == "close" for v in vals):
                df.columns = df.columns.get_level_values(lvl)
                break
        else:
            df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.dropna(subset=["Close"])


def fetch_daily(ticker, start="2024-01-01"):
    df = yf.download(ticker, start=start, end=None, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty daily download: {ticker}")
    return norm(df)


def fetch_60m(ticker, period="60d"):
    df = yf.download(ticker, period=period, interval="60m",
                     auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"empty 60m download: {ticker}")
    return norm(df)


def derive_shared(df):
    """與融合回測 v2 的 derive_shared 完全相同的指標公式。"""
    df = df.copy()
    df["MA5s"] = df["Close"].rolling(5, min_periods=1).mean()
    df["MA20s"] = df["Close"].rolling(20, min_periods=1).mean()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol_sum = df["Volume"].rolling(20, min_periods=1).sum().replace(0, np.nan)
    df["Chip"] = ((tp * df["Volume"]).rolling(20, min_periods=1).sum()
                  / vol_sum).fillna(df["MA20s"])
    prev_c = df["Close"].shift(1).fillna(df["Open"])
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(),
                    (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=1).mean().fillna(df["Close"] * 0.005)
    atr_fast = tr.rolling(5, min_periods=1).mean()
    df["VCP_Ratio"] = (atr_fast / df["ATR"].replace(0, np.nan)).fillna(1.0)
    delta = df["Close"].diff()
    gain2 = delta.where(delta > 0, 0.0).ewm(alpha=1 / 2, adjust=False).mean()
    loss2 = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 2, adjust=False).mean()
    rs2 = gain2 / loss2.replace(0, np.nan)
    df["RSI2"] = (100.0 - (100.0 / (1.0 + rs2))).fillna(50.0)
    std20 = df["Close"].rolling(20, min_periods=1).std().fillna(0.5)
    df["BB_Upper"] = df["MA20s"] + 2.0 * std20
    df["BB_Lower"] = df["MA20s"] - 2.0 * std20
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["DIF"] - df["DEA"]
    low9 = df["Low"].rolling(9, min_periods=1).min()
    high9 = df["High"].rolling(9, min_periods=1).max()
    rsv = ((df["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50.0)
    k_list, d_list = [], []
    k_val, d_val = 50.0, 50.0
    for r in rsv:
        k_val = (2 / 3) * k_val + (1 / 3) * float(r)
        d_val = (2 / 3) * d_val + (1 / 3) * k_val
        k_list.append(k_val)
        d_list.append(d_val)
    df["K"] = k_list
    df["D"] = d_list
    df["J"] = 3 * df["K"] - 2 * df["D"]
    for p in [10, 60, 120]:
        df[f"MA{p}"] = df["Close"].rolling(p).mean()
    return df


def last_before(dates, d):
    return bisect.bisect_left(dates, d) - 1


def shared_signals_at(df, twii, ixic, i):
    """回傳 (conds dict, hedge bool)，公式與融合回測 v2 的 build_shared_signals 完全相同。"""
    row, prev = df.iloc[i], df.iloc[i - 1] if i > 0 else df.iloc[i]
    close = float(row["Close"])
    d = df.index[i]
    chip = float(row["Chip"])
    twii_dates, ixic_dates = list(twii.index), list(ixic.index)
    j = last_before(twii_dates, d)
    tw = twii.iloc[j] if j >= 0 else twii.iloc[0]
    tx_bull = bool(float(tw["Close"]) >= float(tw["MA20"])) if not pd.isna(tw["MA20"]) else False
    tx_bear = not tx_bull
    k = last_before(ixic_dates, d)
    nq = ixic.iloc[k] if k >= 0 else ixic.iloc[0]
    nq_bull = bool(float(nq["Close"]) >= float(nq["MA20"])) if not pd.isna(nq["MA20"]) else False
    nq_bear = not nq_bull
    prev_mid = (float(prev["Open"]) + float(prev["Close"])) / 2.0
    k_, d_ = float(row["K"]), float(row["D"])
    j_ = float(row["J"])
    macd, macd_p = float(row["MACD_Hist"]), float(prev["MACD_Hist"])
    rsi_p = float(prev["RSI2"])
    bb_lo_p = float(prev["BB_Lower"])
    low_p = float(prev["Low"])
    conds = {}
    conds["RES"] = bool(tx_bull and nq_bull and close >= chip
                        and close >= float(row["MA10"]) and k_ > d_
                        and 25 <= j_ <= 78 and macd >= 0)
    conds["VWAP"] = bool(tx_bull and nq_bull and close >= float(row["MA60"])
                         and low_p <= chip * 1.004 and close >= chip
                         and k_ >= d_)
    conds["VCP"] = bool(float(prev["VCP_Ratio"]) <= 0.96
                        and close > float(row["MA10"]) and close > chip
                        and close >= float(prev["High"])
                        and 30 <= j_ <= 84)
    conds["W"] = bool(close >= float(row["MA120"]) * 0.985
                      and close >= prev_mid and macd >= macd_p
                      and not nq_bear
                      and (rsi_p <= 22 or low_p <= bb_lo_p * 1.003))
    hedge = bool((((float(prev["High"]) >= float(prev["BB_Upper"]) * 0.999
                    or rsi_p >= 88) and close < float(row["MA5s"]))
                   or (tx_bear and nq_bear and close < chip
                       and close < float(row["MA20s"]) and k_ < d_))
                  and macd < macd_p)
    return conds, hedge


SHARED_ORDER = ["RES", "VWAP", "VCP", "W"]


def trade_pnl(asset, entry, exit_price, lots, direction="BUY"):
    pts = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
    return int(round(pts * MULT[asset] * lots)) - FEE_PER_LOT * lots


def shared_bar_update(asset, df, twii, ixic, pos, i):
    """處理一根日線 bar。回傳 (new_pos, events)。
    出場序與回測 run_pool 完全一致：SL > 追蹤／鎖定 > 遠程 > 對鎖出場；出場當根不重進。
    EXIT 事件帶 entry／exit／pnl，供記帳與推播用。"""
    row = df.iloc[i]
    close = float(row["Close"])
    hi, lo = float(row["High"]), float(row["Low"])
    conds, hedge = shared_signals_at(df, twii, ixic, i)
    pick = next((s for s in SHARED_ORDER if conds.get(s)), None)
    events = []
    exited = False
    if pos is not None:
        a = pos["atr"]
        pos["peak"] = max(pos["peak"], close)
        trail = pos["peak"] - 2.0 * a
        level, reason = trail, "TRAIL"
        if (not pos["locked"]) and hi >= pos["entry"] + 3 * a:
            pos["locked"] = True
            pos["floor"] = pos["entry"] + 1 * a
        if pos["locked"] and pos["floor"] >= trail:
            level, reason = pos["floor"], "LOCK"
        exit_price, reason_out = None, None
        if lo <= pos["sl"]:
            exit_price, reason_out = pos["sl"], "SL"
        elif lo <= level:
            exit_price, reason_out = level, reason
        elif hi >= pos["entry"] + 5 * a:
            exit_price, reason_out = pos["entry"] + 5 * a, "CAP"
        elif hedge:
            exit_price, reason_out = close, "HEDGE"
        if exit_price is not None:
            pnl = trade_pnl(asset, pos["entry"], exit_price, pos["lots"], "BUY")
            events.append(("EXIT", {"reason": reason_out, "exit": exit_price,
                                    "entry": pos["entry"], "lots": pos["lots"],
                                    "pnl": pnl, "strategy": pos["strategy"]}))
            pos = None
            exited = True
    if pos is None and not exited and pick is not None and i >= 120:
        atr = float(row["ATR"])
        pos = {"asset": asset, "slot": "SHARED", "dir": "BUY",
               "entry": close, "sl": close - 1.25 * atr,
               "entry_i": i, "strategy": "SHARED-" + pick, "lots": LOTS[asset],
               "atr": atr, "peak": close, "locked": False, "floor": None}
        events.append(("ENTRY", {"strategy": pick, "entry": close,
                                 "sl": close - 1.25 * atr, "atr": atr}))
    return pos, events


def prepare_ma(df):
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    prev_c = df["Close"].shift(1).fillna(df["Open"])
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(),
                    (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=1).mean().fillna(df["Close"] * 0.005)
    return df


def ma_bar_update(asset, df, pos, i):
    """處理一根 60m bar。與回測 run_ma 完全一致：先判停損，再判交叉翻倉，
    停損出場後若同根有交叉，依回測邏輯同根重進。EXIT 事件帶 entry／exit／pnl。"""
    row, prev = df.iloc[i], df.iloc[i - 1]
    close = float(row["Close"])
    hi, lo = float(row["High"]), float(row["Low"])
    golden = float(prev["MA5"]) <= float(prev["MA20"]) and float(row["MA5"]) > float(row["MA20"])
    death = float(prev["MA5"]) >= float(prev["MA20"]) and float(row["MA5"]) < float(row["MA20"])
    events = []
    if pos is not None:
        sl = pos["sl"]
        hit = (lo <= sl) if pos["dir"] == "BUY" else (hi >= sl)
        if hit:
            pnl = trade_pnl(asset, pos["entry"], sl, pos["lots"], pos["dir"])
            events.append(("EXIT", {"reason": "SL", "exit": sl,
                                    "entry": pos["entry"], "lots": pos["lots"],
                                    "pnl": pnl, "strategy": pos["strategy"],
                                    "dir": pos["dir"]}))
            pos = None
    if pos is not None and ((pos["dir"] == "BUY" and death) or (pos["dir"] == "SELL" and golden)):
        pnl = trade_pnl(asset, pos["entry"], close, pos["lots"], pos["dir"])
        events.append(("EXIT", {"reason": "CROSS", "exit": close,
                                "entry": pos["entry"], "lots": pos["lots"],
                                "pnl": pnl, "strategy": pos["strategy"],
                                "dir": pos["dir"]}))
        pos = None
    if pos is None and (golden or death):
        direction = "BUY" if golden else "SELL"
        atr = float(row["ATR"])
        sl = close - atr if direction == "BUY" else close + atr
        pos = {"asset": asset, "slot": "MA", "dir": direction,
               "entry": close, "sl": sl, "entry_i": i,
               "strategy": "MA-60m", "lots": LOTS[asset], "atr": atr}
        events.append(("ENTRY", {"strategy": "MA-60m", "dir": direction,
                                 "entry": close, "sl": sl, "atr": atr}))
    return pos, events


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"account_version": "2026-11-01_SHARED_MA_V1",
            "account": {"initial_capital": 100000, "realized_pnl": 0,
                        "total_trades_count": 0, "win_trades_count": 0},
            "positions": {"SRF_SHARED": None, "QFF_SHARED": None, "SRF_MA": None},
            "last_processed": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def daily_bar_closed(bar_date, now):
    """日線 bar 是否已收盤：bar 日期 < 今天，或 bar 日期 == 今天且已過 13:35。"""
    today = now.date()
    if bar_date < today:
        return True
    if bar_date == today and (now.hour, now.minute) >= (13, 35):
        return True
    return False


def ma60_bar_closed(bar_start, now):
    """60m bar 是否已收盤：bar 開始 +65 分鐘 < 現在，或 13:00 bar 且已過 13:35。"""
    naive_now = now.replace(tzinfo=None)
    bar_naive = bar_start.replace(tzinfo=None) if getattr(bar_start, "tzinfo", None) else bar_start
    if bar_naive.hour == 13 and (naive_now.hour, naive_now.minute) >= (13, 35):
        return True
    return bar_naive + timedelta(minutes=65) <= naive_now


def fmt_date(idx_val):
    try:
        return pd.to_datetime(idx_val).strftime("%Y-%m-%d")
    except Exception:
        return str(idx_val)


def book_exit(acct, pnl):
    acct["realized_pnl"] = acct.get("realized_pnl", 0) + pnl
    acct["total_trades_count"] = acct.get("total_trades_count", 0) + 1
    if pnl > 0:
        acct["win_trades_count"] = acct.get("win_trades_count", 0) + 1


def main():
    now = now_taipei()
    state = load_state()
    acct = state.setdefault("account", {})
    positions = state.setdefault("positions", {})
    last = state.setdefault("last_processed", {})
    tg_msgs = []

    # ---- SHARED-A 日線（每資產獨立 last_processed，避免資料延遲漏 bar）----
    try:
        data = {}
        for key, ticker in (("0050", "0050.TW"), ("2330", "2330.TW"),
                            ("TWII", "^TWII"), ("IXIC", "^IXIC")):
            data[key] = fetch_daily(ticker)
        twii = data["TWII"].copy()
        twii["MA20"] = twii["Close"].rolling(20).mean()
        ixic = data["IXIC"].copy()
        ixic["MA20"] = ixic["Close"].rolling(20).mean()
        for asset, key in (("SRF", "0050"), ("QFF", "2330")):
            df = derive_shared(data[key])
            slot = asset + "_SHARED"
            pos = positions.get(slot)
            last_key = "daily_" + asset
            last_val = last.get(last_key)
            for i in range(len(df)):
                bar_str = fmt_date(df.index[i])
                if last_val is not None and bar_str <= last_val:
                    continue
                if not daily_bar_closed(pd.to_datetime(df.index[i]).date(), now):
                    continue
                pos, events = shared_bar_update(asset, df, twii, ixic, pos, i)
                for kind, det in events:
                    if kind == "ENTRY":
                        tg_msgs.append(
                            f"{asset} {signal_display(det['strategy'])} 進場做多 "
                            f"{LOTS[asset]}口 @{det['entry']:.2f} 停損{det['sl']:.2f}")
                    else:
                        book_exit(acct, det["pnl"])
                        tg_msgs.append(
                            f"{asset} {signal_display(det['strategy'])} 出場 "
                            f"{reason_display(det['reason'])} "
                            f"{det['entry']:.2f}→{det['exit']:.2f} 損益{det['pnl']:+d}")
                last[last_key] = bar_str
            positions[slot] = pos
    except Exception as e:
        print(f"[SHARED_ERROR] {e}")
        tg_msgs.append(f"<b>新引擎 SHARED 日線抓取／計算失敗</b>：{e}")

    # ---- MA-60m（SRF 專用）----
    try:
        df60 = prepare_ma(fetch_60m("0050.TW", period="60d"))
        pos = positions.get("SRF_MA")
        last_ma = last.get("ma60_bar")
        for i in range(20, len(df60)):
            bar_key = pd.to_datetime(df60.index[i]).strftime("%Y-%m-%d %H:%M")
            if last_ma is not None and bar_key <= last_ma:
                continue
            if not ma60_bar_closed(df60.index[i], now):
                continue
            pos, events = ma_bar_update("SRF", df60, pos, i)
            for kind, det in events:
                if kind == "ENTRY":
                    tg_msgs.append(
                        f"SRF 均線60m 進場{dir_display(det['dir'])} "
                        f"{LOTS['SRF']}口 @{det['entry']:.2f} 停損{det['sl']:.2f}")
                else:
                    book_exit(acct, det["pnl"])
                    tg_msgs.append(
                        f"SRF 均線60m {pos_display(det['dir'])}出場 "
                        f"{reason_display(det['reason'])} "
                        f"{det['entry']:.2f}→{det['exit']:.2f} 損益{det['pnl']:+d}")
            last["ma60_bar"] = bar_key
        positions["SRF_MA"] = pos
    except Exception as e:
        print(f"[MA60_ERROR] {e}")
        tg_msgs.append(f"<b>新引擎 MA-60m 抓取／計算失敗</b>：{e}")

    # ---- 即時觸價迴圈（TWSE 現股 MIS，25 秒輪詢，僅出場價位）----
    # 只在台股交易時段跑；MIS 日期不是今天（假日／休市）就跳過，避免用舊價觸發。
    # 迴圈 12 分鐘短於 15 分鐘觸發間隔，避免兩個迴圈同時寫 state。
    if REALTIME_TOUCH and now.weekday() < 5:
        hhmm = now.hour * 60 + now.minute
        if 9 * 60 <= hhmm <= 13 * 60 + 35:
            prices, mis_date = fetch_mis_spot()
            today_str = now.strftime("%Y%m%d")
            if prices is None or mis_date != today_str:
                print(f"[TOUCH_SKIP] MIS 無法確認當日即時價（mis_date={mis_date}，today={today_str}）")
            else:
                n_iter = max(1, TOUCH_LOOP_MINUTES * 60 // POLL_INTERVAL_SEC)
                print(f"[TOUCH_LOOP] 開始，{n_iter} 輪 × {POLL_INTERVAL_SEC} 秒")
                for it in range(n_iter):
                    if it > 0:
                        prices, mis_date = fetch_mis_spot()
                    if prices is None:
                        time.sleep(POLL_INTERVAL_SEC)
                        continue
                    for slot in ("SRF_SHARED", "QFF_SHARED", "SRF_MA"):
                        pos = positions.get(slot)
                        if pos is None:
                            continue
                        asset = "SRF" if slot.startswith("SRF") else "QFF"
                        live = prices.get(SPOT_TICKER[asset])
                        if live is None:
                            continue
                        new_pos, ev = touch_check_position(asset, pos, live)
                        positions[slot] = new_pos
                        if ev is not None:
                            book_exit(acct, ev["pnl"])
                            if slot == "SRF_MA":
                                label = f"SRF 均線60m {pos_display(ev['dir'])}"
                            else:
                                label = f"{asset} {signal_display(ev.get('strategy', ''))}"
                            msg = (f"⚡{label} 觸價出場 "
                                   f"{reason_display(ev['reason'])} "
                                   f"{ev['entry']:.2f}→{ev['exit']:.2f} 損益{ev['pnl']:+d} "
                                   f"（現價 {live:.2f}）")
                            print(msg)
                            send_tg_message(msg)
                    if it < n_iter - 1:
                        time.sleep(POLL_INTERVAL_SEC)
                print("[TOUCH_LOOP] 結束")

    save_state(state)
    for m in tg_msgs:
        print(m)
    if tg_msgs:
        send_tg_message("\n\n".join(tg_msgs))
    elif FORCE_REPORT:
        def slot_line(slot, asset, label):
            pos = positions.get(slot)
            if pos is None:
                return f"{asset} {label}：無部位"
            if slot == "SRF_MA":
                return (f"{asset} {label}：{pos_display(pos['dir'])} {pos['lots']}口 "
                        f"@{pos['entry']:.2f} 停損{pos['sl']:.2f}")
            return (f"{asset} {signal_display(pos.get('strategy', ''))}：多單 {pos['lots']}口 "
                    f"@{pos['entry']:.2f} 停損{pos['sl']:.2f}")
        summary = (
            f"<b>新引擎狀態（{now.strftime('%Y-%m-%d %H:%M')}）</b>\n"
            f"{slot_line('SRF_SHARED', 'SRF', '共用')}\n"
            f"{slot_line('QFF_SHARED', 'QFF', '共用')}\n"
            f"{slot_line('SRF_MA', 'SRF', '均線60m')}\n"
            f"已實現：{acct.get('realized_pnl')}")
        print(summary)
        send_tg_message(summary)
    else:
        print("[NO_NEW_EVENTS]")


if __name__ == "__main__":
    main()
