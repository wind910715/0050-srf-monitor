# =========================================================================
# 台積電均線趨勢模擬監控（GitHub Actions 用）
# 策略（凍結版，使用者 2026-10-27 拍板採用）：
#   2330 60m，MA5 上穿 MA20 做多、下穿做空，翻倉制（反向交叉收盤價平舊倉、
#   同價開新倉，恆在場）；硬停損 1.0×ATR（進場根 ATR，live floor 約束）；
#   無 TP；停損出場後空手直至下一個交叉。1 口、每筆 50 元、點數乘數 100。
# 引擎與第 2 輪 R2B 回測（run_ma_trend，SL 1.0）同邏輯，僅另回傳未平倉位。
# 監控以小時 K「收盤」判斷（與回測同口徑，非逐筆）；未完成的小時 K 不納入。
# 純模擬帳本：不含 Telegram 程式碼、不含券商 API、不下單。
# 現階段資料為現股 60m 代理（yfinance 60m 上限 730 天滾動窗口）；
# 期貨真實價格接上後再切換資料源並重測。
# 每次執行皆全量重算（冪等、可重現），state.json 與 monitor_log.md 寫回 repo。
# =========================================================================
import json
import os
import sys
from datetime import datetime, time as dtime, timedelta

import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weekly_backtest import _MULT, _layer_stats, _live_dists, prepare_master_df

TW_TZ = pytz.timezone("Asia/Taipei")
ASSET = "QFF"
MULT = _MULT[ASSET]
LOTS = 1
FEE = 50
SL_MULT = 1.0
STATE_FILE = "state.json"
LOG_FILE = "monitor_log.md"
MAX_LOG_LINES = 500


def bar_complete(bar_start, now):
    """小時 K 是否已收盤。13:00 根為 13:00–13:30 短根，13:30 後即視為收盤；
    其餘根以 start＋60 分鐘為準。"""
    if bar_start.hour == 13 and now.date() == bar_start.date():
        return now.time() >= dtime(13, 30)
    return now >= bar_start + timedelta(minutes=60)


def replay(df):
    """回放 60m 翻倉引擎，回傳（已完成交易列、未平倉位）。"""
    trades = []
    pos = None
    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        hi, lo = float(row["High"]), float(row["Low"])
        close = float(row["Close"])
        t = df.index[i]
        golden = prev["MA5"] <= prev["MA20"] and row["MA5"] > row["MA20"]
        death = prev["MA5"] >= prev["MA20"] and row["MA5"] < row["MA20"]
        if pos is not None:
            sl = pos["sl"]
            hit_sl = (lo <= sl) if pos["dir"] == "BUY" else (hi >= sl)
            if hit_sl:
                pnl = int(round(
                    (sl - pos["entry"]) * MULT * LOTS
                    if pos["dir"] == "BUY"
                    else (pos["entry"] - sl) * MULT * LOTS)) - FEE * LOTS
                trades.append({"dir": pos["dir"], "entry_time": pos["entry_time"],
                               "exit_time": str(t), "entry": pos["entry"],
                               "exit": sl, "reason": "SL", "pnl_ntd": pnl,
                               "entry_i": pos["entry_i"], "exit_i": i})
                pos = None
                continue
            if (pos["dir"] == "BUY" and death) or (pos["dir"] == "SELL" and golden):
                pnl = int(round(
                    (close - pos["entry"]) * MULT * LOTS
                    if pos["dir"] == "BUY"
                    else (pos["entry"] - close) * MULT * LOTS)) - FEE * LOTS
                trades.append({"dir": pos["dir"], "entry_time": pos["entry_time"],
                               "exit_time": str(t), "entry": pos["entry"],
                               "exit": close, "reason": "CROSS", "pnl_ntd": pnl,
                               "entry_i": pos["entry_i"], "exit_i": i})
                pos = None
                direction = "SELL" if death else "BUY"
                sl_d, _, _ = _live_dists(ASSET, float(row["ATR"]), SL_MULT, 2.0, 3.5)
                sl = close - sl_d if direction == "BUY" else close + sl_d
                pos = {"dir": direction, "entry": close, "entry_time": str(t),
                       "sl": sl, "entry_i": i}
                continue
        if pos is None:
            if golden:
                direction = "BUY"
            elif death:
                direction = "SELL"
            else:
                continue
            sl_d, _, _ = _live_dists(ASSET, float(row["ATR"]), SL_MULT, 2.0, 3.5)
            sl = close - sl_d if direction == "BUY" else close + sl_d
            pos = {"dir": direction, "entry": close, "entry_time": str(t),
                   "sl": sl, "entry_i": i}
    return trades, pos


def main():
    now = datetime.now(TW_TZ)
    df = prepare_master_df("2330.TW", "730d", "60m")
    if df.index.tz is None:
        df.index = df.index.tz_localize(TW_TZ)
    else:
        df.index = df.index.tz_convert(TW_TZ)
    df["MA5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    # 剔除未完成的小時 K（僅判斷最後一根）
    if len(df) > 0 and not bar_complete(df.index[-1].to_pydatetime(), now):
        df = df.iloc[:-1]
    trades, pos = replay(df)
    mid_i = len(df) // 2
    # 沿用回測統計口徑（entry_i 為 bar 序號，前後半以 bar 中點切分）
    stat_trades = []
    for tr in trades:
        stat_trades.append({**tr, "lots": LOTS,
                            "pnl_pts": tr["pnl_ntd"] / MULT / LOTS})
    stats = _layer_stats(stat_trades, max(mid_i, 1)) if trades else {
        "total": 0, "win_rate": 0.0, "avg_ntd": 0.0, "rr": 0.0, "net_ntd": 0}
    latest = df.iloc[-1] if len(df) > 0 else None
    latest_bar = {"time": str(df.index[-1]),
                  "close": float(latest["Close"])} if latest is not None else None

    prev = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:  # noqa: BLE001
            prev = None
    new_trades = []
    if prev is not None:
        prev_keys = {(t.get("exit_time"), t.get("dir"), t.get("entry_time"))
                     for t in prev.get("trades", [])}
        new_trades = [t for t in trades
                      if (t["exit_time"], t["dir"], t["entry_time"]) not in prev_keys]
    first_run = prev is None

    state = {
        "updated_at": now.strftime("%Y-%m-%d %H:%M %Z"),
        "strategy": "台積電均線趨勢 60m MA5/MA20 翻倉＋SL 1.0 ATR（模擬，現股代理）",
        "window": [str(df.index[0]), str(df.index[-1]), len(df)],
        "latest_bar": latest_bar,
        "position": pos,
        "summary": {"n": stats["total"], "win_rate": stats["win_rate"],
                    "avg_ntd": stats["avg_ntd"], "rr": stats["rr"],
                    "net_ntd": stats["net_ntd"],
                    "first_avg_ntd": stats.get("first_avg_ntd"),
                    "second_avg_ntd": stats.get("second_avg_ntd")},
        "trades": trades,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    pos_txt = "空手" if pos is None else (
        f"{'多' if pos['dir'] == 'BUY' else '空'}（進場 {pos['entry']:,.0f}，"
        f"停損 {pos['sl']:,.0f}，自 {pos['entry_time']}）")
    new_txt = "首次建立" if first_run else (
        "；".join(f"{t['dir']} {t['reason']} {t['pnl_ntd']:+,}" for t in new_trades)
        if new_trades else "無新完成交易")
    line = (f"- {now.strftime('%Y-%m-%d %H:%M')} 最新60m收盤 "
            f"{latest_bar['time'] if latest_bar else '-'} 收 {latest_bar['close']:,.0f}｜"
            f"倉位：{pos_txt}｜帳本 {stats['total']} 筆、累計 {stats['net_ntd']:+,}｜"
            f"{new_txt}\n")
    lines = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
    lines.append(line)
    lines = lines[-MAX_LOG_LINES:]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(line.strip())
    print(f"position: {pos_txt} | trades: {stats['total']} | net: {stats['net_ntd']:+,}")


if __name__ == "__main__":
    main()
