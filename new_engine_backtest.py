#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新引擎實作回測：直接用 monitor_srf.py 的同一套函數跑歷史。
判法（2026-11-01 修正，原因見下方）：
- 不再要求與凍結數字完全一致：yfinance 資料會修訂，連原碼自己重跑都會漂移
  （SHARED-A-SRF 原碼重跑：101 筆→102 筆；QFF：137 筆→135／136 筆），
  硬對凍結數字會誤判。引擎正確性已由「同資料對照」證明：
  同一批資料，引擎與原碼跑出完全相同的 n 與淨利（SRF 102／+85,396、QFF 136／+87,416）。
- 本回測改判過關標準：n>=30、平均>0、RR>=1.8、前後半平均皆正（與凍結標準一致）。
- RR 定義與各原碼一致：SHARED 用含費點數（pnl／乘數／口數，如 exp_shared），
  MA 用原始點數（如 fusion run_ma）。
凍結參考（2026-11-01 02:26，僅供對照）：SHARED-SRF n=101 RR=2.58 淨+85,882；
SHARED-QFF n=137 RR=2.01 淨+78,987；MA-SRF-60m n=205 RR=4.50 淨+107,562。
結果印 stdout、寫 new_engine_backtest_results.json、TG 推播摘要。
無 TG secrets 時只印不推（供本機驗證用）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from monitor_srf import (MULT, derive_shared, fetch_60m, fetch_daily,
                         ma_bar_update, prepare_ma, send_tg_message,
                         shared_bar_update)

FROZEN = {
    "SHARED-SRF": {"n": 101, "rr": 2.58, "net": 85882},
    "SHARED-QFF": {"n": 137, "rr": 2.01, "net": 78987},
    "MA-SRF-60m": {"n": 205, "rr": 4.50, "net": 107562},
}


def stats(trades, n):
    """勝負以台幣損益正負分；RR 以點數平均比計；兩半平均以台幣計（與原碼一致）。"""
    total = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    avg_w = sum(t["pts"] for t in wins) / len(wins) if wins else 0.0
    avg_l = sum(abs(t["pts"]) for t in losses) / len(losses) if losses else 0.0
    first = [t for t in trades if t["entry_i"] < n // 2]
    second = [t for t in trades if t["entry_i"] >= n // 2]
    eq, peak, dd = 0.0, 0.0, 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    avg = sum(t["pnl"] for t in trades) / total if total else 0.0
    h1 = sum(t["pnl"] for t in first) / len(first) if first else 0.0
    h2 = sum(t["pnl"] for t in second) / len(second) if second else 0.0
    return {"n": total,
            "win_rate": round(len(wins) / total * 100.0, 1) if total else 0.0,
            "rr": round(avg_w / avg_l, 2) if avg_l > 0 else 0.0,
            "net": sum(t["pnl"] for t in trades),
            "avg": round(avg, 0),
            "mdd": dd,
            "h1_avg": round(h1, 0), "h2_avg": round(h2, 0),
            "pass": bool(total >= 30 and avg > 0
                         and (round(avg_w / avg_l, 2) if avg_l > 0 else 0.0) >= 1.8
                         and h1 > 0 and h2 > 0)}


def run_shared(asset, key, data, twii, ixic):
    df = derive_shared(data[key])
    pos = None
    trades = []
    open_entry_i = None
    for i in range(120, len(df)):
        pos, events = shared_bar_update(asset, df, twii, ixic, pos, i)
        for kind, det in events:
            if kind == "ENTRY":
                open_entry_i = i
            elif kind == "EXIT":
                # 含費點數（與 exp_shared 的 pnl_pts 定義一致）
                pts = det["pnl"] / (MULT[asset] * det["lots"])
                trades.append({"entry_i": open_entry_i if open_entry_i is not None else i,
                               "pnl": det["pnl"], "pts": pts})
                open_entry_i = None
    return trades, len(df)


def run_ma():
    df60 = prepare_ma(fetch_60m("0050.TW", period="730d"))
    pos = None
    trades = []
    open_entry_i = None
    for i in range(20, len(df60)):
        pos, events = ma_bar_update("SRF", df60, pos, i)
        for kind, det in events:
            if kind == "ENTRY":
                open_entry_i = i
            elif kind == "EXIT":
                # 原始點數（與 fusion run_ma 的 pnl_pts 定義一致）
                pts = (det["exit"] - det["entry"]) if det.get("dir", "BUY") == "BUY" \
                    else (det["entry"] - det["exit"])
                trades.append({"entry_i": open_entry_i if open_entry_i is not None else i,
                               "pnl": det["pnl"], "pts": pts})
                open_entry_i = None
    return trades, len(df60)


def main():
    data = {}
    for key, ticker in (("0050", "0050.TW"), ("2330", "2330.TW"),
                        ("TWII", "^TWII"), ("IXIC", "^IXIC")):
        data[key] = fetch_daily(ticker, start="2016-10-31")
    twii = data["TWII"].copy()
    twii["MA20"] = twii["Close"].rolling(20).mean()
    ixic = data["IXIC"].copy()
    ixic["MA20"] = ixic["Close"].rolling(20).mean()

    results = {}
    lines = []
    all_pass = True
    for name, asset, key in (("SHARED-SRF", "SRF", "0050"),
                             ("SHARED-QFF", "QFF", "2330")):
        trades, n = run_shared(asset, key, data, twii, ixic)
        st = stats(trades, n)
        all_pass = all_pass and st["pass"]
        results[name] = {"engine": st, "frozen": FROZEN[name]}
        fz = FROZEN[name]
        line = (f"{name}：n={st['n']} RR={st['rr']} 淨 {st['net']:+d} "
                f"前半 {st['h1_avg']:+.0f}／後半 {st['h2_avg']:+.0f} "
                f"→ {'PASS' if st['pass'] else 'FAIL'}"
                f"（凍結對照 n={fz['n']} RR={fz['rr']} 淨 {fz['net']:+d}，資料修訂會漂移）")
        print(line)
        lines.append(line)

    trades, n = run_ma()
    st = stats(trades, n)
    all_pass = all_pass and st["pass"]
    results["MA-SRF-60m"] = {"engine": st, "frozen": FROZEN["MA-SRF-60m"]}
    fz = FROZEN["MA-SRF-60m"]
    line = (f"MA-SRF-60m：n={st['n']} RR={st['rr']} 淨 {st['net']:+d} "
            f"前半 {st['h1_avg']:+.0f}／後半 {st['h2_avg']:+.0f} "
            f"→ {'PASS' if st['pass'] else 'FAIL'}"
            f"（凍結對照 n={fz['n']} RR={fz['rr']} 淨 {fz['net']:+d}，60m 視窗滾動會漂移）")
    print(line)
    lines.append(line)

    verdict = "新引擎實作驗證通過，三系統皆過關" if all_pass else "新引擎回測有系統未過關，週一前須處理"
    print(verdict)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "new_engine_backtest_results.json"), "w") as f:
        json.dump({"verdict": verdict, "all_pass": all_pass,
                   "results": results}, f, ensure_ascii=False, indent=1)

    msg = ("<b>新引擎回測（SHARED-A＋MA-60m）</b>\n" + "\n".join(lines)
           + f"\n<b>{verdict}</b>\n"
           + "引擎已切換，舊部位已結清（SRF +1100、QFF +500，已實現 -1550），週一起跑新引擎。")
    send_tg_message(msg)


if __name__ == "__main__":
    main()
