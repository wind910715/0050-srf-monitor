# =========================================================================
# 穩健版 v1 現股模擬（2026-10-27，使用者批准方向的第一版測試）
# 觸發：python stable_sim_v1.py
# 本機測試用；不發 Telegram、不含券商 API、不下單、不 push。
# 凍結規格（事前註冊，跑完不改參數湊過關）：
# - 資料：yfinance 0050.TW／2330.TW 15m 現股代理，60 天（與前兩輪同一窗口）
# - 策略池（只留順勢類）：SRF：S2（VCP）、S3-live（第一輪 V4 實盤等價放寬條件）；
#   QFF：T2（籌碼順勢共振多）。刪除 S1、S4、T1（逆勢接刀類）
# - 欄位結構：每策略獨立欄位，可並存；全帳戶同時最多 2 個波段部位，
#   滿額時新訊號跳過並計入 skipped（先觸發先佔，不用優先序取捨）
# - 出場：沿用重構版波段參數（SL 1.25 ATR、TP1 2.0 ATR 出一半＋剩餘移保本、
#   TP2 3.5 ATR），出場優先序 SL ＞ TP2 ＞ TP1；每交易日第一根 K 不觸發硬停損
# - 過關標準：與前兩輪一致（平均正、RR ≥ 1.8、前後半皆正；n＜30 標註樣本不足）
# 方法限制（報告中須揭露）：
# - 現貨 15m 代理期貨價；無夜盤 K 線，夜盤緩衝／減半無法驗證
# - 5m 條件以 15m 近似；NQ／ADR 濾網未納入（下一輪與期貨真實價格一併驗證）
# - S2 以回測版 VCP_Ratio 0.92 為代表（實盤為 0.96）；S3-live 以 TX_Bull 代理
#   tx_bullish、NQ 省略；60 天窗口短，2022 空頭與 2020 崩盤不在內
# - 兩商品以時間戳交集對齊後共用一條時間軸跑全帳戶上限；半段以對齊後
#   資料列數中點切分（切點日期可能與前兩輪略差，報告中寫明實際切點）
# =========================================================================
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

import pytz

from weekly_backtest import (
    _LOTS,
    _MULT,
    _SIGNAL_FN,
    _layer_stats,
    _live_dists,
    _variant_verdict,
    prepare_master_df,
)

TW_TZ = pytz.timezone("Asia/Taipei")

SL_MULT, TP1_MULT, TP2_MULT = 1.25, 2.0, 3.5
MAX_OPEN = 2  # 全帳戶同時最多 2 個波段部位

# (asset, slot_key, algo_id, label, param)
SLOTS_SPEC = [
    ("SRF", "S2", "MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80),
    ("SRF", "S3-live", "S3_LIVE", "S3-live 實盤等價條件", 78),
    ("QFF", "T2", "SNIPER_CHIP_MOMENTUM", "T2 日夜盤籌碼順勢共振多", 65),
]


def close_trade(trades, pos, exit_price, reason, exit_i):
    mult = _MULT[pos["asset"]]
    pnl = pos["pnl_partial"] + int(round(
        (exit_price - pos["entry"]) * mult * pos["lots"]
        if pos["dir"] == "BUY"
        else (pos["entry"] - exit_price) * mult * pos["lots"]
    )) - 50 * pos["lots"]
    total_lots = pos["closed_lots"] + pos["lots"]
    trades.append({
        "asset": pos["asset"], "layer": "swing",
        "strategy": pos["strategy"], "slot": pos["slot"],
        "dir": pos["dir"], "entry_i": pos["entry_i"], "exit_i": exit_i,
        "entry": pos["entry"], "exit": exit_price, "reason": reason,
        "lots": total_lots, "pnl_ntd": pnl,
        "pnl_pts": pnl / mult / total_lots,
        "sl_dist": pos["sl_dist"], "tp1_dist": pos["tp1_dist"],
        "tp2_dist": pos["tp2_dist"],
    })


def run_stable_sim_v1():
    now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
    df_0050 = prepare_master_df("0050.TW", "60d", "15m")
    df_2330 = prepare_master_df("2330.TW", "60d", "15m")
    # 兩商品以時間戳交集對齊，共用一條時間軸（全帳戶上限需要在同一時間軸上算）
    unified = sorted(set(df_0050.index) & set(df_2330.index))
    dfs = {"SRF": df_0050.reindex(unified), "QFF": df_2330.reindex(unified)}
    n_bars = len(unified)
    mid_i = n_bars // 2
    cut_date = str(unified[mid_i].date())

    dates = [d.date() for d in unified]
    first_bar = set()
    seen = set()
    for i, d in enumerate(dates):
        if d not in seen:
            seen.add(d)
            first_bar.add(i)

    slots = {key: None for _, key, _, _, _ in SLOTS_SPEC}
    spec_by_key = {key: (asset, algo_id, label, param)
                   for asset, key, algo_id, label, param in SLOTS_SPEC}
    trades = []
    skipped = 0
    censored = 0
    for i in range(30, n_bars):
        suppress_sl = i in first_bar
        exited = set()
        open_count = sum(1 for v in slots.values() if v is not None)
        # ---- 出場（先於進場）----
        for key, pos in list(slots.items()):
            if pos is None:
                continue
            df = dfs[pos["asset"]]
            row = df.iloc[i]
            hi, lo = float(row["High"]), float(row["Low"])
            if pos["dir"] == "BUY":
                hit_sl = (not suppress_sl) and lo <= pos["sl"]
                hit_tp2 = hi >= pos["tp2"]
                hit_tp1 = (not pos["tp1_hit"]) and hi >= pos["tp1"]
            else:
                hit_sl = (not suppress_sl) and hi >= pos["sl"]
                hit_tp2 = lo <= pos["tp2"]
                hit_tp1 = (not pos["tp1_hit"]) and lo <= pos["tp1"]
            if hit_sl:
                close_trade(trades, pos, pos["sl"], "SL", i)
                slots[key] = None
                exited.add(key)
                open_count -= 1
            elif hit_tp2:
                close_trade(trades, pos, pos["tp2"], "TP2", i)
                slots[key] = None
                exited.add(key)
                open_count -= 1
            elif hit_tp1:
                mult = _MULT[pos["asset"]]
                close_n = pos["lots"] // 2
                if close_n > 0:
                    if pos["dir"] == "BUY":
                        pp = int(round((pos["tp1"] - pos["entry"]) * mult * close_n)) - 50 * close_n
                    else:
                        pp = int(round((pos["entry"] - pos["tp1"]) * mult * close_n)) - 50 * close_n
                    pos["pnl_partial"] += pp
                    pos["closed_lots"] += close_n
                    pos["lots"] -= close_n
                pos["tp1_hit"] = True
                pos["sl"] = pos["entry"]
        # ---- 進場（出場後；本根出場的欄位本根不重進）----
        for asset, key, algo_id, label, param in SLOTS_SPEC:
            if slots[key] is not None or key in exited:
                continue
            df = dfs[asset]
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            direction = _SIGNAL_FN[algo_id](row, prev, algo_id, param)
            if direction != "BUY":
                continue
            if open_count >= MAX_OPEN:
                skipped += 1
                continue
            close = float(row["Close"])
            sl_d, tp1_d, tp2_d = _live_dists(
                asset, float(row["ATR"]), SL_MULT, TP1_MULT, TP2_MULT)
            slots[key] = {
                "asset": asset, "slot": key, "layer": "swing", "dir": "BUY",
                "entry": close, "sl": close - sl_d, "tp1": close + tp1_d,
                "tp2": close + tp2_d, "entry_i": i, "strategy": label,
                "priority": key, "tp1_hit": False, "lots": _LOTS[asset],
                "closed_lots": 0, "pnl_partial": 0,
                "sl_dist": sl_d, "tp1_dist": tp1_d, "tp2_dist": tp2_d,
            }
            open_count += 1
    censored = sum(1 for v in slots.values() if v is not None)

    out = {"run_at": now_str,
           "window": {"unified": [str(unified[0]), str(unified[-1]), n_bars],
                      "0050_raw": len(df_0050), "2330_raw": len(df_2330)},
           "mid_i": mid_i, "cut_date": cut_date,
           "max_open": MAX_OPEN, "skipped_by_cap": skipped,
           "censored": censored}
    # 分資產層統計（半段以對齊後中點切分）
    layers = {}
    for asset in ("SRF", "QFF"):
        lt = [t for t in trades if t["asset"] == asset]
        layers[asset] = _layer_stats(lt, mid_i)
    # 逐策略拆解
    breakdown = {}
    for asset, key, algo_id, label, param in SLOTS_SPEC:
        st = [t for t in trades if t["slot"] == key]
        first = [x for x in st if x["entry_i"] < mid_i]
        second = [x for x in st if x["entry_i"] >= mid_i]
        breakdown[key] = {
            "total": len(st),
            "wins": sum(1 for x in st if x["pnl_ntd"] > 0),
            "avg_ntd": round(sum(x["pnl_ntd"] for x in st) / len(st), 0) if st else 0.0,
            "net_ntd": sum(x["pnl_ntd"] for x in st),
            "first_n": len(first),
            "first_avg": round(sum(x["pnl_ntd"] for x in first) / len(first), 0) if first else 0.0,
            "second_n": len(second),
            "second_avg": round(sum(x["pnl_ntd"] for x in second) / len(second), 0) if second else 0.0,
        }
    # 合計（不計半段，兩資產共用同一時間軸但半段判定以分資產為準）
    tot_n = len(trades)
    tot_wins = sum(1 for t in trades if t["pnl_ntd"] > 0)
    win_pts = [t["pnl_pts"] for t in trades if t["pnl_ntd"] > 0]
    loss_pts = [abs(t["pnl_pts"]) for t in trades if t["pnl_ntd"] <= 0]
    avg_w = sum(win_pts) / len(win_pts) if win_pts else 0.0
    avg_l = sum(loss_pts) / len(loss_pts) if loss_pts else 0.0
    combined = {
        "total": tot_n, "wins": tot_wins,
        "win_rate": round(tot_wins / tot_n * 100.0, 1) if tot_n else 0.0,
        "avg_ntd": round(sum(t["pnl_ntd"] for t in trades) / tot_n, 0) if tot_n else 0.0,
        "rr": round(avg_w / avg_l, 2) if avg_l > 0 else 0.0,
        "net_ntd": sum(t["pnl_ntd"] for t in trades),
    }
    out["layers"] = layers
    out["breakdown"] = breakdown
    out["combined"] = combined
    out["trades"] = trades

    print("=== 穩健版 v1 現股模擬（60d 15m spot proxy，本機測試）===")
    print(f"run_at: {now_str}")
    print(f"window unified: {unified[0]} -> {unified[-1]} ({n_bars} bars; "
          f"0050 raw {len(df_0050)}, 2330 raw {len(df_2330)}); mid cut {cut_date}")
    print(f"cap: max {MAX_OPEN} open; skipped by cap: {skipped}; censored: {censored}")
    for asset in ("SRF", "QFF"):
        s = layers[asset]
        print(f"{asset} stable: n={s['total']} win={s['win_rate']}% avg={s['avg_ntd']:,.0f} "
              f"RR={s['rr']} net={s['net_ntd']:+,} 1H={s['first_avg_ntd']:,.0f} "
              f"2H={s['second_avg_ntd']:,.0f} -> {_variant_verdict(s)}")
    for key, b in breakdown.items():
        print(f"  {key}: n={b['total']} win={b['wins']} avg={b['avg_ntd']:,.0f} "
              f"net={b['net_ntd']:+,} 1H={b['first_n']}/{b['first_avg']:,.0f} "
              f"2H={b['second_n']}/{b['second_avg']:,.0f}")
    print(f"COMBINED: n={combined['total']} win={combined['win_rate']}% "
          f"avg={combined['avg_ntd']:,.0f} RR={combined['rr']} net={combined['net_ntd']:+,}")
    with open("stable_sim_v1.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote stable_sim_v1.json")


if __name__ == "__main__":
    run_stable_sim_v1()
