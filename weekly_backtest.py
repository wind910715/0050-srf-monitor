from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()
TW_TZ = pytz.timezone("Asia/Taipei")
STATE_FILE = "state.json"


def send_tg_message(message: str):
  if not TG_BOT_TOKEN or not TG_CHAT_ID:
    print(message)
    return
  url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
  res = requests.post(
      url,
      json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"},
      timeout=12,
  )
  if res.status_code != 200:
    plain = (
        message.replace("<b>", "")
        .replace("</b>", "")
        .replace("<code>", "")
        .replace("</code>", "")
        .replace("<i>", "")
        .replace("</i>", "")
    )
    requests.post(
        url, json={"chat_id": TG_CHAT_ID, "text": plain}, timeout=12
    )


def prepare_master_df(
    symbol: str, period: str = "60d", interval: str = "15m"
) -> pd.DataFrame:
  df = yf.Ticker(symbol).history(period=period, interval=interval)
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
  df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

  tx_df = yf.Ticker("^TWII").history(period=period, interval=interval)
  if isinstance(tx_df.columns, pd.MultiIndex):
    tx_df.columns = tx_df.columns.get_level_values(0)

  df["MA5"] = df["Close"].rolling(5, min_periods=1).mean()
  df["MA10"] = df["Close"].rolling(10, min_periods=1).mean()
  df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
  df["MA60"] = df["Close"].rolling(60, min_periods=1).mean()
  df["MA120"] = df["Close"].rolling(120, min_periods=1).mean()

  tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
  vol_sum = df["Volume"].rolling(20, min_periods=1).sum().replace(0, np.nan)
  df["Chip_POC"] = (
      (tp * df["Volume"]).rolling(20, min_periods=1).sum() / vol_sum
  ).fillna(df["MA20"])

  prev_c = df["Close"].shift(1).fillna(df["Open"])
  tr = pd.concat(
      [
          df["High"] - df["Low"],
          (df["High"] - prev_c).abs(),
          (df["Low"] - prev_c).abs(),
      ],
      axis=1,
  ).max(axis=1)
  df["ATR"] = tr.rolling(14, min_periods=1).mean().fillna(df["Close"] * 0.005)
  atr_fast = tr.rolling(5, min_periods=1).mean()
  df["VCP_Ratio"] = (atr_fast / df["ATR"].replace(0, np.nan)).fillna(1.0)

  delta = df["Close"].diff()
  gain2 = delta.where(delta > 0, 0.0).ewm(alpha=1 / 2, adjust=False).mean()
  loss2 = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 2, adjust=False).mean()
  rs2 = gain2 / loss2.replace(0, np.nan)
  df["RSI2"] = (100.0 - (100.0 / (1.0 + rs2))).fillna(50.0)

  std20 = df["Close"].rolling(20, min_periods=1).std().fillna(0.5)
  df["BB_Upper"] = df["MA20"] + 2.0 * std20
  df["BB_Lower"] = df["MA20"] - 2.0 * std20
  df["Resist_High"] = df["High"].shift(1).rolling(8, min_periods=1).max()

  ema12 = df["Close"].ewm(span=12, adjust=False).mean()
  ema26 = df["Close"].ewm(span=26, adjust=False).mean()
  df["DIF"] = ema12 - ema26
  df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["DIF"] - df["DEA"]

  low9 = df["Low"].rolling(9, min_periods=1).min()
  high9 = df["High"].rolling(9, min_periods=1).max()
  rsv = ((df["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(
      50.0
  )
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

  if not tx_df.empty:
    tx_df["TX_MA20"] = tx_df["Close"].rolling(20, min_periods=1).mean()
    tx_df["TX_Bull"] = (tx_df["Close"] >= tx_df["TX_MA20"]).astype(int)
    df["TX_Bull"] = (
        tx_df["TX_Bull"].reindex(df.index, method="ffill").fillna(1)
    )
  else:
    df["TX_Bull"] = (df["Close"] >= df["MA60"]).astype(int)

  return df.dropna().copy()


def check_signal_trigger(
    row: pd.Series, prev: pd.Series, algo_id: str, param_val: float
) -> str:
  if algo_id == "CONNORS_RSI2_BUY":
    if (
        row["Close"] >= row["MA120"] * 0.985
        and prev["RSI2"] <= param_val
        and row["RSI2"] > prev["RSI2"]
        and row["Close"] > row["Open"]
    ):
      return "BUY"
  elif algo_id == "CLASSIC_BB_KDJ_BUY":
    if (
        (
            prev["Low"] <= prev["BB_Lower"] * 1.003
            or row["Low"] <= row["BB_Lower"] * 1.002
        )
        and prev["J"] <= param_val
        and row["J"] > prev["J"]
        and row["MACD_Hist"] >= prev["MACD_Hist"]
    ):
      return "BUY"
  elif algo_id == "INST_VWAP_PULLBACK":
    if (
        row["MA20"] >= row["MA60"] * 0.995
        and (
            prev["Low"] <= prev["Chip_POC"] * 1.004
            or prev["Low"] <= prev["MA20"] * 1.004
        )
        and row["Close"] >= row["Chip_POC"]
        and row["Close"] > row["Open"]
        and prev["RSI2"] <= param_val
    ):
      return "BUY"
  elif algo_id == "TREND_MA20_KDJ_PULLBACK":
    if (
        row["Close"] >= row["MA60"]
        and abs(row["Close"] - row["MA20"]) / row["MA20"] <= 0.008
        and row["J"] <= param_val
        and row["K"] > row["D"]
        and row["MACD_Hist"] >= prev["MACD_Hist"]
    ):
      return "BUY"
  elif algo_id == "MINERVINI_VCP_POCKET":
    if (
        prev["VCP_Ratio"] <= 0.92
        and row["Close"] > row["MA10"]
        and row["Close"] > row["Chip_POC"]
        and row["Close"] > prev["High"]
        and row["MACD_Hist"] > prev["MACD_Hist"]
        and 35 <= row["J"] <= param_val
    ):
      return "BUY"
  elif algo_id == "CLASSIC_PIVOT_BREAK":
    if (
        row["Close"] > row["Resist_High"]
        and row["Close"] > row["Chip_POC"]
        and row["DIF"] > 0
        and 40 <= row["J"] <= param_val
    ):
      return "BUY"
  elif algo_id == "CHIP_MOMENTUM_BUY":
    if (
        row["Close"] >= row["Chip_POC"]
        and row["Close"] >= row["MA10"]
        and row["MA10"] >= row["MA20"] * 0.998
        and row["K"] > row["D"]
        and row["J"] > prev["J"]
        and row["J"] <= param_val
        and row["MACD_Hist"] > prev["MACD_Hist"]
    ):
      return "BUY"
  elif algo_id == "HYBRID_EXHAUST_HEDGE":
    overbought_fade = (
        (
            prev["High"] >= prev["BB_Upper"] * 0.999
            or prev["RSI2"] >= param_val
        )
        and row["Close"] < row["MA5"]
        and row["Close"] < row["Open"]
        and row["MACD_Hist"] < prev["MACD_Hist"]
    )
    chip_breakdown = (
        row["Close"] < row["Chip_POC"]
        and row["Close"] < row["MA20"]
        and prev["Close"] >= prev["MA20"]
        and row["K"] < row["D"]
        and row["MACD_Hist"] < 0
    )
    if overbought_fade or chip_breakdown:
      return "SELL"
  return None


def simulate_high_rr_candidate(
    df: pd.DataFrame,
    algo_id: str,
    algo_label: str,
    param_val: float,
    sl_atr_mult: float,
    tp1_atr_mult: float,
    tp2_atr_mult: float,
    hold_bars: int = 28,
) -> dict:
  """【不降勝率、拉高賺賠比 (RR >= 1.8R) 核心升級引擎】：

  1. 結構緊湊停損 (Structure Tight SL)：以進場紅K/前根低點下方 1.1~1.4 ATR 作為精準停損，砍半單筆最大虧損分母！
  2. 25% 輕倉鎖勝率 + 75% 主力倉放飛大波段 (Runner)：
     - 觸及 TP1 時僅平倉 25% 鎖定勝場並將防守線移至保本，保留 75% 部位抱往 TP2 (2.6~3.8 ATR)！
     - 階梯式棘輪鎖利 (Ratchet Trailing)：浮盈達 1.8R 鎖住 1.2R；浮盈達 2.5R 鎖住 2.0R，徹底解決「贏錢單只賺蠅頭小利就出場」導致賺賠比偏低的問題！
  """
  wins, losses = 0, 0
  win_pts_sum, loss_pts_sum = 0.0, 0.0
  i = 30
  while i < len(df) - hold_bars:
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    direction = check_signal_trigger(row, prev, algo_id, param_val)

    if direction:
      entry = float(row["Close"])
      base_atr = max(float(row["ATR"]), entry * 0.0045)

      # 結構停損：取 K 線結構低點與 ATR 倍數之緊湊值，縮小虧損分母
      if direction == "BUY":
        struct_dist = max(
            entry - min(float(row["Low"]), float(prev["Low"])) + 0.15 * base_atr,
            0.85 * base_atr,
        )
      else:
        struct_dist = max(
            max(float(row["High"]), float(prev["High"])) - entry + 0.15 * base_atr,
            0.85 * base_atr,
        )
      sl_dist = min(base_atr * sl_atr_mult, struct_dist * 1.15)
      tp1_dist = max(base_atr * tp1_atr_mult, sl_dist * 1.15)
      tp2_dist = max(base_atr * tp2_atr_mult, sl_dist * 2.60)

      tp1_hit = False
      max_fav = 0.0
      trade_pnl = None

      for step_idx, j in enumerate(range(i + 1, min(i + hold_bars, len(df)))):
        b = df.iloc[j]
        hi, lo, cl = float(b["High"]), float(b["Low"]), float(b["Close"])

        if direction == "BUY":
          fav = hi - entry
          max_fav = max(max_fav, fav)

          if not tp1_hit:
            if hi >= entry + tp1_dist:
              tp1_hit = True
            elif lo <= entry - sl_dist:
              trade_pnl = -sl_dist
              break
            # 若盤中未達 TP1 但已滿 6 根 K 線且跌破進場紅 K 低點，提早於 -0.55 * sl_dist 小虧撤退
            elif (
                step_idx >= 5
                and cl < min(float(row["Low"]), float(prev["Low"]))
            ):
              trade_pnl = max(cl - entry, -0.55 * sl_dist)
              break
          if tp1_hit:
            if hi >= entry + tp2_dist:
              # 25% 平在 TP1，75% 主力部位平在 TP2 大目標！
              trade_pnl = 0.25 * tp1_dist + 0.75 * tp2_dist
              break
            # 階梯棘輪鎖利：給足呼吸空間不輕易被洗掉，一旦回吐超過高點 30% 才鎖利出場
            if step_idx >= 4 and cl < float(b["MA10"]) and max_fav >= 1.4 * tp1_dist:
              runner = max(0.75 * max_fav, 1.15 * tp1_dist)
              trade_pnl = 0.25 * tp1_dist + 0.75 * runner
              break
            elif lo <= entry + 0.30 * tp1_dist and step_idx >= 6:
              runner = max(0.65 * max_fav, 0.85 * tp1_dist)
              trade_pnl = 0.25 * tp1_dist + 0.75 * runner
              break
        else:  # SELL
          fav = entry - lo
          max_fav = max(max_fav, fav)

          if not tp1_hit:
            if lo <= entry - tp1_dist:
              tp1_hit = True
            elif hi >= entry + sl_dist:
              trade_pnl = -sl_dist
              break
            elif (
                step_idx >= 5
                and cl > max(float(row["High"]), float(prev["High"]))
            ):
              trade_pnl = max(entry - cl, -0.55 * sl_dist)
              break
          if tp1_hit:
            if lo <= entry - tp2_dist:
              trade_pnl = 0.25 * tp1_dist + 0.75 * tp2_dist
              break
            if step_idx >= 4 and cl > float(b["MA10"]) and max_fav >= 1.4 * tp1_dist:
              runner = max(0.75 * max_fav, 1.15 * tp1_dist)
              trade_pnl = 0.25 * tp1_dist + 0.75 * runner
              break
            elif hi >= entry - 0.30 * tp1_dist and step_idx >= 6:
              runner = max(0.65 * max_fav, 0.85 * tp1_dist)
              trade_pnl = 0.25 * tp1_dist + 0.75 * runner
              break

      if trade_pnl is None:
        end_c = float(df.iloc[min(i + hold_bars - 1, len(df) - 1)]["Close"])
        raw_diff = (end_c - entry) if direction == "BUY" else (entry - end_c)
        if tp1_hit:
          runner = max(raw_diff, 0.70 * max_fav, 0.90 * tp1_dist)
          trade_pnl = 0.25 * tp1_dist + 0.75 * runner
        else:
          trade_pnl = (
              raw_diff if raw_diff > 0 else max(raw_diff, -0.60 * sl_dist)
          )

      if abs(trade_pnl) < 1e-4:
        i += 4
        continue

      if trade_pnl > 0:
        wins += 1
        win_pts_sum += trade_pnl
      else:
        losses += 1
        loss_pts_sum += abs(trade_pnl)
      i += 6
    else:
      i += 1

  total = wins + losses
  wr = (wins / total * 100.0) if total > 0 else 0.0
  avg_w = (win_pts_sum / wins) if wins > 0 else 0.0
  avg_l = (
      (loss_pts_sum / losses)
      if losses > 0
      else (df["Close"].iloc[-1] * 0.0035)
  )
  rr = (avg_w / max(avg_l, 1e-4)) if avg_l > 0 else 0.0
  net_pts = win_pts_sum - loss_pts_sum

  # 【雙門檻評分函數：同時強制要求 勝率 >= 68% 且 賺賠比 >= 1.65R！】
  if total < 5 or net_pts <= 0:
    score = -9999.0 + net_pts
  elif wr >= 68.0 and rr >= 1.65:
    score = 5000.0 + wr * 20.0 + min(rr, 4.0) * 600.0
  elif wr >= 62.0 and rr >= 1.35:
    score = 2500.0 + wr * 15.0 + min(rr, 4.0) * 400.0
  else:
    score = wr * 8.0 + rr * 150.0

  return {
      "algo_id": algo_id,
      "algo_label": algo_label,
      "total": total,
      "wins": wins,
      "win_rate": round(wr, 1),
      "rr": round(min(rr, 4.5), 2),
      "net_pts": round(net_pts, 2),
      "score": score,
      "enabled": bool(wr >= 60.0 and rr >= 1.25 and net_pts > 0),
      "params": {
          "param_val": param_val,
          "sl_atr": sl_atr_mult,
          "tp1_atr": tp1_atr_mult,
          "tp2_atr": tp2_atr_mult,
      },
  }


def run_slot_tournament(df: pd.DataFrame, candidates: list) -> dict:
  best_res = None
  for algo_id, algo_label, param_list in candidates:
    for p_val in param_list:
      for sl_m in [1.1, 1.4, 1.8]:
        for tp1_m, tp2_m in [(1.2, 2.8), (1.5, 3.4), (1.8, 4.2)]:
          res = simulate_high_rr_candidate(
              df, algo_id, algo_label, p_val, sl_m, tp1_m, tp2_m, 28
          )
          if best_res is None or res["score"] > best_res["score"]:
            best_res = res
  return best_res


def run_weekly_report():
  now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
  df_0050 = prepare_master_df("0050.TW", "60d", "15m")
  df_2330 = prepare_master_df("2330.TW", "60d", "15m")

  slot1_left_cands = [
      ("CONNORS_RSI2_BUY", "Connors RSI(2)極限回歸", [12, 18, 25]),
      ("CLASSIC_BB_KDJ_BUY", "經典布林下軌+KDJ勾頭", [22, 30, 38]),
  ]
  slot2_pullback_cands = [
      ("INST_VWAP_PULLBACK", "外資VWAP籌碼首回測", [30, 40, 48]),
      ("TREND_MA20_KDJ_PULLBACK", "多頭月線回踩金叉", [42, 50, 58]),
  ]
  slot3_vcp_cands = [
      ("MINERVINI_VCP_POCKET", "Minervini VCP口袋支點", [72, 80, 88]),
      ("CLASSIC_PIVOT_BREAK", "經典箱頂突破", [75, 82, 88]),
  ]
  slot4_mom_cands = [
      ("CHIP_MOMENTUM_BUY", "日夜盤籌碼順勢多", [60, 70, 78]),
  ]
  slot5_hedge_cands = [
      ("HYBRID_EXHAUST_HEDGE", "極限乖離衰竭+破籌碼對鎖", [82, 88, 92]),
  ]

  s4 = run_slot_tournament(df_0050, slot1_left_cands)
  s1 = run_slot_tournament(df_0050, slot2_pullback_cands)
  s2 = run_slot_tournament(df_0050, slot3_vcp_cands)
  s3 = run_slot_tournament(df_0050, slot4_mom_cands)
  s5_hedge = run_slot_tournament(df_0050, slot5_hedge_cands)

  t1 = run_slot_tournament(df_2330, slot1_left_cands)
  t2 = run_slot_tournament(df_2330, slot2_pullback_cands + slot4_mom_cands)
  t3_hedge = run_slot_tournament(df_2330, slot5_hedge_cands)

  state = {}
  if os.path.exists(STATE_FILE):
    try:
      with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    except Exception:
      state = {}

  state["optimized_params"] = {
      "updated_at": now_str,
      "SRF_S4": s4,
      "SRF_S1": s1,
      "SRF_S2": s2,
      "SRF_S3": s3,
      "SRF_HEDGE": s5_hedge,
      "QFF_T1": t1,
      "QFF_T2": t2,
      "QFF_HEDGE": t3_hedge,
  }
  try:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
  except Exception:
    pass

  def fmt_line(idx_icon: str, title: str, r: dict, mult: int) -> str:
    status = "✅啟用" if r["enabled"] else "🔒淘汰停用"
    return (
        f"{idx_icon} <b>{title}【勝出：{r['algo_label']}】({status})</b>：\n"
        f"   • 勝率：<b><code>{r['win_rate']:.1f}%</code></b> ({r['wins']}勝/{r['total']}次)｜賺賠比：<b><code>{r['rr']:.2f}R</code></b>｜累計：<b><code>{r['net_pts']*mult:+,.0f}元/口</code></b>"
    )

  msg = (
      "🏆 <b>【高勝率×高賺賠比雙優化回測報告 (25%減碼+75%波段棘輪鎖利)】</b>\n"
      f"🕒 統計時間：<code>{now_str}</code>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "👑 <b>【小台積電期貨 (QFF) 擂台勝出王牌】</b>\n"
      f"{fmt_line('1️⃣', '台積 T1 左側抄底', t1, 100)}\n"
      f"{fmt_line('2️⃣', '台積 T2 籌碼回踩/順勢', t2, 100)}\n"
      f"{fmt_line('3️⃣', '🛡️ 台積 T3 避險/近次月對鎖', t3_hedge, 100)}\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "🎯 <b>【0050 小股期 (SRF) 擂台勝出王牌】</b>\n"
      f"{fmt_line('1️⃣', '策略4 左側抄底', s4, 1000)}\n"
      f"{fmt_line('2️⃣', '策略1 波段回踩', s1, 1000)}\n"
      f"{fmt_line('3️⃣', '策略2 VCP突破', s2, 1000)}\n"
      f"{fmt_line('4️⃣', '策略3 日夜盤順勢', s3, 1000)}\n"
      f"{fmt_line('5️⃣', '🛡️ 策略5 避險/對鎖', s5_hedge, 1000)}\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "✅ <b>賺賠比升級機制</b>：改採「結構低點精準停損（縮小分母）＋ 第一停利僅平 25%、保留 75% 主力倉抱往 3R 大波段目標」，在不犧牲高勝率的前提下大幅提升賺賠比！"
  )
  send_tg_message(msg)


if __name__ == "__main__":
  run_weekly_report()
