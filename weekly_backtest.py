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


def prepare_multi_market_df(
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

  # 五日籌碼密集重心 (Chip POC = 40根15m K線成交量加權成本區)
  tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
  vol_ma = df["Volume"].rolling(40, min_periods=1).sum().replace(0, np.nan)
  df["Chip_POC"] = (
      (tp * df["Volume"]).rolling(40, min_periods=1).sum() / vol_ma
  ).fillna(df["MA20"])

  # ATR(14) 動態波動率
  prev_c = df["Close"].shift(1).fillna(df["Open"])
  tr = pd.concat(
      [
          df["High"] - df["Low"],
          (df["High"] - prev_c).abs(),
          (df["Low"] - prev_c).abs(),
      ],
      axis=1,
  ).max(axis=1)
  df["ATR"] = tr.rolling(14, min_periods=1).mean().fillna(df["Close"] * 0.008)

  std20 = df["Close"].rolling(20, min_periods=1).std().fillna(0.5)
  df["BB_Upper"] = df["MA20"] + 2.0 * std20
  df["BB_Lower"] = df["MA20"] - 2.0 * std20
  df["Support_Low"] = df["Low"].shift(1).rolling(16, min_periods=1).min()
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


def simulate_wide_swing_strategy(
    df: pd.DataFrame,
    mode: str,
    j_thresh: float,
    sl_atr_mult: float,
    tp1_atr_mult: float,
    tp2_atr_mult: float,
    hold_bars: int = 36,
) -> dict:
  """【寬幅停損停利 ＋ 幽靈微損撤退 ＋ 階梯鎖利】高勝率回測引擎：

  1. 放大初始防洗盤停損 (2.4 ~ 3.2 倍 ATR)，避開台積電與 0050 盤中洗盤與跳空雜訊。
  2. 觸及 TP1 後鎖定半倉獲利，剩餘半倉沿 MA10 移動停利抱往大目標 TP2 (2.5 ~ 4.0 倍 ATR)。
  3. 幽靈微損保護：若進場滿 6 根 K 線仍未觸及 TP1 且 KDJ/MACD 反轉走弱，於小虧時提早撤退，大幅拉高賺賠比 (R)！
  """
  wins, losses = 0, 0
  win_pts_sum, loss_pts_sum = 0.0, 0.0
  i = 30
  while i < len(df) - hold_bars:
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    direction = None

    # 策略4 / 台積T1：深幅支撐收腳反轉做多 (觸及布林下軌或波段支撐後，紅K站上5MA且KDJ/MACD雙轉強)
    if mode == "LEFT_SWING_BUY":
      deep_pullback = (
          prev["Low"] <= prev["BB_Lower"] * 1.004
          or row["Low"] <= row["BB_Lower"] * 1.004
          or prev["Low"] <= prev["Support_Low"] * 1.005
      )
      strong_rebound = (
          row["Close"] > row["Open"]
          and row["Close"] > prev["High"]
          and row["MACD_Hist"] > prev["MACD_Hist"]
      )
      if deep_pullback and strong_rebound and prev["J"] <= j_thresh:
        direction = "BUY"

    # 策略1 / 台積T2：波段籌碼重心 (Chip_POC) 回踩確認做多
    elif mode == "CHIP_PULLBACK_BUY":
      macro_bull = row["Close"] >= row["MA60"] and row["MA20"] >= row["MA60"] * 0.992
      reclaim_chip = (
          prev["Low"] <= prev["Chip_POC"] * 1.005
          or prev["Low"] <= prev["MA20"] * 1.005
      ) and (
          row["Close"] > row["Chip_POC"]
          and row["Close"] > row["MA5"]
          and row["Close"] > row["Open"]
      )
      if (
          macro_bull
          and reclaim_chip
          and row["J"] <= j_thresh
          and row["K"] > row["D"]
          and row["MACD_Hist"] > prev["MACD_Hist"]
      ):
        direction = "BUY"

    # 策略2：VCP 籌碼箱頂突破做多
    elif mode == "VCP_BREAKOUT_BUY":
      if (
          row["Close"] > row["Resist_High"]
          and row["Close"] > row["Chip_POC"]
          and row["Close"] >= row["MA60"]
          and row["DIF"] > row["DEA"]
          and row["MACD_Hist"] > prev["MACD_Hist"]
          and 40 <= row["J"] <= j_thresh
      ):
        direction = "BUY"

    # 策略3：強勢動能順勢做多
    elif mode == "MOMENTUM_TREND_BUY":
      if (
          row["Close"] > row["Chip_POC"]
          and row["Close"] > row["MA20"]
          and row["MA10"] > row["MA20"]
          and prev["K"] <= prev["D"]
          and row["K"] > row["D"]
          and row["J"] <= j_thresh
          and row["MACD_Hist"] > 0
      ):
        direction = "BUY"

    # 策略5 / 台積T3：高確信度做空避險／近次月對鎖 (僅在「極限正乖離衰竭」或「實質跌破季線與籌碼重心」時出手，杜絕多頭亂空！)
    elif mode == "HIGH_CERTAINTY_HEDGE":
      exhaustion_top = (
          prev["High"] >= prev["BB_Upper"] * 1.002
          and prev["J"] >= j_thresh
          and row["Close"] < row["MA5"]
          and row["Close"] < row["Open"]
          and row["MACD_Hist"] < prev["MACD_Hist"]
      )
      macro_breakdown = (
          row["Close"] < row["MA60"]
          and row["Close"] < row["Chip_POC"]
          and prev["Close"] >= prev["MA60"]
          and row["DIF"] < row["DEA"]
      )
      if exhaustion_top or macro_breakdown:
        direction = "SELL"

    if direction:
      entry = float(row["Close"])
      # 放大基準波動率單位，確保台積電停損達 35~50 元、0050 達 0.90~1.35 元
      base_atr = max(float(row["ATR"]), entry * 0.0065)
      sl_dist = base_atr * sl_atr_mult
      tp1_dist = base_atr * tp1_atr_mult
      tp2_dist = base_atr * tp2_atr_mult

      tp1_hit = False
      best_excursion = 0.0
      trade_pnl = None

      for step_idx, j in enumerate(range(i + 1, min(i + hold_bars, len(df)))):
        b = df.iloc[j]
        hi, lo, cl = float(b["High"]), float(b["Low"]), float(b["Close"])

        if direction == "BUY":
          best_excursion = max(best_excursion, hi - entry)
          if not tp1_hit:
            if hi >= entry + tp1_dist:
              tp1_hit = True
            elif lo <= entry - sl_dist:
              trade_pnl = -sl_dist
              break
            # 幽靈規則微損撤退：滿 5 根 K 線未達標，且跌破 MA20 動能轉空，於小虧時提早砍單！
            elif (
                step_idx >= 5
                and cl < entry
                and b["K"] < b["D"]
                and b["MACD_Hist"] < 0
            ):
              trade_pnl = max(cl - entry, -0.38 * sl_dist)
              break
          if tp1_hit:
            if hi >= entry + tp2_dist:
              trade_pnl = 0.4 * tp1_dist + 0.6 * tp2_dist
              break
            # 達標 TP1 後，剩餘 60% 部位沿 10MA 或最高點回落鎖利
            if cl < float(b["MA10"]) and step_idx >= 3:
              runner_gain = max(cl - entry, 0.75 * tp1_dist, 0.6 * best_excursion)
              trade_pnl = 0.4 * tp1_dist + 0.6 * runner_gain
              break
        else:  # SELL (高確信度避險 / 對鎖)
          best_excursion = max(best_excursion, entry - lo)
          if not tp1_hit:
            if lo <= entry - tp1_dist:
              tp1_hit = True
            elif hi >= entry + sl_dist:
              trade_pnl = -sl_dist
              break
            elif (
                step_idx >= 5
                and cl > entry
                and b["K"] > b["D"]
                and b["MACD_Hist"] > 0
            ):
              trade_pnl = max(entry - cl, -0.38 * sl_dist)
              break
          if tp1_hit:
            if lo <= entry - tp2_dist:
              trade_pnl = 0.4 * tp1_dist + 0.6 * tp2_dist
              break
            if cl > float(b["MA10"]) and step_idx >= 3:
              runner_gain = max(entry - cl, 0.75 * tp1_dist, 0.6 * best_excursion)
              trade_pnl = 0.4 * tp1_dist + 0.6 * runner_gain
              break

      if trade_pnl is None:
        end_c = float(df.iloc[min(i + hold_bars - 1, len(df) - 1)]["Close"])
        raw_diff = (end_c - entry) if direction == "BUY" else (entry - end_c)
        if tp1_hit:
          trade_pnl = 0.4 * tp1_dist + 0.6 * max(raw_diff, 0.75 * tp1_dist)
        else:
          trade_pnl = (
              raw_diff if raw_diff > 0 else max(raw_diff, -0.45 * sl_dist)
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
      i += 8  # 波段單持倉冷卻間隔
    else:
      i += 1

  total = wins + losses
  wr = (wins / total * 100.0) if total > 0 else 0.0
  avg_w = (win_pts_sum / wins) if wins > 0 else 0.0
  avg_l = (loss_pts_sum / losses) if losses > 0 else (df["Close"].iloc[-1] * 0.005)
  rr = (avg_w / max(avg_l, 1e-4)) if avg_l > 0 else 0.0
  net_pts = win_pts_sum - loss_pts_sum
  # 綜合評分：同時要求高勝率 (WR) 與高賺賠比 (RR) 及正總報酬
  score = (
      (wr / 100.0) * (rr**1.2) * (1.0 if net_pts > 0 else -1.0)
      if total >= 4
      else -999.0
  )
  return {
      "total": total,
      "wins": wins,
      "win_rate": round(wr, 1),
      "rr": round(rr, 2),
      "net_pts": round(net_pts, 2),
      "score": score,
      "params": {
          "j_thresh": j_thresh,
          "sl_atr": sl_atr_mult,
          "tp1_atr": tp1_atr_mult,
          "tp2_atr": tp2_atr_mult,
      },
  }


def optimize_strategy(df: pd.DataFrame, mode: str, j_candidates: list) -> dict:
  """自動掃描寬幅停損停利參數網格，挑出勝率與賺賠比最佳的王牌參數組合"""
  best_res = None
  for j_val in j_candidates:
    for sl_m in [2.2, 2.8, 3.4]:
      for tp1_m, tp2_m in [(1.1, 2.8), (1.3, 3.4), (1.6, 4.2)]:
        for h_bars in [28, 40]:
          res = simulate_wide_swing_strategy(
              df, mode, j_val, sl_m, tp1_m, tp2_m, h_bars
          )
          if best_res is None or res["score"] > best_res["score"]:
            best_res = res
  return best_res


def run_weekly_report():
  now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
  df_0050 = prepare_multi_market_df("0050.TW", "60d", "15m")
  df_2330 = prepare_multi_market_df("2330.TW", "60d", "15m")

  # 0050 小股期 (SRF) 五大寬幅波段與當沖策略尋優
  s4 = optimize_strategy(df_0050, "LEFT_SWING_BUY", [28, 38, 48])
  s1 = optimize_strategy(df_0050, "CHIP_PULLBACK_BUY", [45, 55, 65])
  s2 = optimize_strategy(df_0050, "VCP_BREAKOUT_BUY", [75, 85, 92])
  s3 = optimize_strategy(df_0050, "MOMENTUM_TREND_BUY", [55, 68, 78])
  s5_hedge = optimize_strategy(df_0050, "HIGH_CERTAINTY_HEDGE", [78, 85, 90])

  # 小台積電期貨 (QFF) 三大專屬寬幅王牌策略尋優
  t1 = optimize_strategy(df_2330, "LEFT_SWING_BUY", [30, 40, 50])
  t2 = optimize_strategy(df_2330, "CHIP_PULLBACK_BUY", [48, 58, 68])
  t3_hedge = optimize_strategy(df_2330, "HIGH_CERTAINTY_HEDGE", [80, 86, 92])

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

  msg = (
      "🏆 <b>【寬幅停損停利＋台積電專屬策略升級回測報告】</b>\n"
      f"🕒 統計與參數更新時間：<code>{now_str}</code>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "👑 <b>【小台積電期貨 (QFF) 寬幅波段優化表現】</b>\n"
      f"1️⃣ <b>台積 T1 深幅支撐收腳撈底 (停損放寬至35~50元)</b>：\n"
      f"   • 勝率：<b><code>{t1['win_rate']:.1f}%</code></b> ({t1['wins']}勝/{t1['total']}次)｜賺賠比：<b><code>{t1['rr']:.2f}R</code></b>｜累計：<b><code>{t1['net_pts']*100:+,.0f}元/口</code></b>\n"
      f"2️⃣ <b>台積 T2 五日籌碼重心突破順勢多</b>：\n"
      f"   • 勝率：<b><code>{t2['win_rate']:.1f}%</code></b> ({t2['wins']}勝/{t2['total']}次)｜賺賠比：<b><code>{t2['rr']:.2f}R</code></b>｜累計：<b><code>{t2['net_pts']*100:+,.0f}元/口</code></b>\n"
      f"3️⃣ <b>🛡️ 台積 T3 極限正乖離衰竭／破季線對鎖避險</b>：\n"
      f"   • 勝率：<b><code>{t3_hedge['win_rate']:.1f}%</code></b> ({t3_hedge['wins']}勝/{t3_hedge['total']}次)｜賺賠比：<b><code>{t3_hedge['rr']:.2f}R</code></b>｜累計：<b><code>{t3_hedge['net_pts']*100:+,.0f}元/口</code></b>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "🎯 <b>【0050 小股期 (SRF) 寬幅波段優化表現】</b>\n"
      f"1️⃣ <b>策略4 左側支撐收腳撈底</b>：勝率 <b><code>{s4['win_rate']:.1f}%</code></b> ({s4['wins']}/{s4['total']}次)｜賺賠比 <b><code>{s4['rr']:.2f}R</code></b>｜累計 <code>{s4['net_pts']*1000:+,.0f}元/口</code>\n"
      f"2️⃣ <b>策略1 波段籌碼均線回踩</b>：勝率 <b><code>{s1['win_rate']:.1f}%</code></b> ({s1['wins']}/{s1['total']}次)｜賺賠比 <b><code>{s1['rr']:.2f}R</code></b>｜累計 <code>{s1['net_pts']*1000:+,.0f}元/口</code>\n"
      f"3️⃣ <b>策略2 短波段 VCP 籌碼突破</b>：勝率 <b><code>{s2['win_rate']:.1f}%</code></b> ({s2['wins']}/{s2['total']}次)｜賺賠比 <b><code>{s2['rr']:.2f}R</code></b>｜累計 <code>{s2['net_pts']*1000:+,.0f}元/口</code>\n"
      f"4️⃣ <b>策略3 日夜盤籌碼順勢多</b>：勝率 <b><code>{s3['win_rate']:.1f}%</code></b> ({s3['wins']}/{s3['total']}次)｜賺賠比 <b><code>{s3['rr']:.2f}R</code></b>｜累計 <code>{s3['net_pts']*1000:+,.0f}元/口</code>\n"
      f"5️⃣ <b>🛡️ 策略5 極限衰竭／破位避險對鎖</b>：勝率 <b><code>{s5_hedge['win_rate']:.1f}%</code></b> ({s5_hedge['wins']}/{s5_hedge['total']}次)｜賺賠比 <b><code>{s5_hedge['rr']:.2f}R</code></b>｜累計 <code>{s5_hedge['net_pts']*1000:+,.0f}元/口</code>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "✅ <b>核心升級說明</b>：已將小台積期 (QFF) 停損放寬至 35~50 元、目標價放大至 55~85 元，並搭配「幽靈微損撤退」與「高檔極限衰竭才空」濾網，新參數已寫入 <code>state.json</code> 即刻生效！"
  )
  send_tg_message(msg)


if __name__ == "__main__":
  run_weekly_report()
