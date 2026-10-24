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


# =========================================================================
# A. 100% 鎖定維持不變的策略群 (台積 T2、0050 策略1 / 策略2 / 策略3 / 策略5)
# =========================================================================
def check_locked_signal(
    row: pd.Series, prev: pd.Series, algo_id: str, param_val: float
) -> str:
  if algo_id == "MINERVINI_VCP_POCKET":
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
  elif algo_id == "SNIPER_CHIP_MOMENTUM":
    if (
        row["TX_Bull"] == 1
        and row["Close"] >= row["Chip_POC"]
        and row["Close"] >= row["MA10"]
        and row["MA10"] >= row["MA20"]
        and prev["K"] <= prev["D"]
        and row["K"] > row["D"]
        and 25 <= row["J"] <= param_val
        and row["MACD_Hist"] > 0
        and row["MACD_Hist"] > prev["MACD_Hist"]
    ):
      return "BUY"
  elif algo_id == "SNIPER_VWAP_PULLBACK":
    if (
        row["Close"] >= row["MA60"]
        and row["MA20"] >= row["MA60"] * 0.996
        and (
            prev["Low"] <= prev["Chip_POC"] * 1.004
            or prev["Low"] <= prev["MA20"] * 1.004
        )
        and row["Close"] >= row["Chip_POC"]
        and row["Close"] >= row["MA5"]
        and row["Close"] > row["Open"]
        and row["K"] >= row["D"]
        and row["MACD_Hist"] >= prev["MACD_Hist"]
        and prev["RSI2"] <= param_val
    ):
      return "BUY"
  elif algo_id == "SNIPER_MA20_GOLD_PULLBACK":
    if (
        row["Close"] >= row["MA60"]
        and abs(row["Close"] - row["MA20"]) / row["MA20"] <= 0.008
        and row["Close"] >= row["MA20"]
        and row["J"] <= param_val
        and row["K"] > row["D"]
        and row["MACD_Hist"] > prev["MACD_Hist"]
    ):
      return "BUY"
  return None


# =========================================================================
# 對齊版出場引擎 (2026-10-24): 與 monitor_srf.py evaluate_urgent_exits 一致。
# 實盤規則: TP1 首次觸發只把停損移至 entry ± 0.30*TP1_dist (不減碼);
# 觸及 TP2 或停損線 (原始或上移後) 即全數平倉, 損益 = 實際價差 - 手續費 (50 元/口)。
# 無 MA runner、無動能停滯出場、無時間出場、無部分平倉、無虧損夾限。
# 殘差揭露: 實盤 25 秒輪詢 vs 回測 15 分 K——同根 K 內先後觸及停損與停利時,
# 回測採保守順序 (停損優先); 進場價取訊號 K 收盤 (實盤為觸發當下價)。
# =========================================================================
LIVE_DIST = {
    "SRF": {"atr_floor": 0.35, "tick": 0.05, "sl_min": 0.50, "tp1_min": 0.45, "tp2_min": 1.50, "fee_pts": 0.05},
    "QFF": {"atr_floor": 12.0, "tick": 5.0, "sl_min": 20.0, "tp1_min": 20.0, "tp2_min": 55.0, "fee_pts": 0.50},
}


def _live_dists(asset: str, atr: float, sl_mult: float, tp1_mult: float, tp2_mult: float):
  cfg = LIVE_DIST[asset]
  a = max(float(atr), cfg["atr_floor"])
  tick = cfg["tick"]

  def rnd(x: float) -> float:
    return round(x / tick) * tick

  return (
      max(rnd(a * sl_mult), cfg["sl_min"]),
      max(rnd(a * tp1_mult), cfg["tp1_min"]),
      max(rnd(a * tp2_mult), cfg["tp2_min"]),
  )


def simulate_live_exit(
    df: pd.DataFrame,
    signal_fn,
    algo_id: str,
    algo_label: str,
    param_val: float,
    asset: str,
    sl_mult: float = 1.25,
    tp1_mult: float = 1.05,
    tp2_mult: float = 3.50,
) -> dict:
  """與實盤一致的出場模擬。sl/tp mult 預設為實盤值, 僅供優化評估傳入變體。"""
  cfg = LIVE_DIST[asset]
  wins = losses = censored = 0
  win_pts_sum = loss_pts_sum = 0.0
  i = 30
  while i < len(df) - 1:
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    direction = signal_fn(row, prev, algo_id, param_val)
    if not direction:
      i += 1
      continue
    entry = float(row["Close"])
    sl_dist, tp1_dist, tp2_dist = _live_dists(
        asset, float(row["ATR"]), sl_mult, tp1_mult, tp2_mult
    )
    if direction == "BUY":
      sl_line = entry - sl_dist
      tp1 = entry + tp1_dist
      tp2 = entry + tp2_dist
    else:
      sl_line = entry + sl_dist
      tp1 = entry - tp1_dist
      tp2 = entry - tp2_dist
    tp1_hit = False
    trade_pnl = None
    exit_j = None
    for j in range(i + 1, len(df)):
      b = df.iloc[j]
      hi, lo = float(b["High"]), float(b["Low"])
      if direction == "BUY":
        if lo <= sl_line:
          trade_pnl = sl_line - entry
          exit_j = j
          break
        if hi >= tp2:
          trade_pnl = tp2 - entry
          exit_j = j
          break
        if not tp1_hit and hi >= tp1:
          tp1_hit = True
          sl_line = round(entry + 0.30 * tp1_dist, 2)
      else:
        if hi >= sl_line:
          trade_pnl = entry - sl_line
          exit_j = j
          break
        if lo <= tp2:
          trade_pnl = entry - tp2
          exit_j = j
          break
        if not tp1_hit and lo <= tp1:
          tp1_hit = True
          sl_line = round((entry - 0.30 * tp1_dist) / 5) * 5 if asset == "QFF" else round(entry - 0.30 * tp1_dist, 2)
    if trade_pnl is None:
      censored += 1
      break
    trade_pnl -= cfg["fee_pts"]
    if trade_pnl > 0:
      wins += 1
      win_pts_sum += trade_pnl
    else:
      losses += 1
      loss_pts_sum += abs(trade_pnl)
    i = exit_j + 1

  total = wins + losses
  wr = (wins / total * 100.0) if total > 0 else 0.0
  avg_w = (win_pts_sum / wins) if wins > 0 else 0.0
  avg_l = (loss_pts_sum / losses) if losses > 0 else 0.0
  rr = (avg_w / avg_l) if avg_l > 0 else 0.0
  net_pts = win_pts_sum - loss_pts_sum
  return {
      "algo_id": algo_id,
      "algo_label": algo_label,
      "total": total,
      "wins": wins,
      "win_rate": round(wr, 1),
      "rr": round(min(rr, 4.5), 2),
      "net_pts": round(net_pts, 2),
      "censored": censored,
      "sample_ok": total >= 30,
      "score": net_pts,
      "enabled": True,
      "params": {
          "param_val": param_val,
          "sl_mult": sl_mult,
          "tp1_mult": tp1_mult,
          "tp2_mult": tp2_mult,
          "asset": asset,
      },
  }


def run_locked_slot(df: pd.DataFrame, candidates: list, asset: str) -> dict:
  best_res = None
  for algo_id, algo_label, param_list in candidates:
    for p_val in param_list:
      res = simulate_live_exit(df, check_locked_signal, algo_id, algo_label, p_val, asset)
      if best_res is None or res["net_pts"] > best_res["net_pts"]:
        best_res = res
  return best_res


# =========================================================================
# B. 專項優化引擎：針對「台積 T1 (提昇賺賠比)」、「台積 T3 (解決勝率低)」、「0050 策略4 (提昇賺賠比)」
# =========================================================================
def check_targeted_upgrade_signal(
    row: pd.Series, prev: pd.Series, algo_id: str, param_val: float
) -> str:
  # 1. 台積 T1 & 0050 策略4 專屬：Connors RSI(2) / 布林下軌 W底吞噬確認 (高勝率 + 緊湊結構停損拉高賺賠比)
  if algo_id == "UPGRADE_RSI2_SWING":
    prev_mid = (float(prev["Open"]) + float(prev["Close"])) / 2.0
    if (
        row["Close"] >= row["MA120"] * 0.985
        and (
            prev["RSI2"] <= param_val
            or prev["Low"] <= prev["BB_Lower"] * 1.002
        )
        and row["Close"] > row["Open"]
        and row["Close"] > prev_mid
        and row["Close"] >= row["MA5"] * 0.999
        and row["MACD_Hist"] > prev["MACD_Hist"]
    ):
      return "BUY"

  elif algo_id == "UPGRADE_BB_W_BOTTOM":
    if (
        (
            prev["Low"] <= prev["BB_Lower"] * 1.003
            or row["Low"] <= row["BB_Lower"] * 1.002
        )
        and prev["J"] <= param_val
        and row["J"] > prev["J"]
        and row["Close"] > row["Open"]
        and row["Close"] > prev["Close"]
        and row["MACD_Hist"] > prev["MACD_Hist"]
    ):
      return "BUY"

  # 2. 台積 T3 避險/對鎖檢討重構：放棄滯後的「雙均線死叉」，改用「高檔極限正乖離衰竭首根轉弱」+「初破籌碼重心」
  elif algo_id == "UPGRADE_TSMC_CLIMAX_HEDGE":
    overbought_exhaust = (
        (
            prev["High"] >= prev["BB_Upper"] * 0.999
            or prev["RSI2"] >= param_val
            or prev["J"] >= 86
        )
        and row["Close"] < row["MA5"]
        and row["Close"] < row["Open"]
        and row["MACD_Hist"] < prev["MACD_Hist"]
    )
    first_chip_break = (
        prev["Close"] >= prev["Chip_POC"]
        and row["Close"] < row["Chip_POC"]
        and row["Close"] < row["MA20"]
        and row["K"] < row["D"]
        and row["MACD_Hist"] < 0
    )
    if overbought_exhaust or first_chip_break:
      return "SELL"

  return None


def run_targeted_slot(df: pd.DataFrame, candidates: list, asset: str) -> dict:
  best_res = None
  for algo_id, algo_label, param_list in candidates:
    for p_val in param_list:
      res = simulate_live_exit(df, check_targeted_upgrade_signal, algo_id, algo_label, p_val, asset)
      if best_res is None or res["net_pts"] > best_res["net_pts"]:
        best_res = res
  return best_res


def run_weekly_report():
  now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
  df_0050 = prepare_master_df("0050.TW", "60d", "15m")
  df_2330 = prepare_master_df("2330.TW", "60d", "15m")

  # 1. 維持不變的已啟用王牌策略
  s1 = run_locked_slot(
      df_0050,
      [
          ("SNIPER_VWAP_PULLBACK", "外資VWAP籌碼首回測確認", [30, 38, 45]),
          ("SNIPER_MA20_GOLD_PULLBACK", "多頭月線回踩金叉確認", [40, 48, 55]),
      ],
      "SRF",
  )
  s2 = run_locked_slot(
      df_0050,
      [
          ("MINERVINI_VCP_POCKET", "Minervini VCP口袋支點", [72, 80, 88]),
          ("CLASSIC_PIVOT_BREAK", "經典箱頂突破", [75, 82, 88]),
      ],
      "SRF",
  )
  s3 = run_locked_slot(
      df_0050, [("SNIPER_CHIP_MOMENTUM", "日夜盤籌碼順勢共振多", [55, 65, 72])], "SRF"
  )
  s5_hedge = run_locked_slot(
      df_0050,
      [("HYBRID_EXHAUST_HEDGE", "極限乖離衰竭+破籌碼對鎖", [82, 88, 92])],
      "SRF",
  )
  t2 = run_locked_slot(
      df_2330, [("SNIPER_CHIP_MOMENTUM", "日夜盤籌碼順勢共振多", [55, 65, 72])], "QFF"
  )

  # 2. 專項升級：小台積電 T1 (拉高賺賠比)、小台積電 T3 (解決勝率低)、小 0050 策略4 (拉高賺賠比)
  left_upgrade_cands = [
      ("UPGRADE_RSI2_SWING", "Connors RSI(2)極限收腳確認", [10, 14, 18, 22]),
      ("UPGRADE_BB_W_BOTTOM", "布林下軌W底雙確認", [18, 24, 30]),
  ]
  t1 = run_targeted_slot(df_2330, left_upgrade_cands, "QFF")
  t3_hedge = run_targeted_slot(
      df_2330,
      [("UPGRADE_TSMC_CLIMAX_HEDGE", "高檔極限衰竭+初破籌碼對鎖", [82, 86, 90])],
      "QFF",
  )
  s4 = run_targeted_slot(df_0050, left_upgrade_cands, "SRF")

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
    # 原子寫入: 避免寫到一半被中斷造成 state.json 截斷。
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STATE_FILE)
  except Exception as e:
    print(f"⚠️ 寫入 state.json 失敗: {e}")

  def fmt_line(idx_icon: str, title: str, r: dict, mult: int) -> str:
    if not r.get("sample_ok"):
      tag = "⚠️樣本不足(n<30)，僅參考"
    elif r.get("net_pts", 0) <= 0:
      tag = "⚠️誠實回測為負，建議停用"
    else:
      tag = "✅啟用"
    return (
        f"{idx_icon} <b>{title}【勝出：{r['algo_label']}】({tag})</b>：\n"
        f"   • 勝率：<b><code>{r['win_rate']:.1f}%</code></b> ({r['wins']}勝/{r['total']}次)｜賺賠比：<b><code>{r['rr']:.2f}R</code></b>｜累計：<b><code>{r['net_pts']*mult:+,.0f}元/口</code></b>"
    )

  msg = (
      "🏆 <b>【回測口徑對齊完成！實盤出場口徑誠實回測報告】</b>\n"
      f"🕒 統計時間：<code>{now_str}</code>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "👑 <b>【小台積電期貨 (QFF) 實盤回測】</b>\n"
      f"{fmt_line('1️⃣', '台積 T1 左側抄底 (賺賠比升級)', t1, 100)}\n"
      f"{fmt_line('2️⃣', '台積 T2 籌碼回踩/順勢 (原版鎖定)', t2, 100)}\n"
      f"{fmt_line('3️⃣', '🛡️ 台積 T3 避險/對鎖 (勝率重構升級)', t3_hedge, 100)}\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "🎯 <b>【0050 小股期 (SRF) 實盤回測】</b>\n"
      f"{fmt_line('1️⃣', '策略4 左側抄底 (賺賠比升級)', s4, 1000)}\n"
      f"{fmt_line('2️⃣', '策略1 波段回踩 (原版鎖定)', s1, 1000)}\n"
      f"{fmt_line('3️⃣', '策略2 VCP突破 (原版鎖定)', s2, 1000)}\n"
      f"{fmt_line('4️⃣', '策略3 日夜盤順勢 (原版鎖定)', s3, 1000)}\n"
      f"{fmt_line('5️⃣', '🛡️ 策略5 避險/對鎖 (原版鎖定)', s5_hedge, 1000)}\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "✅ <b>口徑說明</b>：\n"
      "1. 出場已與實盤對齊：TP1 只移停損不減碼、無 MA runner／動能停滯／時間出場、虧損不夾限、每筆扣 50 元手續費。\n"
      "2. 優勝者以淨值選出，不再用勝率／RR 加權分數與停損／停利網格篩選；n&lt;30 的槽位僅參考，不建議直接啟用。"
  )
  send_tg_message(msg)


if __name__ == "__main__":
  run_weekly_report()
