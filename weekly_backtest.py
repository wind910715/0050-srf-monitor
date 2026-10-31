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


# =========================================================================
# C. 2026-10-27 重構版投資組合回測（分層＋優先序＋新 SL/TP）
#    觸發：python weekly_backtest.py refactor
#    絕不呼叫 send_tg_message；結果只印 stdout 並寫 refactor_results.json。
#    方法限制（報告中須揭露）：
#    - 現貨 15m 代理期貨價（與已發表回測相同限制）；無夜盤 K 線，夜盤開盤緩衝無法驗證。
#    - 5m 條件以 15m 槽位近似；NQ 濾網、ADR 溢價無資料未納入；S2 以回測版 VCP_Ratio 0.92
#      為代表（實盤為 0.96）；訊號參數為固定代表值，不做優勝選拔。
#    - 舊結構對照為同一投資組合 harness 重跑（單一欄位、舊出場、無開盤緩衝），
#      與已發表 per-slot 數字（合計 +14,770、QFF_HEDGE -6,050）口徑不同，僅作參考並列。
# =========================================================================
REFACTOR_SWING_ORDER = {
    "SRF": [
        ("SNIPER_CHIP_MOMENTUM", "S3 日夜盤籌碼順勢共振多", 65, "P1"),
        ("SNIPER_VWAP_PULLBACK", "S1 外資VWAP籌碼首回測確認", 38, "P2"),
        ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P3"),
        ("UPGRADE_BB_W_BOTTOM", "S4 布林下軌W底雙確認", 24, "P4"),
    ],
    "QFF": [
        ("SNIPER_CHIP_MOMENTUM", "T2 日夜盤籌碼順勢共振多", 65, "P1"),
        ("UPGRADE_RSI2_SWING", "T1 Connors RSI(2)極限收腳", 22, "P2"),
    ],
}
OLD_SWING_ORDER = {
    "SRF": [
        ("UPGRADE_BB_W_BOTTOM", "S4 布林下軌W底雙確認", 24, "P4"),
        ("SNIPER_VWAP_PULLBACK", "S1 外資VWAP籌碼首回測確認", 38, "P2"),
        ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P3"),
        ("SNIPER_CHIP_MOMENTUM", "S3 日夜盤籌碼順勢共振多", 65, "P1"),
    ],
    "QFF": [
        ("UPGRADE_RSI2_SWING", "T1 Connors RSI(2)極限收腳", 22, "P2"),
        ("SNIPER_CHIP_MOMENTUM", "T2 日夜盤籌碼順勢共振多", 65, "P1"),
    ],
}
REFACTOR_HEDGE = {
    "SRF": ("HYBRID_EXHAUST_HEDGE", "SRF 對鎖 極限乖離衰竭+破籌碼", 88),
    "QFF": ("UPGRADE_TSMC_CLIMAX_HEDGE", "QFF 對鎖 高檔極限衰竭+初破籌碼", 86),
}
_SIGNAL_FN = {
    "MINERVINI_VCP_POCKET": check_locked_signal,
    "SNIPER_CHIP_MOMENTUM": check_locked_signal,
    "SNIPER_VWAP_PULLBACK": check_locked_signal,
    "HYBRID_EXHAUST_HEDGE": check_locked_signal,
    "UPGRADE_BB_W_BOTTOM": check_targeted_upgrade_signal,
    "UPGRADE_RSI2_SWING": check_targeted_upgrade_signal,
    "UPGRADE_TSMC_CLIMAX_HEDGE": check_targeted_upgrade_signal,
}
_LOTS = {"SRF": 2, "QFF": 1}
_MULT = {"SRF": 1000, "QFF": 100}


def _refactor_close_trade(trades, pos, exit_price, reason, exit_i, mult):
    pnl = pos["pnl_partial"] + int(round(
        (exit_price - pos["entry"]) * mult * pos["lots"]
        if pos["dir"] == "BUY"
        else (pos["entry"] - exit_price) * mult * pos["lots"]
    )) - 50 * pos["lots"]
    total_lots = pos["closed_lots"] + pos["lots"]
    trades.append({
        "asset": pos["asset"], "layer": pos["layer"],
        "strategy": pos["strategy"], "priority": pos["priority"],
        "dir": pos["dir"], "entry_i": pos["entry_i"], "exit_i": exit_i,
        "entry": pos["entry"], "exit": exit_price, "reason": reason,
        "lots": total_lots, "pnl_ntd": pnl,
        "pnl_pts": pnl / mult / total_lots,
        "sl_dist": pos["sl_dist"], "tp1_dist": pos["tp1_dist"],
        "tp2_dist": pos["tp2_dist"],
    })


def _refactor_run_asset(df, asset, swing_order, hedge, structure):
    """Portfolio engine: one swing slot + one hedge slot (new) or one single
    slot (old). Exits before entries each bar; a slot that exits on bar j
    cannot re-enter on bar j (matches per-slot resume-at-next-bar)."""
    is_new = structure == "new"
    sl_mult, tp1_mult, tp2_mult = (1.25, 2.0, 3.5) if is_new else (1.25, 1.05, 3.5)
    mult = _MULT[asset]
    lots0 = _LOTS[asset]
    dates = [d.date() for d in df.index]
    first_bar = set()
    seen = set()
    for i, d in enumerate(dates):
        if d not in seen:
            seen.add(d)
            first_bar.add(i)
    trades = []
    slots = {"swing": None, "hedge": None} if is_new else {"single": None}
    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        hi, lo = float(row["High"]), float(row["Low"])
        close = float(row["Close"])
        # 新結構開盤緩衝：每交易日第一根 K 不觸發硬停損（TP 照常）；舊結構無此規則
        suppress_sl = is_new and i in first_bar
        exited = set()
        for slot_name in list(slots):
            pos = slots[slot_name]
            if pos is None:
                continue
            if pos["dir"] == "BUY":
                hit_sl = (not suppress_sl) and lo <= pos["sl"]
                hit_tp2 = hi >= pos["tp2"]
                hit_tp1 = (not pos["tp1_hit"]) and hi >= pos["tp1"]
            else:
                hit_sl = (not suppress_sl) and hi >= pos["sl"]
                hit_tp2 = lo <= pos["tp2"]
                hit_tp1 = (not pos["tp1_hit"]) and lo <= pos["tp1"]
            if hit_sl:
                _refactor_close_trade(trades, pos, pos["sl"], "SL", i, mult)
                slots[slot_name] = None
                exited.add(slot_name)
            elif hit_tp2:
                _refactor_close_trade(trades, pos, pos["tp2"], "TP2", i, mult)
                slots[slot_name] = None
                exited.add(slot_name)
            elif hit_tp1:
                if is_new:
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
                else:
                    if pos["dir"] == "BUY":
                        pos["sl"] = round(pos["entry"] + 0.30 * pos["tp1_dist"], 2)
                    else:
                        pos["sl"] = (round((pos["entry"] - 0.30 * pos["tp1_dist"]) / 5) * 5
                                     if asset == "QFF"
                                     else round(pos["entry"] - 0.30 * pos["tp1_dist"], 2))
                    pos["tp1_hit"] = True
        # ---- entries (after exits) ----
        def open_pos(slot_name, layer, direction, entry, sl_d, tp1_d, tp2_d, strat, pri):
            if direction == "BUY":
                sl, tp1, tp2 = entry - sl_d, entry + tp1_d, entry + tp2_d
            else:
                sl, tp1, tp2 = entry + sl_d, entry - tp1_d, entry - tp2_d
            slots[slot_name] = {
                "asset": asset, "layer": layer, "dir": direction,
                "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                "entry_i": i, "strategy": strat, "priority": pri,
                "tp1_hit": False, "lots": lots0, "closed_lots": 0,
                "pnl_partial": 0, "sl_dist": sl_d, "tp1_dist": tp1_d,
                "tp2_dist": tp2_d,
            }
        if is_new:
            swing_opened = False
            if slots["swing"] is None and "swing" not in exited:
                for algo_id, label, param, pri in swing_order:
                    direction = _SIGNAL_FN[algo_id](row, prev, algo_id, param)
                    if direction == "BUY":
                        sl_d, tp1_d, tp2_d = _live_dists(asset, float(row["ATR"]), sl_mult, tp1_mult, tp2_mult)
                        open_pos("swing", "swing", "BUY", close, sl_d, tp1_d, tp2_d, label, pri)
                        swing_opened = True
                        break
            if slots["hedge"] is None and "hedge" not in exited and not swing_opened:
                algo_id, label, param = hedge
                direction = _SIGNAL_FN[algo_id](row, prev, algo_id, param)
                if direction == "SELL":
                    sw = slots["swing"]
                    if sw is not None and sw["dir"] == "BUY":
                        _refactor_close_trade(trades, sw, close, "hedge-triggered", i, mult)
                        slots["swing"] = None
                    else:
                        sl_d, tp1_d, tp2_d = _live_dists(asset, float(row["ATR"]), sl_mult, tp1_mult, tp2_mult)
                        open_pos("hedge", "hedge", "SELL", close, sl_d, tp1_d, tp2_d, label, "H")
        else:
            if slots["single"] is None and "single" not in exited:
                opened = False
                for algo_id, label, param, pri in swing_order:
                    direction = _SIGNAL_FN[algo_id](row, prev, algo_id, param)
                    if direction == "BUY":
                        sl_d, tp1_d, tp2_d = _live_dists(asset, float(row["ATR"]), sl_mult, tp1_mult, tp2_mult)
                        open_pos("single", "swing", "BUY", close, sl_d, tp1_d, tp2_d, label, pri)
                        opened = True
                        break
                if not opened:
                    algo_id, label, param = hedge
                    direction = _SIGNAL_FN[algo_id](row, prev, algo_id, param)
                    if direction == "SELL":
                        sl_d, tp1_d, tp2_d = _live_dists(asset, float(row["ATR"]), sl_mult, tp1_mult, tp2_mult)
                        open_pos("single", "hedge", "SELL", close, sl_d, tp1_d, tp2_d, label, "H")
    censored = sum(1 for v in slots.values() if v is not None)
    return trades, censored


def _layer_stats(trades, mid_i):
    total = len(trades)
    wins = [t for t in trades if t["pnl_ntd"] > 0]
    losses = [t for t in trades if t["pnl_ntd"] <= 0]
    win_pts = [t["pnl_pts"] for t in wins]
    loss_pts = [abs(t["pnl_pts"]) for t in losses]
    avg_w = sum(win_pts) / len(win_pts) if win_pts else 0.0
    avg_l = sum(loss_pts) / len(loss_pts) if loss_pts else 0.0
    first = [t for t in trades if t["entry_i"] < mid_i]
    second = [t for t in trades if t["entry_i"] >= mid_i]
    avg = sum(t["pnl_ntd"] for t in trades) / total if total else 0.0
    return {
        "total": total, "wins": len(wins),
        "win_rate": round(len(wins) / total * 100.0, 1) if total else 0.0,
        "avg_ntd": round(avg, 0),
        "avg_pts": round(sum(t["pnl_pts"] for t in trades) / total, 3) if total else 0.0,
        "avg_win_pts": round(avg_w, 3),
        "avg_loss_pts": round(avg_l, 3),
        "rr": round(avg_w / avg_l, 2) if avg_l > 0 else 0.0,
        "net_ntd": sum(t["pnl_ntd"] for t in trades),
        "first_n": len(first),
        "first_avg_ntd": round(sum(t["pnl_ntd"] for t in first) / len(first), 0) if first else 0.0,
        "second_n": len(second),
        "second_avg_ntd": round(sum(t["pnl_ntd"] for t in second) / len(second), 0) if second else 0.0,
    }


def run_refactor_backtest():
    now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
    df_0050 = prepare_master_df("0050.TW", "60d", "15m")
    df_2330 = prepare_master_df("2330.TW", "60d", "15m")
    out = {"run_at": now_str,
           "window": {"0050": [str(df_0050.index[0]), str(df_0050.index[-1]), len(df_0050)],
                      "2330": [str(df_2330.index[0]), str(df_2330.index[-1]), len(df_2330)]},
           "structures": {}}
    for structure, swing_order in (("new", REFACTOR_SWING_ORDER), ("old", OLD_SWING_ORDER)):
        all_trades = []
        struct = {"layers": {}, "censored": 0}
        for asset, df in (("SRF", df_0050), ("QFF", df_2330)):
            mid_i = len(df) // 2
            trades, censored = _refactor_run_asset(
                df, asset, swing_order[asset], REFACTOR_HEDGE[asset], structure)
            struct["censored"] += censored
            for layer in ("swing", "hedge"):
                lt = [t for t in trades if t["layer"] == layer]
                key = f"{asset}_{layer}"
                struct["layers"][key] = _layer_stats(lt, mid_i)
                struct["layers"][key]["trades"] = lt
            all_trades.extend(trades)
        struct["layers"]["TOTAL"] = _layer_stats(all_trades, 10 ** 9)
        # TOTAL 半段欄位不具意義（兩商品中點不同）：半段判定一律以分層為準，
        # 此處將 TOTAL 的半段欄位清空，避免誤讀。
        struct["layers"]["TOTAL"]["first_n"] = None
        struct["layers"]["TOTAL"]["first_avg_ntd"] = None
        struct["layers"]["TOTAL"]["second_n"] = None
        struct["layers"]["TOTAL"]["second_avg_ntd"] = None
        out["structures"][structure] = struct
    # strip raw trades from TOTAL-only view for json size, keep layer trades
    print("=== 2026-10-27 SRF refactor backtest (60d 15m spot proxy) ===")
    print(f"run_at: {now_str}")
    for asset in ("0050", "2330"):
        w = out["window"][asset]
        print(f"window {asset}: {w[0]} -> {w[1]} ({w[2]} bars)")
    for structure in ("new", "old"):
        print(f"--- structure: {structure} ---")
        struct = out["structures"][structure]
        for key in ("SRF_swing", "QFF_swing", "SRF_hedge", "QFF_hedge"):
            s = struct["layers"][key]
            print(f"{key}: n={s['total']} win={s['win_rate']}% avg={s['avg_ntd']:,.0f} NTD "
                  f"({s['avg_pts']:+.3f} pts) RR={s['rr']} net={s['net_ntd']:+,} "
                  f"1H n={s['first_n']} avg={s['first_avg_ntd']:,.0f} | "
                  f"2H n={s['second_n']} avg={s['second_avg_ntd']:,.0f}")
        print(f"censored (still open at end): {struct['censored']}")
    with open("refactor_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote refactor_results.json")

# =========================================================================
# D. 2026-10-27 變體對照實驗（V0-V5，事前註冊）
#    觸發：python weekly_backtest.py variants
#    絕不呼叫 send_tg_message；結果只印 stdout 並寫 variants_results.json。
#    共用口徑：同一 harness、同一 60 天窗口、同一費用（每口 50 元）、
#    同一過關標準（平均正、RR ≥ 1.8、前後半皆正；n＜30 標註樣本不足）。
#    V0 同時引用已發表數字與在本次資料上重算（窗口滾動可能造成微差，兩者並列）。
# =========================================================================
def _s3_live_signal(row, prev, algo_id, param_val):
  """實盤版 SRF S3 等價條件（monitor_srf.py cond_srf_s3）：
  tx_bullish + nq_bullish + price>=Chip_POC + price>=MA10 + (K>D) +
  25<=J<=78 + MACD>=0。
  回測近似（須揭露）：tx_bullish 以 TX_Bull（台指收盤>=台指 MA20）代理；
  NQ 無資料省略；5m K/D 以 15m 代替；無 MA10>=MA20 要求；無 K/D 交叉要求
  （實盤只要當根 K>D）；J 上限 78（回測版為 65）；MACD 只要 >=0（回測版要求 >0 且上升）。"""
  if (
      row["TX_Bull"] == 1
      and row["Close"] >= row["Chip_POC"]
      and row["Close"] >= row["MA10"]
      and row["K"] > row["D"]
      and 25 <= row["J"] <= 78
      and row["MACD_Hist"] >= 0
  ):
    return "BUY"
  return None


_SIGNAL_FN["S3_LIVE"] = _s3_live_signal

VARIANT_SRF_NO_S4 = [
    ("SNIPER_CHIP_MOMENTUM", "S3 日夜盤籌碼順勢共振多", 65, "P1"),
    ("SNIPER_VWAP_PULLBACK", "S1 外資VWAP籌碼首回測確認", 38, "P2"),
    ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P3"),
]
VARIANT_QFF_NO_T1 = [
    ("SNIPER_CHIP_MOMENTUM", "T2 日夜盤籌碼順勢共振多", 65, "P1"),
]
VARIANT_SRF_S3_LIVE = [
    ("S3_LIVE", "S3-live 實盤等價條件", 78, "P1"),
    ("SNIPER_VWAP_PULLBACK", "S1 外資VWAP籌碼首回測確認", 38, "P2"),
    ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P3"),
    ("UPGRADE_BB_W_BOTTOM", "S4 布林下軌W底雙確認", 24, "P4"),
]


def _daytrade_run_single(df, asset, algo_id, label, param, pri):
  """當沖鏡像單策略引擎：進場條件與波段版相同；出場 SL 1.0／TP1 1.5 全出／
  TP2 2.5 ATR；當日最後一根 K 收盤強制出場（EOD，不論盈虧）。
  出場優先序：硬停損 ＞ TP2 ＞ TP1 ＞ EOD（與波段引擎一致的保守順序）。
  限制：資料為現貨日盤 15m，無夜盤 K 線，故夜盤 05:00 強制出場無法驗證；
  日盤強制出場以當日最後一根 K 執行。開盤首根 K 不觸發停損的緩衝對當沖
  部位恆不適用（當沖部位不過夜，首根 K 時無持倉）。"""
  fn = _SIGNAL_FN[algo_id]
  mult = _MULT[asset]
  lots0 = _LOTS[asset]
  dates = [d.date() for d in df.index]
  last_bar = set()
  for i in range(len(dates)):
    if i == len(dates) - 1 or dates[i + 1] != dates[i]:
      last_bar.add(i)
  trades = []
  pos = None
  censored = 0
  for i in range(30, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    hi, lo = float(row["High"]), float(row["Low"])
    close = float(row["Close"])
    exited = False
    if pos is not None:
      if pos["dir"] == "BUY":
        sl, tp1, tp2 = pos["entry"] - pos["sl_d"], pos["entry"] + pos["tp1_d"], pos["entry"] + pos["tp2_d"]
        hit_sl, hit_tp2, hit_tp1 = lo <= sl, hi >= tp2, hi >= tp1
        exit_px = {"SL": sl, "TP2": tp2, "TP1": tp1}.get(
            "SL" if hit_sl else "TP2" if hit_tp2 else "TP1" if hit_tp1 else None)
        reason = "SL" if hit_sl else "TP2" if hit_tp2 else "TP1" if hit_tp1 else None
      else:
        sl, tp1, tp2 = pos["entry"] + pos["sl_d"], pos["entry"] - pos["tp1_d"], pos["entry"] - pos["tp2_d"]
        hit_sl, hit_tp2, hit_tp1 = hi >= sl, lo <= tp2, lo <= tp1
        exit_px = {"SL": sl, "TP2": tp2, "TP1": tp1}.get(
            "SL" if hit_sl else "TP2" if hit_tp2 else "TP1" if hit_tp1 else None)
        reason = "SL" if hit_sl else "TP2" if hit_tp2 else "TP1" if hit_tp1 else None
      if reason is None and i in last_bar:
        reason, exit_px = "EOD", close
      if reason is not None:
        pnl = int(round(
            (exit_px - pos["entry"]) * mult * pos["lots"]
            if pos["dir"] == "BUY"
            else (pos["entry"] - exit_px) * mult * pos["lots"]
        )) - 50 * pos["lots"]
        trades.append({
            "asset": asset, "layer": "daytrade", "strategy": label,
            "priority": pri, "dir": pos["dir"], "entry_i": pos["entry_i"],
            "exit_i": i, "entry": pos["entry"], "exit": exit_px,
            "reason": reason, "lots": pos["lots"], "pnl_ntd": pnl,
            "pnl_pts": pnl / mult / pos["lots"],
            "sl_dist": pos["sl_d"], "tp1_dist": pos["tp1_d"],
            "tp2_dist": pos["tp2_d"],
        })
        pos = None
        exited = True
    if not exited and pos is None and i not in last_bar:
      direction = fn(row, prev, algo_id, param)
      if direction == "BUY":
        sl_d, tp1_d, tp2_d = _live_dists(asset, float(row["ATR"]), 1.0, 1.5, 2.5)
        pos = {"dir": "BUY", "entry": close, "entry_i": i,
               "sl_d": sl_d, "tp1_d": tp1_d, "tp2_d": tp2_d, "lots": lots0}
  if pos is not None:
    censored = 1
  return trades, censored


def _variant_verdict(s):
  """三項標準：平均正、RR ≥ 1.8、前後半皆正。n=0 為無法判定；
  全勝（無虧損筆）時 RR 視為通過；虧損筆皆為損平（平均虧損點數為 0）時
  RR 無法計算，單獨標註；半段無交易時標註為無交易而非負值。
  樣本不足（n<30）一律加註，不論過關與否。"""
  if s["total"] == 0:
    return "無法判定（0 筆）"
  losses = s["total"] - s["wins"]
  if losses == 0 and s["wins"] > 0:
    rr_ok, rr_note = True, ""
  elif s.get("avg_loss_pts", 0) == 0:
    rr_ok, rr_note = False, "RR無法計算（虧損筆皆損平）"
  else:
    rr_ok, rr_note = (s["rr"] >= 1.8), ""
  half_note = ""
  if s["first_n"] == 0 or s["second_n"] == 0:
    half_ok = False
    half_note = "半段無交易"
  else:
    half_ok = (s["first_avg_ntd"] > 0) and (s["second_avg_ntd"] > 0)
  ok = (s["avg_ntd"] > 0) and rr_ok and half_ok
  tag = "（樣本不足 n<30，僅參考）" if s["total"] < 30 else ""
  if ok:
    return "過關" + tag
  reasons = []
  if s["avg_ntd"] <= 0:
    reasons.append("平均≤0")
  if not rr_ok:
    reasons.append(rr_note if rr_note else "RR<1.8")
  if half_note:
    reasons.append(half_note)
  else:
    if s["first_avg_ntd"] <= 0:
      reasons.append("前半≤0")
    if s["second_avg_ntd"] <= 0:
      reasons.append("後半≤0")
  return "不過關：" + "、".join(reasons) + tag


def run_variants_backtest():
  now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
  df_0050 = prepare_master_df("0050.TW", "60d", "15m")
  df_2330 = prepare_master_df("2330.TW", "60d", "15m")
  out = {"run_at": now_str,
         "window": {"0050": [str(df_0050.index[0]), str(df_0050.index[-1]), len(df_0050)],
                    "2330": [str(df_2330.index[0]), str(df_2330.index[-1]), len(df_2330)]},
         "variants": {}}
  dfs = {"SRF": df_0050, "QFF": df_2330}

  def run_new(asset, swing_order):
    df = dfs[asset]
    trades, censored = _refactor_run_asset(
        df, asset, swing_order, REFACTOR_HEDGE[asset], "new")
    return trades, censored, len(df) // 2

  # V0 重算（與本次資料同一窗口，供 V1／V2／V4 對照）
  v0 = {}
  for asset in ("SRF", "QFF"):
    trades, censored, mid_i = run_new(asset, REFACTOR_SWING_ORDER[asset])
    for layer in ("swing", "hedge"):
      lt = [t for t in trades if t["layer"] == layer]
      s = _layer_stats(lt, mid_i)
      s["trades"] = lt
      s["censored"] = censored
      v0[f"{asset}_{layer}"] = s
  out["variants"]["V0_recomputed"] = {k: {kk: vv for kk, vv in v.items() if kk != "trades"} for k, v in v0.items()}
  out["variants"]["V0_recomputed_trades"] = {
      k: v["trades"] for k, v in v0.items()}

  # V1：SRF 波段去 S4
  trades, censored, mid_i = run_new("SRF", VARIANT_SRF_NO_S4)
  lt = [t for t in trades if t["layer"] == "swing"]
  s = _layer_stats(lt, mid_i)
  s["trades"] = lt
  s["censored"] = censored
  out["variants"]["V1_SRF_swing_no_S4"] = {k: v for k, v in s.items() if k != "trades"}
  out["variants"]["V1_trades"] = lt
  lt_h = [t for t in trades if t["layer"] == "hedge"]
  out["variants"]["V1_SRF_hedge_side"] = _layer_stats(lt_h, mid_i)

  # V2：QFF 波段去 T1
  trades, censored, mid_i = run_new("QFF", VARIANT_QFF_NO_T1)
  lt = [t for t in trades if t["layer"] == "swing"]
  s = _layer_stats(lt, mid_i)
  s["trades"] = lt
  s["censored"] = censored
  out["variants"]["V2_QFF_swing_no_T1"] = {k: v for k, v in s.items() if k != "trades"}
  out["variants"]["V2_trades"] = lt
  lt_h = [t for t in trades if t["layer"] == "hedge"]
  out["variants"]["V2_QFF_hedge_side"] = _layer_stats(lt_h, mid_i)

  # V3：當沖鏡像（每支波段策略各別跑當沖出場）
  v3 = {}
  v3_trades = {}
  for asset, strat_list in (("SRF", REFACTOR_SWING_ORDER["SRF"]), ("QFF", REFACTOR_SWING_ORDER["QFF"])):
    df = dfs[asset]
    mid_i = len(df) // 2
    for algo_id, label, param, pri in strat_list:
      trades, censored = _daytrade_run_single(df, asset, algo_id, label, param, pri)
      s = _layer_stats(trades, mid_i)
      s["censored"] = censored
      key = f"{asset}_{label.split(' ')[0]}_daytrade"
      v3[key] = s
      v3_trades[key] = trades
  out["variants"]["V3_daytrade_mirror"] = v3
  out["variants"]["V3_trades"] = v3_trades

  # V4：SRF 波段 S3 實盤等價條件
  trades, censored, mid_i = run_new("SRF", VARIANT_SRF_S3_LIVE)
  lt = [t for t in trades if t["layer"] == "swing"]
  s = _layer_stats(lt, mid_i)
  s["trades"] = lt
  s["censored"] = censored
  out["variants"]["V4_SRF_swing_S3_live"] = {k: v for k, v in s.items() if k != "trades"}
  out["variants"]["V4_trades"] = lt
  s3_trades = [t for t in lt if t["strategy"].startswith("S3-live")]
  out["variants"]["V4_S3_live_only"] = _layer_stats(s3_trades, mid_i)
  lt_h = [t for t in trades if t["layer"] == "hedge"]
  out["variants"]["V4_SRF_hedge_side"] = _layer_stats(lt_h, mid_i)

  # V5：V0 重算交易的策略 × 前後半拆解
  breakdown = {}
  for asset in ("SRF", "QFF"):
    df = dfs[asset]
    mid_i = len(df) // 2
    for layer in ("swing", "hedge"):
      lt = [t for t in v0[f"{asset}_{layer}"]["trades"]]
      by_strat = {}
      for t in lt:
        by_strat.setdefault(t["strategy"], []).append(t)
      for strat, st in by_strat.items():
        first = [x for x in st if x["entry_i"] < mid_i]
        second = [x for x in st if x["entry_i"] >= mid_i]
        breakdown[f"{asset}_{layer}_{strat}"] = {
            "total": len(st),
            "wins": sum(1 for x in st if x["pnl_ntd"] > 0),
            "net_ntd": sum(x["pnl_ntd"] for x in st),
            "avg_ntd": round(sum(x["pnl_ntd"] for x in st) / len(st), 0) if st else 0.0,
            "first_n": len(first),
            "first_net": sum(x["pnl_ntd"] for x in first),
            "first_avg": round(sum(x["pnl_ntd"] for x in first) / len(first), 0) if first else 0.0,
            "second_n": len(second),
            "second_net": sum(x["pnl_ntd"] for x in second),
            "second_avg": round(sum(x["pnl_ntd"] for x in second) / len(second), 0) if second else 0.0,
        }
  out["variants"]["V5_breakdown"] = breakdown

  with open("variants_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
  print("=== 2026-10-27 SRF variants backtest (60d 15m spot proxy) ===")
  print(f"run_at: {now_str}")
  for k, v in out["variants"].items():
    if isinstance(v, dict) and "total" in v:
      print(f"{k}: n={v['total']} win={v['win_rate']}% avg={v['avg_ntd']:,.0f} RR={v['rr']} "
            f"net={v['net_ntd']:+,} 1H={v['first_avg_ntd']:,.0f} 2H={v['second_avg_ntd']:,.0f} -> {_variant_verdict(v)}")
  for k, v in v3.items():
    print(f"V3 {k}: n={v['total']} win={v['win_rate']}% avg={v['avg_ntd']:,.0f} RR={v['rr']} "
          f"net={v['net_ntd']:+,} 1H={v['first_avg_ntd']:,.0f} 2H={v['second_avg_ntd']:,.0f} -> {_variant_verdict(v)}")
  print("wrote variants_results.json")


# =========================================================================
# E. 2026-10-27 第二輪變體對照實驗（V6-V8，事前註冊，限波段）
#    觸發：python weekly_backtest.py variants2
#    絕不呼叫 send_tg_message；結果只印 stdout 並寫 variants_round2.json。
#    共用口徑與上一輪完全一致：同一 harness、同一 60 天 15m 現貨代理窗口、
#    同一費用（每口 50 元）、同一過關標準（平均正、RR ≥ 1.8、前後半皆正；
#    n＜30 標註樣本不足）。本輪只跑波段：對鎖層不跑、當沖層不跑。
#    波段純淨做法：hedge 欄位傳入永不觸發的 NEVER 訊號，使引擎不開對鎖單、
#    不產生 hedge-triggered 平倉，波段交易即為純波段；既有 refactor／週報／
#    V0-V5 路徑一律不動。
# =========================================================================
def _never_signal(row, prev, algo_id, param_val):
  """永不觸發的訊號（第二輪波段純淨用）：對鎖欄位傳入此訊號後，
  引擎永遠不開對鎖單、不產生 hedge-triggered 平倉。"""
  return None


_SIGNAL_FN["NEVER"] = _never_signal

ROUND2_NO_HEDGE = ("NEVER", "無對鎖（本輪不跑）", 0)

# V6：SRF 波段去 S1（S3-live／S2／S4，S1 移除後順位遞補）
VARIANT_V6_SRF_NO_S1 = [
    ("S3_LIVE", "S3-live 實盤等價條件", 78, "P1"),
    ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P2"),
    ("UPGRADE_BB_W_BOTTOM", "S4 布林下軌W底雙確認", 24, "P3"),
]
# V7：SRF 波段去 S1＋S4（只留 S3-live／S2）
VARIANT_V7_SRF_NO_S1_S4 = [
    ("S3_LIVE", "S3-live 實盤等價條件", 78, "P1"),
    ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P2"),
]
# V8：優先序對調（全留，S2 → S3-live → S1 → S4）
VARIANT_V8_SRF_S2_FIRST = [
    ("MINERVINI_VCP_POCKET", "S2 Minervini VCP口袋支點", 80, "P1"),
    ("S3_LIVE", "S3-live 實盤等價條件", 78, "P2"),
    ("SNIPER_VWAP_PULLBACK", "S1 外資VWAP籌碼首回測確認", 38, "P3"),
    ("UPGRADE_BB_W_BOTTOM", "S4 布林下軌W底雙確認", 24, "P4"),
]


def _round2_breakdown(trades, mid_i):
  """逐策略拆解：筆數／勝數／平均／累計／前半／後半。"""
  out = {}
  by_strat = {}
  for t in trades:
    by_strat.setdefault(t["strategy"], []).append(t)
  for strat, st in by_strat.items():
    first = [x for x in st if x["entry_i"] < mid_i]
    second = [x for x in st if x["entry_i"] >= mid_i]
    out[strat] = {
        "total": len(st),
        "wins": sum(1 for x in st if x["pnl_ntd"] > 0),
        "avg_ntd": round(sum(x["pnl_ntd"] for x in st) / len(st), 0) if st else 0.0,
        "net_ntd": sum(x["pnl_ntd"] for x in st),
        "first_n": len(first),
        "first_avg": round(sum(x["pnl_ntd"] for x in first) / len(first), 0) if first else 0.0,
        "second_n": len(second),
        "second_avg": round(sum(x["pnl_ntd"] for x in second) / len(second), 0) if second else 0.0,
    }
  return out


def run_variants_round2():
  now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M")
  df_0050 = prepare_master_df("0050.TW", "60d", "15m")
  mid_i = len(df_0050) // 2
  out = {"run_at": now_str,
         "window": {"0050": [str(df_0050.index[0]), str(df_0050.index[-1]), len(df_0050)]},
         "variants": {}}

  def run_swing_only(order):
    trades, censored = _refactor_run_asset(
        df_0050, "SRF", order, ROUND2_NO_HEDGE, "new")
    lt = [t for t in trades if t["layer"] == "swing"]
    assert all(t["layer"] == "swing" for t in trades), "波段純淨檢查失敗：出現非 swing 交易"
    s = _layer_stats(lt, mid_i)
    s["censored"] = censored
    return s, lt

  for key, order in (("V6_SRF_swing_no_S1", VARIANT_V6_SRF_NO_S1),
                     ("V7_SRF_swing_no_S1_S4", VARIANT_V7_SRF_NO_S1_S4),
                     ("V8_SRF_swing_S2_first", VARIANT_V8_SRF_S2_FIRST)):
    s, lt = run_swing_only(order)
    out["variants"][key] = {k: v for k, v in s.items()}
    out["variants"][key + "_trades"] = lt
    out["variants"][key + "_breakdown"] = _round2_breakdown(lt, mid_i)

  with open("variants_round2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
  print("=== 2026-10-27 SRF variants round 2 (swing only, 60d 15m spot proxy) ===")
  print(f"run_at: {now_str}")
  print(f"window 0050: {out['window']['0050'][0]} -> {out['window']['0050'][1]} ({out['window']['0050'][2]} bars)")
  for key in ("V6_SRF_swing_no_S1", "V7_SRF_swing_no_S1_S4", "V8_SRF_swing_S2_first"):
    v = out["variants"][key]
    print(f"{key}: n={v['total']} win={v['win_rate']}% avg={v['avg_ntd']:,.0f} RR={v['rr']} "
          f"net={v['net_ntd']:+,} 1H={v['first_avg_ntd']:,.0f} 2H={v['second_avg_ntd']:,.0f} -> {_variant_verdict(v)}")
    for strat, b in out["variants"][key + "_breakdown"].items():
      print(f"    {strat}: n={b['total']} win={b['wins']} avg={b['avg_ntd']:,.0f} net={b['net_ntd']:+,} "
            f"1H={b['first_n']}/{b['first_avg']:,.0f} 2H={b['second_n']}/{b['second_avg']:,.0f}")
  print("wrote variants_round2.json")


if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1 and sys.argv[1] == "refactor":
    run_refactor_backtest()
  elif len(sys.argv) > 1 and sys.argv[1] == "variants":
    run_variants_backtest()
  elif len(sys.argv) > 1 and sys.argv[1] == "variants2":
    run_variants_round2()
  else:
    run_weekly_report()
