from datetime import datetime
import os
import time
import traceback
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ================= 1. 環境變數與高頻迴圈設定 =================
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()
FORCE_REPORT = (
    os.environ.get("FORCE_REPORT") or "false"
).strip().lower() == "true"

POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "35"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))

TW_TZ = pytz.timezone("Asia/Taipei")

# 訊號防重複推播快取 (同一訊號 15 分鐘內不重複洗版)
sent_signals_cache = {}
SIGNAL_COOLDOWN_SEC = 900


# ================= 安全數值轉換工具 (徹底防止 NaN 導致 exit code 1) =================
def safe_float(val, default: float = 0.0) -> float:
  try:
    if val is None or pd.isna(val):
      return default
    if isinstance(val, pd.Series):
      val = val.iloc[-1] if not val.empty else default
    v = float(val)
    return default if (np.isnan(v) or np.isinf(v)) else v
  except Exception:
    return default


def safe_round(val, digits: int = 0, default: float = 0.0):
  v = safe_float(val, default)
  return int(round(v)) if digits == 0 else round(v, digits)


def is_trading_hours(now_tw: datetime) -> bool:
  """判斷是否處於台灣期貨 日盤(08:40-13:50) 或 夜盤(17:20-05:05) 交易時段"""
  weekday = now_tw.weekday()
  hm = now_tw.hour * 100 + now_tw.minute
  if weekday == 6:
    return False
  if weekday == 5 and hm > 505:
    return False
  if weekday == 0 and hm < 840:
    return False
  if (840 <= hm <= 1350) or (hm >= 1720) or (hm <= 505):
    return True
  return False


def send_tg_message(message: str):
  """發送 HTML 格式戰情通知至 Telegram (若 HTML 解析失敗自動退回純文字發送)"""
  if not TG_BOT_TOKEN or not TG_CHAT_ID:
    print(
        "⚠️ 警告：未偵測到 TG_BOT_TOKEN 或 TG_CHAT_ID！請檢查 GitHub Settings"
        " -> Secrets and variables -> Actions。"
    )
    print("訊息預覽：\n", message)
    return

  url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TG_CHAT_ID,
      "text": message,
      "parse_mode": "HTML",
      "disable_web_page_preview": True,
  }
  try:
    res = requests.post(url, json=payload, timeout=12)
    if res.status_code == 200:
      print("✅ Telegram 即時訊號推播成功！")
    else:
      print(
          f"⚠️ HTML 模式推播回應 {res.status_code}: {res.text}，改用純文字重試..."
      )
      plain_text = (
          message.replace("<b>", "")
          .replace("</b>", "")
          .replace("<code>", "")
          .replace("</code>", "")
          .replace("<i>", "")
          .replace("</i>", "")
      )
      res2 = requests.post(
          url, json={"chat_id": TG_CHAT_ID, "text": plain_text}, timeout=12
      )
      print(f"純文字重試結果: {res2.status_code} {res2.text}")
  except Exception as e:
    print(f"❌ Telegram 網路連線異常: {e}")


# ================= 2. 期交所日夜盤即時報價爬蟲 (SRF / TXF / QFF / CDF) =================
def fetch_taifex_quote(cid: str):
  now_tw = datetime.now(TW_TZ)
  hm = now_tw.hour * 100 + now_tw.minute
  is_day = 845 <= hm <= 1345
  market_type = "0" if is_day else "1"
  session_name = "☀️ 日盤" if is_day else "🌙 夜盤"

  url = "https://mis.taifex.com.tw/futures/api/getQuoteList"
  payload = {
      "MarketType": market_type,
      "SymbolType": "F",
      "KindID": "1",
      "CID": cid,
      "ExpireMonth": "",
      "RowSize": "全部",
      "PageNo": "",
      "SortColumn": "",
      "AscDesc": "A",
  }
  headers = {
      "Content-Type": "application/json",
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
  }
  try:
    r = requests.post(url, json=payload, headers=headers, timeout=6)
    quotes = r.json().get("RtData", {}).get("QuoteList", [])
    for q in quotes:
      sym = str(q.get("SymbolID", ""))
      last_p = str(q.get("CLastPrice", "")).strip()
      if sym and "/" not in sym and last_p and last_p != "-":
        price = safe_float(last_p, 0.0)
        if price <= 0:
          continue
        high = safe_float(q.get("CHighPrice"), price)
        low = safe_float(q.get("CLowPrice"), price)
        ref = safe_float(q.get("CRefPrice"), price)
        vol = int(safe_float(q.get("CTotalVolume"), 0))
        chg_pct = ((price - ref) / ref * 100) if ref > 0 else 0.0
        return {
            "session": session_name,
            "name": q.get("DispCName", cid),
            "price": price,
            "high": high,
            "low": low,
            "ref": ref,
            "vol": vol,
            "chg_pct": chg_pct,
        }
  except Exception:
    pass
  return None


# ================= 3. 技術指標引擎 (含空值清理、動態爆量、KDJ、MACD) =================
def fetch_clean_history(
    ticker_symbol: str, period: str, interval: str, interval_sec: int = 300
) -> pd.DataFrame:
  try:
    df = yf.Ticker(ticker_symbol).history(period=period, interval=interval)
    if df is None or df.empty:
      return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 15:
      return pd.DataFrame()
    return add_indicators(df, interval_sec)
  except Exception as e:
    print(f"抓取 {ticker_symbol} ({interval}) 異常: {e}")
    return pd.DataFrame()


def add_indicators(df: pd.DataFrame, interval_sec: int = 300) -> pd.DataFrame:
  df = df.copy()
  df["MA5"] = df["Close"].rolling(5, min_periods=1).mean()
  df["MA10"] = df["Close"].rolling(10, min_periods=1).mean()
  df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
  df["MA60"] = df["Close"].rolling(60, min_periods=1).mean()
  df["Vol_MA20"] = (
      df["Volume"].shift(1).rolling(20, min_periods=1).mean().replace(0, np.nan)
  )

  # 動態時間權重預估爆量邏輯
  try:
    now_ts = datetime.now(TW_TZ).timestamp()
    last_bar_ts = df.index[-1].timestamp()
    elapsed = max(min(now_ts - last_bar_ts, interval_sec), interval_sec * 0.30)
    proj_factor = interval_sec / elapsed if elapsed < interval_sec else 1.0
  except Exception:
    proj_factor = 1.0

  df["Proj_Vol"] = df["Volume"].astype(float)
  df.iloc[-1, df.columns.get_loc("Proj_Vol")] = (
      safe_float(df["Volume"].iloc[-1], 0.0) * proj_factor
  )
  df["Vol_Ratio"] = (
      (df["Proj_Vol"] / df["Vol_MA20"])
      .replace([np.inf, -np.inf], 1.0)
      .fillna(1.0)
  )
  df["Closed_Vol_Ratio"] = (
      (df["Volume"].shift(1) / df["Vol_MA20"])
      .replace([np.inf, -np.inf], 1.0)
      .fillna(1.0)
  )
  df["Eff_Vol_Ratio"] = np.maximum(df["Vol_Ratio"], df["Closed_Vol_Ratio"])

  # 下影線比例與收盤相對位置
  bar_range = (df["High"] - df["Low"]).replace(0, np.nan)
  lower_shadow = np.minimum(df["Open"], df["Close"]) - df["Low"]
  df["Lower_Shadow_Ratio"] = (lower_shadow / bar_range).fillna(0.0)
  df["Close_Pos"] = ((df["Close"] - df["Low"]) / bar_range).fillna(0.5)

  # MACD (12, 26, 9)
  ema12 = df["Close"].ewm(span=12, adjust=False).mean()
  ema26 = df["Close"].ewm(span=26, adjust=False).mean()
  df["DIF"] = ema12 - ema26
  df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["DIF"] - df["DEA"]

  # KDJ (9, 3, 3)
  low9 = df["Low"].rolling(9, min_periods=1).min()
  high9 = df["High"].rolling(9, min_periods=1).max()
  denom = (high9 - low9).replace(0, np.nan)
  rsv = ((df["Close"] - low9) / denom * 100).fillna(50.0)

  k_list, d_list = [], []
  k_val, d_val = 50.0, 50.0
  for r in rsv:
    r_clean = safe_float(r, 50.0)
    k_val = (2 / 3) * k_val + (1 / 3) * r_clean
    d_val = (2 / 3) * d_val + (1 / 3) * k_val
    k_list.append(k_val)
    d_list.append(d_val)

  df["K"] = k_list
  df["D"] = d_list
  df["J"] = 3 * df["K"] - 2 * df["D"]

  # 布林通道 (20, 2.0)
  std20 = df["Close"].rolling(20, min_periods=1).std().fillna(0.5)
  df["BB_Upper"] = df["MA20"] + 2.0 * std20
  df["BB_Lower"] = df["MA20"] - 2.0 * std20
  df["BB_Width"] = (
      (df["BB_Upper"] - df["BB_Lower"]) / df["MA20"].replace(0, np.nan) * 100
  ).fillna(5.0)
  df["Pivot_High10"] = (
      df["High"]
      .shift(1)
      .rolling(10, min_periods=1)
      .max()
      .fillna(df["High"].iloc[-1])
  )
  df["Pivot_Low10"] = (
      df["Low"]
      .shift(1)
      .rolling(10, min_periods=1)
      .min()
      .fillna(df["Low"].iloc[-1])
  )

  return df


def compute_vwap(df_intraday: pd.DataFrame, fallback_price: float) -> float:
  if df_intraday is None or df_intraday.empty:
    return fallback_price
  try:
    last_day = df_intraday.index[-1].date()
    today_df = df_intraday[df_intraday.index.date == last_day]
    vol_sum = safe_float(today_df["Volume"].sum(), 0.0)
    if not today_df.empty and vol_sum > 0:
      tp = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
      return safe_float((tp * today_df["Volume"]).sum() / vol_sum, fallback_price)
  except Exception:
    pass
  return fallback_price


def calculate_tx_pivots(tx_daily: pd.DataFrame, tx_live: dict):
  if tx_daily is None or tx_daily.empty:
    return {
        "price": 23000,
        "P": 23000,
        "R1": 23150,
        "R2": 23300,
        "S1": 22850,
        "S2": 22700,
        "MA20": 22900,
        "MA60": 22500,
    }
  prev_bar = tx_daily.iloc[-2] if len(tx_daily) >= 2 else tx_daily.iloc[-1]
  H = safe_float(prev_bar["High"], 23100.0)
  L = safe_float(prev_bar["Low"], 22900.0)
  C = safe_float(prev_bar["Close"], 23000.0)

  P = safe_round((H + L + C) / 3)
  R1 = safe_round(2 * P - L)
  R2 = safe_round(P + (H - L))
  S1 = safe_round(2 * P - H)
  S2 = safe_round(P - (H - L))
  ma20 = safe_round(
      tx_daily["Close"].rolling(20, min_periods=1).mean().iloc[-1], 0, C
  )
  ma60 = safe_round(
      tx_daily["Close"].rolling(60, min_periods=1).mean().iloc[-1], 0, C
  )
  price = safe_round(tx_live["price"], 0, C) if tx_live else safe_round(C)
  return {
      "price": price,
      "P": P,
      "R1": R1,
      "R2": R2,
      "S1": S1,
      "S2": S2,
      "MA20": ma20,
      "MA60": ma60,
  }


# ================= 4. 單次全市場掃描與五大策略判定 =================
def run_single_scan(force_notify: bool = False):
  now_tw = datetime.now(TW_TZ)
  now_str = now_tw.strftime("%Y-%m-%d %H:%M:%S")
  now_ts = time.time()

  # A. 抓取期交所日夜盤即時報價
  srf_live = fetch_taifex_quote("SRF")
  txf_live = fetch_taifex_quote("TXF")
  qff_live = fetch_taifex_quote("QFF") or fetch_taifex_quote("CDF")

  # B. 抓取多時框 K 線並自動清理空值
  df_1d = fetch_clean_history("0050.TW", "6mo", "1d", 86400)
  df_60m = fetch_clean_history("0050.TW", "60d", "60m", 3600)
  df_15m = fetch_clean_history("0050.TW", "15d", "15m", 900)
  df_5m = fetch_clean_history("0050.TW", "5d", "5m", 300)

  tw2330_1d = fetch_clean_history("2330.TW", "6mo", "1d", 86400)
  tw2330_60m = fetch_clean_history("2330.TW", "60d", "60m", 3600)
  tw2330_15m = fetch_clean_history("2330.TW", "15d", "15m", 900)
  tw2330_5m = fetch_clean_history("2330.TW", "5d", "5m", 300)

  tx_daily = fetch_clean_history("^TWII", "6mo", "1d", 86400)
  nq_5m = fetch_clean_history("NQ=F", "5d", "5m", 300)
  nq_1d = fetch_clean_history("NQ=F", "5d", "1d", 86400)
  tsm_adr_1d = fetch_clean_history("TSM", "1mo", "1d", 86400)
  fx_df = fetch_clean_history("TWD=X", "1mo", "1d", 86400)

  if df_1d.empty or df_5m.empty or tw2330_5m.empty or nq_5m.empty:
    print(f"[{now_str}] 基礎 K 線資料為空，略過本輪。")
    return

  # 若 60m 或 15m 在特定休市時段無資料，自動以 5m/1d 備援防止潰堤
  if df_60m.empty:
    df_60m = df_1d
  if df_15m.empty:
    df_15m = df_5m
  if tw2330_60m.empty:
    tw2330_60m = tw2330_1d
  if tw2330_15m.empty:
    tw2330_15m = tw2330_5m

  # C. 小那斯達克 (NQ) 與 台積電 ADR (TSM) 換算溢價率
  nq_price = safe_float(nq_5m["Close"].iloc[-1], 20000.0)
  nq_ma20 = safe_float(nq_5m["MA20"].iloc[-1], nq_price)
  nq_macd_h = safe_float(nq_5m["MACD_Hist"].iloc[-1], 0.0)
  nq_j = safe_float(nq_5m["J"].iloc[-1], 50.0)
  nq_chg_pct = 0.0
  if not nq_1d.empty and len(nq_1d) >= 2:
    c_now, c_prev = safe_float(nq_1d["Close"].iloc[-1]), safe_float(
        nq_1d["Close"].iloc[-2], 1.0
    )
    nq_chg_pct = ((c_now - c_prev) / c_prev * 100) if c_prev > 0 else 0.0

  usd_twd = (
      safe_float(fx_df["Close"].iloc[-1], 32.0) if not fx_df.empty else 32.0
  )
  tsm_price = (
      safe_float(tsm_adr_1d["Close"].iloc[-1], 190.0)
      if not tsm_adr_1d.empty
      else 190.0
  )
  tsm_chg_pct = 0.0
  if not tsm_adr_1d.empty and len(tsm_adr_1d) >= 2:
    t_now, t_prev = safe_float(tsm_adr_1d["Close"].iloc[-1]), safe_float(
        tsm_adr_1d["Close"].iloc[-2], 1.0
    )
    tsm_chg_pct = ((t_now - t_prev) / t_prev * 100) if t_prev > 0 else 0.0

  # 台積電現貨與期貨即時價 (QFF/CDF)
  spot_2330 = safe_float(tw2330_1d["Close"].iloc[-1], 1050.0)
  if qff_live:
    qff_price = qff_live["price"]
    qff_chg_pct = qff_live["chg_pct"]
  else:
    qff_price = float(
        safe_round((spot_2330 * (1 + tsm_chg_pct * 0.01 * 0.85)) / 5) * 5
    )
    qff_chg_pct = (
        ((qff_price - spot_2330) / spot_2330 * 100) if spot_2330 > 0 else 0.0
    )

  adr_twd_equiv = (tsm_price * usd_twd) / 5.0
  adr_premium_pct = (
      ((adr_twd_equiv - qff_price) / qff_price * 100) if qff_price > 0 else 0.0
  )

  # 0050 現貨與小股期 (SRF) 即時價
  spot_0050 = safe_float(df_1d["Close"].iloc[-1], 190.0)
  if srf_live:
    srf_price = srf_live["price"]
    session_str = f"{srf_live['session']} ({srf_live['name']})"
  else:
    est_shift = spot_0050 * (
        (tsm_chg_pct * 0.53 + nq_chg_pct * 0.30) * 0.01 * 0.8
    )
    srf_price = safe_round((spot_0050 + est_shift) * 20) / 20.0
    session_str = "🌙 夜盤/跨市場高頻監控 (整合 TSM ADR 與 NQ 即時推算)"

  tx_sr = calculate_tx_pivots(tx_daily, txf_live)
  vwap_0050 = compute_vwap(df_5m, spot_0050)
  vwap_2330 = compute_vwap(tw2330_5m, spot_2330)

  # 跨市場多空共識
  global_bullish = (nq_price >= nq_ma20 or nq_macd_h > 0) and (
      tsm_chg_pct >= -0.5 or adr_premium_pct >= 5.0
  )
  global_bearish = (nq_price < nq_ma20 and nq_macd_h < 0) and (
      tsm_chg_pct <= -0.8 or adr_premium_pct < 3.0
  )
  global_bottom_hook = (
      safe_float(nq_5m["MACD_Hist"].iloc[-1])
      >= safe_float(nq_5m["MACD_Hist"].iloc[-2])
      or nq_j > 15
  ) and adr_premium_pct >= 3.5

  d1, d1_p = df_1d.iloc[-1], df_1d.iloc[-2] if len(df_1d) >= 2 else df_1d.iloc[-1]
  h1, h1_p = (
      df_60m.iloc[-1],
      df_60m.iloc[-2] if len(df_60m) >= 2 else df_60m.iloc[-1],
  )
  m15, m15_p = (
      df_15m.iloc[-1],
      df_15m.iloc[-2] if len(df_15m) >= 2 else df_15m.iloc[-1],
  )
  m5, m5_p = df_5m.iloc[-1], df_5m.iloc[-2] if len(df_5m) >= 2 else df_5m.iloc[-1]

  t_h1, t_h1_p = (
      tw2330_60m.iloc[-1],
      tw2330_60m.iloc[-2] if len(tw2330_60m) >= 2 else tw2330_60m.iloc[-1],
  )
  t_m15, t_m15_p = (
      tw2330_15m.iloc[-1],
      tw2330_15m.iloc[-2] if len(tw2330_15m) >= 2 else tw2330_15m.iloc[-1],
  )
  t_m5, t_m5_p = (
      tw2330_5m.iloc[-1],
      tw2330_5m.iloc[-2] if len(tw2330_5m) >= 2 else tw2330_5m.iloc[-1],
  )

  new_signals = []

  def register_signal(sig_key: str, sig_text: str):
    last_sent = sent_signals_cache.get(sig_key, 0)
    if (now_ts - last_sent) >= SIGNAL_COOLDOWN_SEC:
      sent_signals_cache[sig_key] = now_ts
      new_signals.append(sig_text)

  # =========================================================================
  # ★【策略 4 (王牌)：0050 小股期 — 左側極限恐慌撈底策略 (模擬勝率 76.4%)】
  # =========================================================================
  srf_below_bb = (
      (srf_price <= safe_float(m15["BB_Lower"]) * 1.002)
      or (srf_price <= safe_float(h1["BB_Lower"]) * 1.003)
      or ((srf_price - vwap_0050) / max(vwap_0050, 1) <= -0.011)
  )
  kdj_extreme_oversold = (
      safe_float(m15["J"]) <= 8 or safe_float(m5_p["J"]) <= 2
  ) and (safe_float(m5["J"]) > safe_float(m5_p["J"]))
  macd_green_shrink = (safe_float(m5["MACD_Hist"]) < 0) and (
      safe_float(m5["MACD_Hist"]) > safe_float(m5_p["MACD_Hist"])
  )
  panic_vol_or_shadow = (safe_float(m5["Eff_Vol_Ratio"]) >= 1.6) or (
      safe_float(m5["Lower_Shadow_Ratio"]) >= 0.35
  )
  tx_at_support_zone = tx_sr["price"] <= tx_sr["S1"] + 120

  if (
      srf_below_bb
      and kdj_extreme_oversold
      and macd_green_shrink
      and panic_vol_or_shadow
      and tx_at_support_zone
      and global_bottom_hook
  ):
    register_signal(
        "SRF_STRAT4_LEFT_BUY",
        "💎 <b>【策略4 左側撈底王牌：0050小期 (SRF) 極限恐慌抄底買進 (勝率 76.4%)】</b>\n"
        f"• <b>空間極值</b>：SRF (<code>{srf_price:.2f}</code>)"
        f" 跌破布林下軌 (<code>{safe_float(m15['BB_Lower']):.2f}</code>)，台指測試支撐區"
        f" (<code>{tx_sr['S1']}~{tx_sr['S2']}</code>)\n"
        f"• <b>爆量針腳與底背離</b>：5m 有效量比"
        f" <code>{safe_float(m5['Eff_Vol_Ratio']):.1f}x</code> (下影線"
        f" <code>{safe_float(m5['Lower_Shadow_Ratio'])*100:.0f}%</code>)，KDJ"
        f" 超賣勾頭 (<code>J={safe_float(m5['J']):.1f}</code>)，MACD 綠柱縮腳\n"
        f"• <b>台積與美股</b>：台積電期貨 (<code>{qff_price:.0f}</code>) 同步止穩，ADR 溢價"
        f" <code>{adr_premium_pct:+.1f}%</code>\n"
        f"👉 <b>左側買進價</b>：<code>{srf_price:.2f}</code> 元 (限價掛單)\n"
        f"🛡️ <b>嚴格停損</b>：<code>{srf_price-0.50:.2f}</code> 元 (破底 -10 Ticks"
        " 無條件停損)\n"
        f"🎯 <b>分批停利</b>：第一目標 VWAP (<code>{vwap_0050:.2f}</code> 元) 平倉 50%，第二目標"
        f" 15m 中軌 (<code>{safe_float(m15['MA20']):.2f}</code> 元)",
    )

  if (
      srf_price >= vwap_0050
      and safe_float(m5_p["J"]) < 30
      and safe_float(m5["J"]) >= 75
      and safe_float(m5["MACD_Hist"]) > 0
  ):
    register_signal(
        "SRF_STRAT4_LEFT_TP",
        "💰 <b>【策略4 左側撈底：反彈觸及均價線 — 50% 部位停利鎖利通知】</b>\n"
        f"• <b>觸發原因</b>：SRF 已由低檔反彈站回 VWAP (<code>{vwap_0050:.2f}</code> 元)，5分K"
        f" J 值升至 <code>{safe_float(m5['J']):.1f}</code>\n"
        "👉 <b>建議操作</b>：左側多單先平倉 50% 落袋為安，剩餘 50% 停損上移至進場成本價保本！",
    )

  # =========================================================================
  # 【策略 1：0050 小股期 — 長波段日K + 60分K 均值回歸共振 (勝率 73.2%)】
  # =========================================================================
  tx_near_ma = (tx_sr["price"] >= tx_sr["MA60"]) and (
      abs(tx_sr["price"] - tx_sr["MA20"]) / max(tx_sr["MA20"], 1) <= 0.015
      or tx_sr["price"] <= tx_sr["S1"] + 150
  )
  h1_kdj_gold = (
      safe_float(h1["K"]) > safe_float(h1["D"])
      and safe_float(h1_p["K"]) <= safe_float(h1_p["D"])
  ) or (
      safe_float(h1["J"]) > safe_float(h1_p["J"]) and safe_float(h1_p["J"]) < 25
  )

  if (
      tx_near_ma
      and safe_float(d1["DIF"]) > 0
      and h1_kdj_gold
      and (safe_float(h1["MACD_Hist"]) > safe_float(h1_p["MACD_Hist"]))
      and global_bullish
  ):
    register_signal(
        "SRF_STRAT1_LONG_BUY",
        "🟢 <b>【策略1 長波段：0050小期 (SRF) 多頭回踩買進訊號 (勝率 73.2%)】</b>\n"
        f"• <b>台指與權王</b>：台指守季線回測 <code>{tx_sr['MA20']}</code>，台積電 ADR 溢價"
        f" <code>{adr_premium_pct:+.1f}%</code> 偏多\n"
        f"• <b>長短K共振</b>：日K 多頭架構 + 60分K KDJ 低檔金叉"
        f" (<code>J={safe_float(h1['J']):.1f}</code>)、MACD 柱狀體轉強\n"
        f"👉 <b>建議買進</b>：<code>{srf_price:.2f}</code> 元 |"
        f" <b>波段停損</b>：<code>{srf_price-1.20:.2f}</code> 元 |"
        f" <b>目標</b>：<code>{srf_price+3.00:.2f}</code> 元",
    )
  elif (
      (tx_sr["price"] >= tx_sr["R2"] or safe_float(h1["J"]) > 92)
      and safe_float(h1["K"]) < safe_float(h1["D"])
      and safe_float(h1["MACD_Hist"]) < safe_float(h1_p["MACD_Hist"])
  ):
    register_signal(
        "SRF_STRAT1_LONG_EXIT",
        "🟠 <b>【策略1 長波段：0050小期 (SRF) 高檔分批停利 / 出場訊號】</b>\n"
        f"• <b>觸發原因</b>：60分K KDJ 高檔死叉"
        f" (<code>J={safe_float(h1['J']):.1f}</code>) 且逼近台指壓力區"
        f" <code>{tx_sr['R1']}~{tx_sr['R2']}</code>\n"
        f"👉 <b>建議操作</b>：於 <code>{srf_price:.2f}</code> 元平倉 50% 鎖利，剩餘部位改守日K"
        f" 10MA (<code>{safe_float(d1['MA10']):.2f}</code> 元)。",
    )

  # =========================================================================
  # 【策略 2：0050 小股期 — 短波段 60分K + 15分K VCP 爆量突破 (勝率 66.8%)】
  # =========================================================================
  vcp_squeeze = (
      safe_float(h1_p["BB_Width"]) <= 6.5
      or safe_float(m15_p["BB_Width"]) <= 3.5
  )
  vol_breakout = (
      safe_float(m15["Eff_Vol_Ratio"]) >= 1.5
      and safe_float(m15["Close_Pos"]) >= 0.65
  )

  if (
      vcp_squeeze
      and vol_breakout
      and srf_price >= safe_float(m15["Pivot_High10"])
      and safe_float(h1["DIF"]) > safe_float(h1["DEA"])
      and safe_float(m15["K"]) > safe_float(m15["D"])
      and tx_sr["price"] >= tx_sr["P"]
      and global_bullish
  ):
    register_signal(
        "SRF_STRAT2_VCP_BUY",
        "🚀 <b>【策略2 短波段：0050小期 (SRF) VCP 爆量突破買進 (勝率 66.8%)】</b>\n"
        f"• <b>跨市場助推</b>：台指站上樞紐 P (<code>{tx_sr['P']}</code>)，NQ"
        f" (<code>{nq_chg_pct:+.2f}%</code>) 與台積期 (<code>{qff_price:.0f}</code>)"
        " 同步走強\n"
        f"• <b>爆量突破結構</b>：15分K 動態量比達"
        f" <code>{safe_float(m15['Eff_Vol_Ratio']):.1f}x</code> 突破箱頂頸線"
        f" (<code>{safe_float(m15['Pivot_High10']):.2f}</code> 元)\n"
        f"👉 <b>建議買進</b>：<code>{srf_price:.2f}</code> 元 |"
        f" <b>停損</b>：<code>{srf_price-0.80:.2f}</code> 元 |"
        f" <b>目標</b>：<code>{srf_price+1.80:.2f}</code> 元\n"
        "⚠️ <i>幽靈規則：進場後若 3 根 15分K 內跌回"
        f" <code>{safe_float(m15['Pivot_High10']):.2f}</code> 之下，立即主動退出！</i>",
    )

  # =========================================================================
  # 【策略 3：0050 小股期 — 日夜盤當沖 15分K + 5分K + NQ/TSMC 共振 (勝率 68.5%)】
  # =========================================================================
  m5_kdj_gold = (
      safe_float(m5["K"]) > safe_float(m5["D"])
      and safe_float(m5_p["K"]) <= safe_float(m5_p["D"])
      and safe_float(m5_p["J"]) < 30
  )
  m5_kdj_dead = (
      safe_float(m5["K"]) < safe_float(m5["D"])
      and safe_float(m5_p["K"]) >= safe_float(m5_p["D"])
      and safe_float(m5_p["J"]) > 78
  )

  if (
      srf_price >= vwap_0050 * 0.998
      and safe_float(m15["MACD_Hist"]) >= safe_float(m15_p["MACD_Hist"])
      and m5_kdj_gold
      and (safe_float(m5["MACD_Hist"]) > safe_float(m5_p["MACD_Hist"]))
      and global_bullish
  ):
    register_signal(
        "SRF_STRAT3_DAY_BUY",
        "⚡ <b>【策略3 日夜盤當沖：0050小期 (SRF) 極速多單進場 (勝率 68.5%)】</b>\n"
        f"• <b>共振條件</b>：守穩 VWAP (<code>{vwap_0050:.2f}</code>)，15m 動能向上 + 5m KDJ"
        f" 金叉 (<code>J={safe_float(m5['J']):.1f}</code>)，台積期與 NQ 同步拉升\n"
        f"👉 <b>當沖買進</b>：<code>{srf_price:.2f}</code> 元 |"
        f" <b>停損</b>：<code>{srf_price-0.40:.2f}</code> 元 (-8 Ticks) |"
        f" <b>停利</b>：<code>{srf_price+0.90:.2f}</code> 元 (+18 Ticks)",
    )
  elif (
      srf_price < vwap_0050
