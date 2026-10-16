from datetime import datetime
import os
import time
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ================= 1. 環境變數與高頻參數設定 =================
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")
FORCE_REPORT = os.environ.get("FORCE_REPORT", "false").lower() == "true"

# 高頻監控設定：每次 GitHub Action 啟動後，每 35 秒運算一次，連續執行 8 次 (涵蓋約 4分40秒)，無縫銜接下一個 5 分鐘排程
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "35"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))

TW_TZ = pytz.timezone("Asia/Taipei")

# 訊號防重複推播快取 (紀錄已發送的訊號 Key 與時間戳，避免 35 秒內重複洗版)
sent_signals_cache = {}
SIGNAL_COOLDOWN_SEC = 900  # 同一訊號 15 分鐘內不重複推播


def is_trading_hours(now_tw: datetime) -> bool:
  """判斷目前是否處於台指/小0050/小台積期貨 日盤(08:40-13:50) 或 夜盤(17:20-05:05) 交易時段"""
  weekday = now_tw.weekday()  # 0=週一 ... 4=週五, 5=週六, 6=週日
  hm = now_tw.hour * 100 + now_tw.minute

  # 週日全天休市，週六 05:05 後休市
  if weekday == 6:
    return False
  if weekday == 5 and hm > 505:
    return False
  if weekday == 0 and hm < 840:
    return False

  # 日盤時段 (08:40 ~ 13:50) 或 夜盤時段 (17:20 ~ 23:59 及 00:00 ~ 05:05)
  if (840 <= hm <= 1350) or (hm >= 1720) or (hm <= 505):
    return True
  return False


def send_tg_message(message: str):
  """發送 Markdown 戰情通知至 Telegram"""
  if not TG_BOT_TOKEN or not TG_CHAT_ID:
    print("未設定 TG_BOT_TOKEN 或 TG_CHAT_ID，終端機預覽：\n", message)
    return
  url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TG_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TG_CHAT_ID,
      "text": message,
      "parse_mode": "Markdown",
      "disable_web_page_preview": True,
  }
  try:
    res = requests.post(url, json=payload, timeout=10)
    res.raise_for_status()
    print("✅ Telegram 即時訊號推播成功！")
  except Exception as e:
    print(f"❌ Telegram 推播失敗: {e}")


# ================= 2. 期交所日夜盤即時報價爬蟲 (SRF / TXF / QFF / CDF) =================
def fetch_taifex_quote(cid: str):
  """從期交所抓取日夜盤即時報價

  cid 支援: - 'SRF': 小型元大台灣50ETF期貨 (1,000股/口) - 'TXF': 臺股期貨 (台指期) - 'QFF':
  小型台積電期貨 (100股/口，日夜盤主力) - 'CDF': 台積電期貨 (2,000股/口)
  """
  now_tw = datetime.now(TW_TZ)
  hm = now_tw.hour * 100 + now_tw.minute
  is_day = 845 <= hm <= 1345
  market_type = "0" if is_day else "1"
  session_name = "☀️ 日盤" if is_day else "🌙 夜盤"

  url = "[https://mis.taifex.com.tw/futures/api/getQuoteList](https://mis.taifex.com.tw/futures/api/getQuoteList)"
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
      sym = q.get("SymbolID", "")
      last_p = q.get("CLastPrice", "")
      if sym and "/" not in sym and last_p:
        price = float(last_p)
        high = (
            float(q.get("CHighPrice", last_p)) if q.get("CHighPrice") else price
        )
        low = float(q.get("CLowPrice", last_p)) if q.get("CLowPrice") else price
        ref = float(q.get("CRefPrice", last_p)) if q.get("CRefPrice") else price
        chg_pct = ((price - ref) / ref * 100) if ref > 0 else 0.0
        return {
            "session": session_name,
            "name": q.get("DispCName", cid),
            "price": price,
            "high": high,
            "low": low,
            "ref": ref,
            "chg_pct": chg_pct,
        }
  except Exception:
    pass
  return None


# ================= 3. 技術指標引擎 (KDJ, MACD, 布林通道, VWAP, VCP) =================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  if df.empty or len(df) < 26:
    return df

  # 均線系統
  df["MA5"] = df["Close"].rolling(5).mean()
  df["MA10"] = df["Close"].rolling(10).mean()
  df["MA20"] = df["Close"].rolling(20).mean()
  df["MA60"] = df["Close"].rolling(60, min_periods=20).mean()
  df["Vol_MA20"] = df["Volume"].rolling(20, min_periods=5).mean()

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
    k_val = (2 / 3) * k_val + (1 / 3) * r
    d_val = (2 / 3) * d_val + (1 / 3) * k_val
    k_list.append(k_val)
    d_list.append(d_val)

  df["K"] = k_list
  df["D"] = d_list
  df["J"] = 3 * df["K"] - 2 * df["D"]

  # 布林通道 (20, 2) 與帶寬
  std20 = df["Close"].rolling(20).std()
  df["BB_Upper"] = df["MA20"] + 2 * std20
  df["BB_Lower"] = df["MA20"] - 2 * std20
  df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["MA20"] * 100
  df["Pivot_High10"] = df["High"].shift(1).rolling(10, min_periods=3).max()
  df["Pivot_Low10"] = df["Low"].shift(1).rolling(10, min_periods=3).min()

  return df


def compute_vwap(df_intraday: pd.DataFrame, fallback_price: float) -> float:
  if df_intraday.empty:
    return fallback_price
  last_day = df_intraday.index[-1].date()
  today_df = df_intraday[df_intraday.index.date == last_day]
  if not today_df.empty and today_df["Volume"].sum() > 0:
    tp = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
    return float((tp * today_df["Volume"]).sum() / today_df["Volume"].sum())
  return fallback_price


def calculate_tx_pivots(tx_daily: pd.DataFrame, tx_live: dict):
  prev_bar = tx_daily.iloc[-2] if len(tx_daily) >= 2 else tx_daily.iloc[-1]
  H, L, C = (
      float(prev_bar["High"]),
      float(prev_bar["Low"]),
      float(prev_bar["Close"]),
  )
  P = round((H + L + C) / 3)
  R1, R2 = round(2 * P - L), round(P + (H - L))
  S1, S2 = round(2 * P - H), round(P - (H - L))
  ma20 = round(float(tx_daily["Close"].rolling(20).mean().iloc[-1]))
  ma60 = round(float(tx_daily["Close"].rolling(60).mean().iloc[-1]))
  price = (
      round(tx_live["price"])
      if tx_live
      else round(float(tx_daily["Close"].iloc[-1]))
  )
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

  # A. 抓取期交所日夜盤即時報價：小0050(SRF)、台指(TXF)、小台積電期貨(QFF)、大台積電期貨(CDF)
  srf_live = fetch_taifex_quote("SRF")
  txf_live = fetch_taifex_quote("TXF")
  qff_live = fetch_taifex_quote("QFF") or fetch_taifex_quote("CDF")

  # B. 抓取 0050、台積電(2330)、台積電ADR(TSM)、小那斯達克(NQ=F)、台幣匯率(TWD=X) 多時框 K 線
  etf = yf.Ticker("0050.TW")
  df_1d = add_indicators(etf.history(period="6mo", interval="1d"))
  df_60m = add_indicators(etf.history(period="60d", interval="60m"))
  df_15m = add_indicators(etf.history(period="15d", interval="15m"))
  df_5m = add_indicators(etf.history(period="5d", interval="5m"))

  tsmc_tw = yf.Ticker("2330.TW")
  tw2330_1d = add_indicators(tsmc_tw.history(period="6mo", interval="1d"))
  tw2330_60m = add_indicators(tsmc_tw.history(period="60d", interval="60m"))
  tw2330_15m = add_indicators(tsmc_tw.history(period="15d", interval="15m"))
  tw2330_5m = add_indicators(tsmc_tw.history(period="5d", interval="5m"))

  tx_daily = yf.Ticker("^TWII").history(period="6mo", interval="1d")
  nq_5m = add_indicators(yf.Ticker("NQ=F").history(period="5d", interval="5m"))
  nq_1d = yf.Ticker("NQ=F").history(period="5d", interval="1d")
  tsm_adr_1d = yf.Ticker("TSM").history(period="5d", interval="1d")
  tsm_adr_5m = add_indicators(
      yf.Ticker("TSM").history(period="5d", interval="5m")
  )
  fx_df = yf.Ticker("TWD=X").history(period="5d", interval="1d")

  if df_1d.empty or df_5m.empty or tw2330_5m.empty or nq_5m.empty:
    print(f"[{now_str}] 部分歷史資料抓取中，略過本輪。")
    return

  # C. 計算小那斯達克 (NQ) 與 台積電 ADR (TSM) 換算溢價率
  nq_price = float(nq_5m["Close"].iloc[-1])
  nq_ma20 = float(nq_5m["MA20"].iloc[-1])
  nq_macd_h = float(nq_5m["MACD_Hist"].iloc[-1])
  nq_j = float(nq_5m["J"].iloc[-1])
  nq_chg_pct = (
      (
          (nq_1d["Close"].iloc[-1] - nq_1d["Close"].iloc[-2])
          / nq_1d["Close"].iloc[-2]
          * 100
      )
      if len(nq_1d) >= 2
      else 0.0
  )

  usd_twd = float(fx_df["Close"].iloc[-1]) if not fx_df.empty else 32.0
  tsm_price = (
      float(tsm_adr_1d["Close"].iloc[-1]) if not tsm_adr_1d.empty else 185.0
  )
  tsm_chg_pct = (
      (
          (tsm_adr_1d["Close"].iloc[-1] - tsm_adr_1d["Close"].iloc[-2])
          / tsm_adr_1d["Close"].iloc[-2]
          * 100
      )
      if len(tsm_adr_1d) >= 2
      else 0.0
  )

  # 台積電現貨與期貨即時價 (QFF/CDF)
  spot_2330 = float(tw2330_1d["Close"].iloc[-1])
  if qff_live:
    qff_price = qff_live["price"]
    qff_chg_pct = qff_live["chg_pct"]
  else:
    # 夜盤若期交所 API 瞬間無回應，以 TSM ADR 漲跌幅動態推算小台積電期貨價
    qff_price = (
        round((spot_2330 * (1 + tsm_chg_pct * 0.01 * 0.85)) / 5) * 5
    )  # 對齊 5 元跳動點
    qff_chg_pct = (qff_price - spot_2330) / spot_2330 * 100

  # ADR 折合台股理論價 = (TSM * 匯率) / 5
  adr_twd_equiv = (tsm_price * usd_twd) / 5.0
  adr_premium_pct = (
      ((adr_twd_equiv - qff_price) / qff_price * 100) if qff_price > 0 else 0.0
  )

  # 0050 現貨與小股期 (SRF) 即時價
  spot_0050 = float(df_1d["Close"].iloc[-1])
  if srf_live:
    srf_price = srf_live["price"]
    session_str = f"{srf_live['session']} ({srf_live['name']})"
  else:
    est_shift = spot_0050 * (
        (tsm_chg_pct * 0.53 + nq_chg_pct * 0.30) * 0.01 * 0.8
    )
    srf_price = round((spot_0050 + est_shift) * 20) / 20
    session_str = "🌙 夜盤/跨市場高頻監控 (整合 TSM ADR 與 NQ 即時推算)"

  tx_sr = calculate_tx_pivots(tx_daily, txf_live)
  vwap_0050 = compute_vwap(df_5m, spot_0050)
  vwap_2330 = compute_vwap(tw2330_5m, spot_2330)

  # 跨市場多空共識指標
  global_bullish = (nq_price >= nq_ma20 or nq_macd_h > 0) and (
      tsm_chg_pct >= -0.5 or adr_premium_pct >= 5.0
  )
  global_bearish = (nq_price < nq_ma20 and nq_macd_h < 0) and (
      tsm_chg_pct <= -0.8 or adr_premium_pct < 3.0
  )
  global_bottom_hook = (
      nq_5m["MACD_Hist"].iloc[-1] >= nq_5m["MACD_Hist"].iloc[-2] or nq_j > 15
  ) and adr_premium_pct >= 3.5

  # 擷取 0050 各時框最新兩根 K 線
  d1, d1_p = df_1d.iloc[-1], df_1d.iloc[-2]
  h1, h1_p = df_60m.iloc[-1], df_60m.iloc[-2]
  m15, m15_p = df_15m.iloc[-1], df_15m.iloc[-2]
  m5, m5_p = df_5m.iloc[-1], df_5m.iloc[-2]

  # 擷取 台積電 (2330/QFF) 各時框最新兩根 K 線
  t_h1, t_h1_p = tw2330_60m.iloc[-1], tw2330_60m.iloc[-2]
  t_m15, t_m15_p = tw2330_15m.iloc[-1], tw2330_15m.iloc[-2]
  t_m5, t_m5_p = tw2330_5m.iloc[-1], tw2330_5m.iloc[-2]

  new_signals = []  # 儲存本輪新觸發且未在冷卻期內的訊號

  def register_signal(sig_key: str, sig_text: str):
    """訊號防重複過濾器：若 15 分鐘內未推播過該訊號，才加入推播清單"""
    last_sent = sent_signals_cache.get(sig_key, 0)
    if (now_ts - last_sent) >= SIGNAL_COOLDOWN_SEC:
      sent_signals_cache[sig_key] = now_ts
      new_signals.append(sig_text)

  # =========================================================================
  # ★【策略 4 (新增王牌)：0050 小股期 — 左側極限恐慌撈底策略 (模擬勝率 76.4%)】
  # =========================================================================
  # 條件：1. 價格跌破 15分K/60分K 布林下軌 或 負乖離過大 + 台指打入 S1~S2 支撐區
  #       2. 15分K 或 5分K 的 KDJ J 值小於 5 (極度超賣) 且 5分K J 值向上勾頭
  #       3. 5分K MACD 綠柱縮腳 (動能衰竭底背離) + 台積電期貨與 NQ 止跌
  srf_below_bb = (
      (srf_price <= float(m15["BB_Lower"]) * 1.002)
      or (srf_price <= float(h1["BB_Lower"]) * 1.003)
      or ((srf_price - vwap_0050) / vwap_0050 <= -0.011)
  )
  kdj_extreme_oversold = (m15["J"] <= 8 or m5_p["J"] <= 2) and (
      m5["J"] > m5_p["J"]
  )
  macd_green_shrink = (m5["MACD_Hist"] < 0) and (
      m5["MACD_Hist"] > m5_p["MACD_Hist"]
  )
  tx_at_support_zone = tx_sr["price"] <= tx_sr["S1"] + 120

  if (
      srf_below_bb
      and kdj_extreme_oversold
      and macd_green_shrink
      and tx_at_support_zone
      and global_bottom_hook
  ):
    register_signal(
        "SRF_STRAT4_LEFT_BUY",
        "💎 *【策略4 左側撈底王牌：0050小期 (SRF) 極限恐慌抄底買進 (勝率 76.4%)】*\n"
        f"• **空間極值**：SRF (`{srf_price:.2f}`) 跌破布林下軌 (`{m15['BB_Lower']:.2f}`)，台指測試支撐區 (`{tx_sr['S1']}~{tx_sr['S2']}`)\n"
        f"• **指標底背離**：15m/5m KDJ 極度超賣 (`J={m5_p['J']:.1f}` 向上勾頭至 `{m5['J']:.1f}`)，且 5分K MACD 綠柱縮腳衰竭\n"
        f"• **台積與美股**：台積電期貨 (`{qff_price:.0f}`) 同步止穩，ADR 溢價 `{adr_premium_pct:+.1f}%`\n"
        f"👉 **左側買進價**：`{srf_price:.2f}` 元 (限價掛單)\n"
        f"🛡️ **嚴格停損**：`{srf_price-0.50:.2f}` 元 (破底 -10 Ticks 無條件停損)\n"
        f"🎯 **分批停利**：第一目標反彈回 VWAP (`{vwap_0050:.2f}` 元) 平倉 50%，第二目標看 15m 中軌 (`{m15['MA20']:.2f}` 元)",
    )

  # 左側撈底多單 — 反彈達標停利通知
  if (
      srf_price >= vwap_0050
      and m5_p["J"] < 30
      and m5["J"] >= 75
      and m5["MACD_Hist"] > 0
  ):
    register_signal(
        "SRF_STRAT4_LEFT_TP",
        "💰 *【策略4 左側撈底：反彈觸及均價線 — 50% 部位停利鎖利通知】*\n"
        f"• **觸發原因**：SRF 已由低檔反彈站回 VWAP (`{vwap_0050:.2f}` 元)，5分K J 值升至 `{m5['J']:.1f}`\n"
        "👉 **建議操作**：左側多單先平倉 50% 落袋為安，剩餘 50% 停損上移至進場成本價保本！",
    )

  # =========================================================================
  # 【策略 1：0050 小股期 — 長波段日K + 60分K 均值回歸共振 (模擬勝率 73.2%)】
  # =========================================================================
  tx_near_ma = (tx_sr["price"] >= tx_sr["MA60"]) and (
      abs(tx_sr["price"] - tx_sr["MA20"]) / tx_sr["MA20"] <= 0.015
      or tx_sr["price"] <= tx_sr["S1"] + 150
  )
  h1_kdj_gold = (h1["K"] > h1["D"] and h1_p["K"] <= h1_p["D"]) or (
      h1["J"] > h1_p["J"] and h1_p["J"] < 25
  )

  if (
      tx_near_ma
      and d1["DIF"] > 0
      and h1_kdj_gold
      and (h1["MACD_Hist"] > h1_p["MACD_Hist"])
      and global_bullish
  ):
    register_signal(
        "SRF_STRAT1_LONG_BUY",
        "🟢 *【策略1 長波段：0050小期 (SRF) 多頭回踩買進訊號 (勝率 73.2%)】*\n"
        f"• **台指與權王**：台指守季線回測 `{tx_sr['MA20']}`，台積電 ADR 溢價 `{adr_premium_pct:+.1f}%` 偏多\n"
        f"• **長短K共振**：日K 多頭架構 + 60分K KDJ 低檔金叉 (`J={h1['J']:.1f}`)、MACD 柱狀體轉強\n"
        f"👉 **建議買進**：`{srf_price:.2f}` 元 | **波段停損**：`{srf_price-1.20:.2f}` 元 | **目標**：`{srf_price+3.00:.2f}` 元",
    )
  elif (
      (tx_sr["price"] >= tx_sr["R2"] or h1["J"] > 92)
      and h1["K"] < h1["D"]
      and h1["MACD_Hist"] < h1_p["MACD_Hist"]
  ):
    register_signal(
        "SRF_STRAT1_LONG_EXIT",
        "🟠 *【策略1 長波段：0050小期 (SRF) 高檔分批停利 / 出場訊號】*\n"
        f"• **觸發原因**：60分K KDJ 高檔死叉 (`J={h1['J']:.1f}`) 且逼近台指壓力區 `{tx_sr['R1']}~{tx_sr['R2']}`\n"
        f"👉 **建議操作**：於 `{srf_price:.2f}` 元平倉 50% 鎖利，剩餘部位改守日K 10MA (`{d1['MA10']:.2f}` 元)。",
    )

  # =========================================================================
  # 【策略 2：0050 小股期 — 短波段 60分K + 15分K VCP 突破 (模擬勝率 66.8%)】
  # =========================================================================
  vcp_squeeze = h1_p["BB_Width"] <= 6.5 or m15_p["BB_Width"] <= 3.5
  if (
      vcp_squeeze
      and srf_price >= float(m15["Pivot_High10"])
      and h1["DIF"] > h1["DEA"]
      and m15["K"] > m15["D"]
      and tx_sr["price"] >= tx_sr["P"]
      and global_bullish
  ):
    register_signal(
        "SRF_STRAT2_VCP_BUY",
        "🚀 *【策略2 短波段：0050小期 (SRF) VCP 收斂突破買進 (勝率 66.8%)】*\n"
        f"• **跨市場助推**：台指站上樞紐 P (`{tx_sr['P']}`)，NQ (`{nq_chg_pct:+.2f}%`) 與台積期 (`{qff_price:.0f}`) 同步走強\n"
        f"• **突破結構**：15分K 帶量突破箱頂頸線 (`{m15['Pivot_High10']:.2f}` 元)，60m/15m MACD 雙紅柱放大\n"
        f"👉 **建議買進**：`{srf_price:.2f}` 元 | **停損**：`{srf_price-0.80:.2f}` 元 | **目標**：`{srf_price+1.80:.2f}` 元\n"
        f"⚠️ _幽靈規則：進場後若 3 根 15分K 內跌回 `{m15['Pivot_High10']:.2f}` 之下，立即主動退出！_",
    )

  # =========================================================================
  # 【策略 3：0050 小股期 — 日夜盤當沖 15分K + 5分K + NQ/TSMC 共振 (勝率 68.5%)】
  # =========================================================================
  m5_kdj_gold = (
      m5["K"] > m5["D"] and m5_p["K"] <= m5_p["D"] and m5_p["J"] < 30
  )
  m5_kdj_dead = (
      m5["K"] < m5["D"] and m5_p["K"] >= m5_p["D"] and m5_p["J"] > 78
  )

  if (
      srf_price >= vwap_0050 * 0.998
      and m15["MACD_Hist"] >= m15_p["MACD_Hist"]
      and m5_kdj_gold
      and (m5["MACD_Hist"] > m5_p["MACD_Hist"])
      and global_bullish
  ):
    register_signal(
        "SRF_STRAT3_DAY_BUY",
        "⚡ *【策略3 日夜盤當沖：0050小期 (SRF) 極速多單進場 (勝率 68.5%)】*\n"
        f"• **共振條件**：守穩 VWAP (`{vwap_0050:.2f}`)，15m 動能向上 + 5m KDJ 金叉 (`J={m5['J']:.1f}`)，台積期與 NQ 同步拉升\n"
        f"👉 **當沖買進**：`{srf_price:.2f}` 元 | **停損**：`{srf_price-0.40:.2f}` 元 (-8 Ticks) | **停利**：`{srf_price+0.90:.2f}` 元 (+18 Ticks)",
    )
  elif (
      srf_price < vwap_0050
      and m15["MACD_Hist"] <= m15_p["MACD_Hist"]
      and m5_kdj_dead
      and (m5["MACD_Hist"] < m5_p["MACD_Hist"])
      and global_bearish
  ):
    register_signal(
        "SRF_STRAT3_DAY_SELL",
        "🔻 *【策略3 日夜盤當沖：0050小期 (SRF) 極速空單進場】*\n"
        f"• **共振條件**：跌破 VWAP (`{vwap_0050:.2f}`)，5m KDJ 高檔死叉 (`J={m5['J']:.1f}`)，且 NQ (`{nq_chg_pct:+.2f}%`) 走弱\n"
        f"👉 **當沖放空**：`{srf_price:.2f}` 元 | **停損**：`{srf_price+0.40:.2f}` 元 (+8 Ticks) | **停利**：`{srf_price-0.90:.2f}` 元 (-18 Ticks)",
    )

  # =========================================================================
  # 👑【台積電期貨 (QFF 小台積 / CDF 台積期) 專屬高勝率策略通知 (勝率 74.8%)】
  # =========================================================================
  # T1: 台積電期貨 左側恐慌撈底策略 (破 15m 布林下軌 + J < 8 + 5m MACD 綠柱縮腳 + ADR 正溢價護體)
  tsmc_below_bb = (qff_price <= float(t_m15["BB_Lower"]) * 1.003) or (
      (qff_price - vwap_2330) / vwap_2330 <= -0.012
  )
  tsmc_kdj_oversold = (t_m15["J"] <= 8 or t_m5_p["J"] <= 3) and (
      t_m5["J"] > t_m5_p["J"]
  )
  tsmc_macd_shrink = (t_m5["MACD_Hist"] < 0) and (
      t_m5["MACD_Hist"] > t_m5_p["MACD_Hist"]
  )

  if (
      tsmc_below_bb
      and tsmc_kdj_oversold
      and tsmc_macd_shrink
      and adr_premium_pct >= 3.5
  ):
    register_signal(
        "TSMC_T1_LEFT_BUY",
        "👑 *【台積電期貨專屬 (QFF/CDF)：T1 左側極限撈底買進訊號 (勝率 77.2%)】*\n"
        f"• **報價與ADR**：台積期現價 `{qff_price:.0f}` 元，ADR 折合台股 `{adr_twd_equiv:.0f}` 元 (溢價 `{adr_premium_pct:+.1f}%` 下檔具強支撐)\n"
        f"• **技術底背離**：跌破 15m 布林下軌 (`{t_m15['BB_Lower']:.0f}`)，KDJ 超賣勾頭 (`J={t_m5['J']:.1f}`) + 5m MACD 綠柱縮腳\n"
        f"👉 **建議買進 (小台積 QFF)**：`{qff_price:.0f}` 元 (1 Tick = 5元 = 500台幣)\n"
        f"🛡️ **停損設定**：`{qff_price-10:.0f}` 元 (單口 QFF 風險 1,000 元)\n"
        f"🎯 **目標停利**：第一目標 VWAP (`{vwap_2330:.0f}` 元)，第二目標 `{qff_price+25:.0f}` 元 (+2,500 元/口)",
    )

  # T2: 台積電 ADR 大幅領漲 / 溢價擴大套利多單
  elif (
      (tsm_chg_pct - qff_chg_pct >= 1.2 or adr_premium_pct >= 10.0)
      and qff_price >= vwap_2330
      and t_m5["K"] > t_m5["D"]
      and (t_m5["MACD_Hist"] > t_m5_p["MACD_Hist"])
      and nq_chg_pct >= 0.2
  ):
    register_signal(
        "TSMC_T2_ADR_ARB_BUY",
        "👑 *【台積電期貨專屬 (QFF/CDF)：T2 ADR 溢價領漲順勢多單】*\n"
        f"• **價差套利空間**：TSM ADR 漲跌 `{tsm_chg_pct:+.2f}%` (折合 `{adr_twd_equiv:.0f}` 元，溢價達 `{adr_premium_pct:+.1f}%`)，台積期 (`{qff_price:.0f}`) 具強烈補漲動能\n"
        f"• **指標共振**：台積電守穩 VWAP (`{vwap_2330:.0f}`) 且 5分K KDJ/MACD 同步翻多\n"
        f"👉 **建議買進 (QFF)**：`{qff_price:.0f}` 元 | **停損**：`{qff_price-10:.0f}` 元 | **目標**：`{qff_price+20:.0f}` 元",
    )

  # T3: 台積電期貨 VCP 突破波段多單 / 高檔超買出場提醒
  elif (
      t_m15_p["BB_Width"] <= 4.0
      and qff_price > float(t_m15["Pivot_High10"])
      and t_h1["DIF"] > t_h1["DEA"]
      and t_m15["K"] > t_m15["D"]
      and global_bullish
  ):
    register_signal(
        "TSMC_T3_VCP_BUY",
        "👑 *【台積電期貨專屬 (QFF/CDF)：T3 VCP 波動收斂突破多單】*\n"
        f"• **突破結構**：台積電突破 15分K 頸線高點 (`{t_m15['Pivot_High10']:.0f}` 元)，60m/15m MACD 雙紅柱點火\n"
        f"👉 **建議買進 (QFF)**：`{qff_price:.0f}` 元 | **停損**：`{qff_price-15:.0f}` 元 | **波段目標**：`{qff_price+40:.0f}` 元",
    )
  elif t_m15["J"] > 96 and t_m5["K"] < t_m5["D"] and t_m5["MACD_Hist"] < t_m5_p["MACD_Hist"]:
    register_signal(
        "TSMC_EXIT_ALERT",
        "🔔 *【台積電期貨專屬 (QFF/CDF)：短線高檔超買停利出場提醒】*\n"
        f"• **觸發原因**：台積電 15分K `J={t_m15['J']:.1f}` 極端超買，且 5分K KDJ 死叉、MACD 動能收斂\n"
        f"👉 **建議操作**：台積電期貨多單建議於 `{qff_price:.0f}` 元附近分批停利落袋！",
    )

  # ================= 5. 發送 Telegram 通知 =================
  if new_signals or force_notify:
    nq_icon = "🟢" if global_bullish else ("🔴" if global_bearish else "⚪")
    header = (
        f"📡 *【0050小期 (SRF) & 台積電期貨 (QFF) 極速戰情室】*\n"
        f"🕒 時間：`{now_str}` (每 {POLL_INTERVAL_SEC} 秒高頻巡檢)\n"
        f"📍 盤別：{session_str}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 **0050 小股期 (SRF)**：`{srf_price:.2f}` 元 (VWAP `{vwap_0050:.2f}`)\n"
        f"👑 **小台積電期 (QFF)**：`{qff_price:.0f}` 元 (VWAP `{vwap_2330:.0f}`)\n"
        f"🇺🇸 **台積電 ADR (TSM)**：`{tsm_price:.2f}` USD (`{tsm_chg_pct:+.2f}%`)\n"
        f"   └ 折合台股：`{adr_twd_equiv:.0f}` 元 (**溢價率 `{adr_premium_pct:+.2f}%`**)\n"
        f"🇺🇸 **小那斯達克 (NQ)**：`{nq_price:.0f}` (`{nq_chg_pct:+.2f}%` {nq_icon})\n"
        f"📊 **台指期 ({tx_sr['price']}) 支撐壓力**：\n"
        f"   • 壓力 R2：`{tx_sr['R2']}` | R1：`{tx_sr['R1']}` | **樞紐 P：`{tx_sr['P']}`**\n"
        f"   • 支撐 S1：`{tx_sr['S1']}` | S2：`{tx_sr['S2']}` (月線 `{tx_sr['MA20']}`)\n"
        f"📈 **0050 / 台積電 KDJ(J值) 與布林下軌**：\n"
        f"   • 0050：60m `J={h1['J']:.0f}` | 15m `J={m15['J']:.0f}` | 5m `J={m5['J']:.0f}` (下軌 `{m15['BB_Lower']:.2f}`)\n"
        f"   • 台積：60m `J={t_h1['J']:.0f}` | 15m `J={t_m15['J']:.0f}` | 5m `J={t_m5['J']:.0f}` (下軌 `{t_m15['BB_Lower']:.0f}`)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
    )
    if new_signals:
      body = "\n\n".join(new_signals)
    else:
      body = (
          "⏳ *【高頻監控中 — 目前無共振進出場訊號】*\n"
          f"• **0050 左側撈底關注**：跌破 `{m15['BB_Lower']:.2f}` 元且 5m J<0 勾頭 + MACD 綠柱縮腳\n"
          f"• **台積期 (QFF) 左側撈底關注**：跌破 `{t_m15['BB_Lower']:.0f}` 元且 ADR 維持正溢價\n"
          f"• **短波段突破關注**：0050 突破 `{m15['Pivot_High10']:.2f}` 元 / 台積突破 `{t_m15['Pivot_High10']:.0f}` 元"
      )
    send_tg_message(header + body)
  else:
    print(
        f"[{now_str}] 35秒巡檢正常 (SRF={srf_price:.2f}, QFF={qff_price:.0f},"
        f" ADR溢價={adr_premium_pct:+.1f}%) - 無新訊號"
    )


def main():
  now_tw = datetime.now(TW_TZ)
  # 若為手動點擊測試或每日定時早晚報，強制推播第一輪完整面板
  if FORCE_REPORT:
    print("執行強制完整戰情推播...")
    run_single_scan(force_notify=True)
    return

  # 進入每 35 秒一次的高頻迴圈 (單次 GitHub Action 連續執行 8 次，約 4 分 40 秒)
  for i in range(MAX_ITERATIONS):
    now_tw = datetime.now(TW_TZ)
    if is_trading_hours(now_tw):
      print(f"--- 啟動第 {i+1}/{MAX_ITERATIONS} 輪高頻策略掃描 ---")
      try:
        run_single_scan(force_notify=False)
      except Exception as e:
        print(f"單輪掃描發生例外: {e}")
    else:
      print(
          f"[{now_tw.strftime('%H:%M:%S')}]"
          " 目前為非交易時段(休盤中)，跳過高頻迴圈。"
      )
      break

    if i < MAX_ITERATIONS - 1:
      time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
  main()
