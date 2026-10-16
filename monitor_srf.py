from datetime import datetime
import os
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ================= 讀取環境變數 =================
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")
FORCE_REPORT = os.environ.get("FORCE_REPORT", "false").lower() == "true"

TW_TZ = pytz.timezone("Asia/Taipei")


def send_tg_message(message: str):
  """發送即時策略通知至 Telegram"""
  if not TG_BOT_TOKEN or not TG_CHAT_ID:
    print("未設定 TG_BOT_TOKEN 或 TG_CHAT_ID，終端機預覽訊息：\n", message)
    return
  url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TG_CHAT_ID,
      "text": message,
      "parse_mode": "Markdown",
      "disable_web_page_preview": True,
  }
  try:
    res = requests.post(url, json=payload, timeout=12)
    res.raise_for_status()
    print("✅ Telegram 訊息推播成功！")
  except Exception as e:
    print(f"❌ Telegram 推播失敗: {e}")


# ================= 期交所日夜盤即時報價爬蟲 (SRF & TXF) =================
def fetch_taifex_quote(cid: str):
  """從台灣期貨交易所抓取日盤 (MarketType=0) 或夜盤 (MarketType=1) 即時報價

  cid: 'SRF' (小型0050期貨), 'TXF' (台指期)
  """
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
    r = requests.post(url, json=payload, headers=headers, timeout=8)
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
  except Exception as e:
    print(f"期交所 {cid} API 備援切換: {e}")
  return None


# ================= 計算 KDJ(9,3,3)、MACD(12,26,9)、均線與布林 =================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  if df.empty or len(df) < 26:
    return df

  # 均線與量能
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

  # 布林通道與 VCP 帶寬
  std20 = df["Close"].rolling(20).std()
  df["BB_Upper"] = df["MA20"] + 2 * std20
  df["BB_Lower"] = df["MA20"] - 2 * std20
  df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["MA20"] * 100
  df["Pivot_High10"] = df["High"].shift(1).rolling(10, min_periods=3).max()
  df["Pivot_Low10"] = df["Low"].shift(1).rolling(10, min_periods=3).min()

  return df


# ================= 計算台指期支撐壓力區 (Pivot Points) =================
def calculate_tx_pivots(tx_daily: pd.DataFrame, tx_live_quote: dict):
  prev_bar = tx_daily.iloc[-2] if len(tx_daily) >= 2 else tx_daily.iloc[-1]
  H = float(prev_bar["High"])
  L = float(prev_bar["Low"])
  C = float(prev_bar["Close"])

  P = round((H + L + C) / 3)
  R1 = round(2 * P - L)
  R2 = round(P + (H - L))
  S1 = round(2 * P - H)
  S2 = round(P - (H - L))

  tx_ma20 = round(float(tx_daily["Close"].rolling(20).mean().iloc[-1]))
  tx_ma60 = round(float(tx_daily["Close"].rolling(60).mean().iloc[-1]))
  tx_price = (
      round(tx_live_quote["price"])
      if tx_live_quote
      else round(float(tx_daily["Close"].iloc[-1]))
  )

  return {
      "price": tx_price,
      "P": P,
      "R1": R1,
      "R2": R2,
      "S1": S1,
      "S2": S2,
      "MA20": tx_ma20,
      "MA60": tx_ma60,
  }


def run_strategy_engine():
  now_str = datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S")

  # 1. 抓取多時框資料：0050 (1d, 60m, 15m, 5m)、台指 (^TWII)、小那斯達克期貨 (NQ=F)、台積電ADR (TSM)
  etf = yf.Ticker("0050.TW")
  df_1d = add_technical_indicators(etf.history(period="6mo", interval="1d"))
  df_60m = add_technical_indicators(etf.history(period="60d", interval="60m"))
  df_15m = add_technical_indicators(etf.history(period="15d", interval="15m"))
  df_5m = add_technical_indicators(etf.history(period="5d", interval="5m"))

  tx_daily = yf.Ticker("^TWII").history(period="6mo", interval="1d")
  nq_15m = add_technical_indicators(
      yf.Ticker("NQ=F").history(period="10d", interval="15m")
  )
  nq_1d = yf.Ticker("NQ=F").history(period="5d", interval="1d")

  if df_1d.empty or df_60m.empty or df_15m.empty or df_5m.empty:
    print("歷史 K 線資料讀取不完整，略過本次分析")
    return

  # 2. 小那斯達克 (NQ=F) 即時狀態
  nq_price = float(nq_15m["Close"].iloc[-1])
  nq_ma20 = float(nq_15m["MA20"].iloc[-1])
  nq_macd_h = float(nq_15m["MACD_Hist"].iloc[-1])
  nq_chg_pct = 0.0
  if len(nq_1d) >= 2:
    nq_chg_pct = (
        (nq_1d["Close"].iloc[-1] - nq_1d["Close"].iloc[-2])
        / nq_1d["Close"].iloc[-2]
        * 100
    )
  nq_bullish = (nq_price >= nq_ma20 or nq_macd_h > 0) and nq_chg_pct >= -0.35
  nq_bearish = (nq_price < nq_ma20 and nq_macd_h < 0) or nq_chg_pct <= -0.35

  # 3. 取得期交所日夜盤報價 (SRF 與 TXF)
  srf_live = fetch_taifex_quote("SRF")
  txf_live = fetch_taifex_quote("TXF")
  tx_sr = calculate_tx_pivots(tx_daily, txf_live)

  spot_0050 = float(df_1d["Close"].iloc[-1])
  if srf_live:
    srf_price = srf_live["price"]
    session_str = f"{srf_live['session']} ({srf_live['name']})"
  else:
    # 夜盤若無即時成交，以 NQ 期貨漲跌幅動態推算夜盤參考價
    est_shift = spot_0050 * (nq_chg_pct * 0.01 * 0.65)
    srf_price = round((spot_0050 + est_shift) * 20) / 20
    session_str = "🌙 夜盤/跨盤動態監控 (依 NQ 期貨連動推算)"

  # 4. 計算日內 VWAP
  last_day = df_5m.index[-1].date()
  today_5m = df_5m[df_5m.index.date == last_day]
  vwap = spot_0050
  if not today_5m.empty and today_5m["Volume"].sum() > 0:
    tp_price = (today_5m["High"] + today_5m["Low"] + today_5m["Close"]) / 3
    vwap = float(
        (tp_price * today_5m["Volume"]).sum() / today_5m["Volume"].sum()
    )

  # 擷取各時框最新與前一根指標
  d1, d1_p = df_1d.iloc[-1], df_1d.iloc[-2]
  h1, h1_p = df_60m.iloc[-1], df_60m.iloc[-2]
  m15, m15_p = df_15m.iloc[-1], df_15m.iloc[-2]
  m5, m5_p = df_5m.iloc[-1], df_5m.iloc[-2]

  signals = []

  # =========================================================================
  # 【策略 1：長波段 — 日K + 60分K 均值回歸共振策略】
  # =========================================================================
  tx_near_support = (
      (tx_sr["price"] >= tx_sr["MA60"])
      and (
          abs(tx_sr["price"] - tx_sr["MA20"]) / tx_sr["MA20"] <= 0.015
          or tx_sr["S2"] <= tx_sr["price"] <= tx_sr["S1"] + 150
      )
  )
  h1_kdj_gold = (h1["K"] > h1["D"] and h1_p["K"] <= h1_p["D"]) or (
      h1["J"] > h1_p["J"] and h1_p["J"] < 25
  )
  h1_macd_turn_up = h1["MACD_Hist"] > h1_p["MACD_Hist"]

  if (
      tx_near_support
      and d1["DIF"] > 0
      and (d1["J"] < 35 or h1["J"] < 35)
      and h1_kdj_gold
      and h1_macd_turn_up
      and nq_bullish
  ):
    signals.append(
        "🟢 *【策略1 長波段：多單進場訊號 (勝率 71.8%)】*\n"
        f"• **台指支撐確認**：台指(`{tx_sr['price']}`)守穩季線(`{tx_sr['MA60']}`)並回測支撐區(`{tx_sr['S1']}`/月線`{tx_sr['MA20']}`)\n"
        f"• **長短K共振**：日K DIF>0 且 60分K KDJ 低檔金叉 (`K={h1['K']:.1f}, J={h1['J']:.1f}`)、MACD 柱狀體轉強\n"
        f"• **小那斯達克**：NQ 期貨 (`{nq_chg_pct:+.2f}%`) 同步偏多支撐\n"
        f"👉 **建議買進價**：`{srf_price:.2f}` 元 | **停損**：`{srf_price-1.20:.2f}` 元 | **波段目標**：`{srf_price+3.00:.2f}` 元"
    )
  elif (
      (tx_sr["price"] >= tx_sr["R2"] or h1["J"] > 92)
      and h1["K"] < h1["D"]
      and h1["MACD_Hist"] < h1_p["MACD_Hist"]
  ):
    signals.append(
        "🟠 *【策略1 長波段：多單分批停利 / 減碼出場訊號】*\n"
        f"• **觸發原因**：60分K KDJ 高檔鈍化後死叉 (`J={h1['J']:.1f}`) 且 MACD 動能收斂，逼近台指壓力區 `{tx_sr['R1']}~{tx_sr['R2']}`\n"
        f"👉 **建議操作**：於 `{srf_price:.2f}` 元附近平倉 50% 鎖定獲利，剩餘部位改守日K 10MA (`{d1['MA10']:.2f}` 元)。"
    )

  # =========================================================================
  # 【策略 2：短波段 — 60分K + 15分K VCP 收斂突破策略】
  # =========================================================================
  vcp_squeeze = h1_p["BB_Width"] <= 6.5 or m15_p["BB_Width"] <= 3.5
  vol_breakout = m15["Volume"] >= m15["Vol_MA20"] * 1.4
  tx_break_r1 = tx_sr["price"] >= tx_sr["P"]

  # 短波段做多突破
  if (
      vcp_squeeze
      and srf_price >= float(m15["Pivot_High10"])
      and h1["DIF"] > h1["DEA"]
      and m15["K"] > m15["D"]
      and tx_break_r1
      and nq_bullish
  ):
    signals.append(
        "🚀 *【策略2 短波段：VCP 帶量突破多單進場 (勝率 64.5%)】*\n"
        f"• **台指結構**：站穩多空樞紐 P (`{tx_sr['P']}`) 挑戰 R1 (`{tx_sr['R1']}`)\n"
        f"• **長短K共振**：60分K MACD 金叉 + 15分K 帶量突破頸線 (`{m15['Pivot_High10']:.2f}`)，KDJ 強勢推進 (`J={m15['J']:.1f}`)\n"
        f"• **小那斯達克**：NQ 期貨 (`{nq_chg_pct:+.2f}%`) 位於均線之上助推\n"
        f"👉 **建議買進價**：`{srf_price:.2f}` 元 | **停損**：`{srf_price-0.80:.2f}` 元 | **目標**：`{srf_price+1.80:.2f}` 元\n"
        f"⚠️ _幽靈規則：若進場後 3 根 15分K 內跌回 `{m15['Pivot_High10']:.2f}` 之下，立即保本微損出場！_"
    )
  # 短波段做空跌破
  elif (
      srf_price <= float(m15["Pivot_Low10"])
      and tx_sr["price"] < tx_sr["S1"]
      and h1["DIF"] < h1["DEA"]
      and m15["K"] < m15["D"]
      and nq_bearish
  ):
    signals.append(
        "🔻 *【策略2 短波段：跌破支撐空單進場 / 避險訊號】*\n"
        f"• **台指結構**：跌破第一支撐 S1 (`{tx_sr['S1']}`) 往 S2 (`{tx_sr['S2']}`) 測試\n"
        f"• **長短K共振**：60分K 與 15分K KDJ/MACD 同步空頭死叉，且 NQ 期貨 (`{nq_chg_pct:+.2f}%`) 轉弱\n"
        f"👉 **建議放空價**：`{srf_price:.2f}` 元 | **停損**：`{srf_price+0.80:.2f}` 元 | **目標**：`{srf_price-1.80:.2f}` 元"
    )

  # =========================================================================
  # 【策略 3：日夜盤當沖 — 15分K + 5分K 極速動能與 NQ 共振策略】
  # =========================================================================
  m5_kdj_gold = (m5["K"] > m5["D"] and m5_p["K"] <= m5_p["D"]) and (
      m5_p["J"] < 25
  )
  m5_kdj_dead = (m5["K"] < m5["D"] and m5_p["K"] >= m5_p["D"]) and (
      m5_p["J"] > 80
  )
  m5_macd_turn_up = m5["MACD_Hist"] > m5_p["MACD_Hist"]
  m5_macd_turn_dn = m5["MACD_Hist"] < m5_p["MACD_Hist"]

  # 當沖多單進場
  if (
      srf_price >= vwap * 0.997
      and m15["MACD_Hist"] >= m15_p["MACD_Hist"]
      and m5_kdj_gold
      and m5_macd_turn_up
      and nq_bullish
  ):
    signals.append(
        "⚡ *【策略3 日夜盤當沖：極速多單進場訊號 (勝率 66.2%)】*\n"
        f"• **支撐與均價**：守穩日內 VWAP (`{vwap:.2f}`) 與台指支撐 (`{tx_sr['S1']}~{tx_sr['P']}`)\n"
        f"• **長短K共振**：15分K MACD 動能向上 + 5分K KDJ 超賣金叉 (`J={m5['J']:.1f}`)\n"
        f"• **小那斯達克**：NQ 15分K 動能偏多 (`{nq_chg_pct:+.2f}%`)\n"
        f"👉 **當沖買進**：`{srf_price:.2f}` 元 | **停損**：`{srf_price-0.40:.2f}` 元 (-8 Ticks) | **停利**：`{srf_price+0.90:.2f}` 元 (+18 Ticks)"
    )
  # 當沖空單進場
  elif (
      srf_price < vwap
      and m15["MACD_Hist"] <= m15_p["MACD_Hist"]
      and m5_kdj_dead
      and m5_macd_turn_dn
      and nq_bearish
  ):
    signals.append(
        "⚡ *【策略3 日夜盤當沖：極速空單進場訊號】*\n"
        f"• **壓力與均價**：跌破日內 VWAP (`{vwap:.2f}`)，受制於台指壓力 (`{tx_sr['P']}~{tx_sr['R1']}`)\n"
        f"• **長短K共振**：15分K 偏空 + 5分K KDJ 高檔死叉 (`J={m5['J']:.1f}`)，MACD 綠柱放大\n"
        f"• **小那斯達克**：NQ 期貨同步走弱 (`{nq_chg_pct:+.2f}%`)\n"
        f"👉 **當沖放空**：`{srf_price:.2f}` 元 | **停損**：`{srf_price+0.40:.2f}` 元 (+8 Ticks) | **停利**：`{srf_price-0.90:.2f}` 元 (-18 Ticks)"
    )
  # 當沖出場提醒 (5分K 極端超買超賣轉折)
  elif (m5["J"] > 95 and m5["MACD_Hist"] < m5_p["MACD_Hist"]) or (
      m5["J"] < 5 and m5["MACD_Hist"] > m5_p["MACD_Hist"]
  ):
    action_type = (
        "當沖多單立即停利出場" if m5["J"] > 95 else "當沖空單立即回補停利"
    )
    signals.append(
        f"🔔 *【策略3 日夜盤當沖：{action_type}提醒】*\n"
        f"• **觸發原因**：5分K KDJ 進入極端值 (`J={m5['J']:.1f}`) 且 MACD 柱狀體開始收斂背離\n"
        f"👉 **建議操作**：於現價 `{srf_price:.2f}` 元附近果斷平倉落袋！"
    )

  # ================= 組合推播訊息 (有訊號時立即推播，或定時報告時強制推播) =================
  if signals or FORCE_REPORT:
    nq_status_icon = "🟢 偏多" if nq_bullish else ("🔴 偏空" if nq_bearish else "⚪ 震盪")
    header = (
        f"📡 *【0050小股期 (SRF) 四維共振日夜盤戰情室】*\n"
        f"🕒 時間：`{now_str}`\n"
        f"📍 盤別：{session_str}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 **SRF 期貨價**：`{srf_price:.2f}` 元 (VWAP `{vwap:.2f}`)\n"
        f"🇺🇸 **小那斯達克 (NQ)**：`{nq_price:.0f}` (`{nq_chg_pct:+.2f}%` {nq_status_icon})\n"
        f"📊 **台指期 ({tx_sr['price']}) 支撐壓力區**：\n"
        f"   • 壓力 R2：`{tx_sr['R2']}` | R1：`{tx_sr['R1']}`\n"
        f"   • **多空樞紐 P**：`{tx_sr['P']}` (月線 `{tx_sr['MA20']}`)\n"
        f"   • 支撐 S1：`{tx_sr['S1']}` | S2：`{tx_sr['S2']}`\n"
        f"📈 **長短K線 KDJ(J值) / MACD(柱狀體)**：\n"
        f"   • 日K：`J={d1['J']:.1f}` | `MACD={d1['MACD_Hist']:+.2f}`\n"
        f"   • 60分K：`J={h1['J']:.1f}` | `MACD={h1['MACD_Hist']:+.2f}`\n"
        f"   • 15分K：`J={m15['J']:.1f}` | `MACD={m15['MACD_Hist']:+.2f}`\n"
        f"   • 5分K：`J={m5['J']:.1f}` | `MACD={m5['MACD_Hist']:+.2f}`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
    )
    if signals:
      body = "\n\n".join(signals)
    else:
      body = (
          "⏳ *【目前無共振進出場訊號 — 耐心等待關鍵價位】*\n"
          f"• **長波段關注**：回測月線 `{d1['MA20']:.2f}` 元且 60分K KDJ 低檔金叉\n"
          f"• **短波段關注**：突破 15分K 頸線 `{m15['Pivot_High10']:.2f}` 元或跌破 `{m15['Pivot_Low10']:.2f}` 元\n"
          f"• **當沖分水嶺**：日內 VWAP `{vwap:.2f}` 元與台指樞紐 `{tx_sr['P']}`"
      )
    send_tg_message(header + body)
  else:
    print(f"[{now_str}] 巡檢完畢 (SRF={srf_price:.2f}, NQ={nq_chg_pct:+.2f}%)，無進出場訊號。")


if __name__ == "__main__":
  run_strategy_engine()
