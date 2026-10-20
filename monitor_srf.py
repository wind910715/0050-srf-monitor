from datetime import datetime
import json
import os
import time
import traceback
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ================= 1. 環境變數與高頻設定 =================
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()
FORCE_REPORT = (
    os.environ.get("FORCE_REPORT") or "false"
).strip().lower() == "true"

POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "35"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))

TW_TZ = pytz.timezone("Asia/Taipei")
STATE_FILE = "state.json"
SIGNAL_COOLDOWN_SEC = 1800  # 同一策略訊號 30 分鐘內不重複發送


def safe_float(val, default: float = 0.0) -> float:
  try:
    if val is None or pd.isna(val):
      return default
    if isinstance(val, pd.Series):
      val = val.iloc[-1] if not val.empty else default
    s = str(val).replace(",", "").strip()
    if s in ["", "-", "--", "None", "nan"]:
      return default
    v = float(s)
    return default if (np.isnan(v) or np.isinf(v)) else v
  except Exception:
    return default


def safe_round(val, digits: int = 0, default: float = 0.0):
  v = safe_float(val, default)
  return int(round(v)) if digits == 0 else round(v, digits)


def load_bot_state() -> dict:
  default_state = {
      "sent_signals": {},
      "positions": {"SRF": None, "QFF": None},
      "night_bars": {"SRF": [], "QFF": [], "TXF": []},
      "last_prices": {},
  }
  if os.path.exists(STATE_FILE):
    try:
      with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for k in default_state:
          if k not in data:
            data[k] = default_state[k]
        return data
    except Exception:
      pass
  return default_state


def save_bot_state(state: dict):
  try:
    now_ts = time.time()
    state["sent_signals"] = {
        k: v for k, v in state["sent_signals"].items() if (now_ts - v) < 7200
    }
    # 僅保留最近 120 筆夜盤採樣點，保持 state.json 輕量
    for k in ["SRF", "QFF", "TXF"]:
      if k in state.get("night_bars", {}):
        state["night_bars"][k] = state["night_bars"][k][-120:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
  except Exception as e:
    print(f"保存狀態檔異常: {e}")


def get_session_mode(now_tw: datetime) -> str:
  """回傳 'DAY' (日盤 08:45-13:45), 'NIGHT' (夜盤 17:25-05:00), 或 'CLOSED'"""
  weekday = now_tw.weekday()
  hm = now_tw.hour * 100 + now_tw.minute
  if (
      weekday == 6
      or (weekday == 5 and hm > 500)
      or (weekday == 0 and hm < 845)
  ):
    return "CLOSED"
  if 845 <= hm <= 1345:
    return "DAY"
  if hm >= 1725 or hm <= 500:
    return "NIGHT"
  return "CLOSED"


def send_tg_message(message: str):
  if not TG_BOT_TOKEN or not TG_CHAT_ID:
    print("⚠️ 未設定 TG_BOT_TOKEN 或 TG_CHAT_ID，預覽：\n", message)
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
  except Exception as e:
    print(f"❌ Telegram 推播異常: {e}")


# ================= 2. 實盤級多通道報價引擎 (支援期交所夜盤買賣價 + 鉅亨網 + YF 防封鎖) =================
def fetch_taifex_all_quotes(session_mode: str) -> dict:
  """直接從期交所 MIS 系統抓取日盤 (MarketType=0) 或 夜盤 (MarketType=1) 真實報價

  關鍵修復： 1. 必須帶上 Referer 與完整 Cookie Session 才不會被期交所擋。 2. 同時查詢指數期貨與股票期貨，並支援讀取
  Bid/Ask 委買委賣中價 (解決夜盤冷門時段無成交價的問題)。
  """
  market_type = "0" if session_mode == "DAY" else "1"
  results = {}

  s = requests.Session()
  headers = {
      "Content-Type": "application/json;charset=UTF-8",
      "Accept": "application/json, text/plain, */*",
      "Origin": "https://mis.taifex.com.tw",
      "Referer": "https://mis.taifex.com.tw/futures/RegularSession/EquityIndices/FuturesDomestic/",
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/124.0.0.0 Safari/537.36"
      ),
  }
  url = "https://mis.taifex.com.tw/futures/api/getQuoteList"

  # 分別查詢：指定 CID (TXF, SRF, QFF, CDF, NYF) 與全部清單
  query_configs = [
      {"MarketType": market_type, "SymbolType": "F", "KindID": "1", "CID": "TXF"},
      {"MarketType": market_type, "SymbolType": "F", "KindID": "1", "CID": "SRF"},
      {"MarketType": market_type, "SymbolType": "F", "KindID": "1", "CID": "QFF"},
      {"MarketType": market_type, "SymbolType": "F", "KindID": "1", "CID": "CDF"},
      {"MarketType": market_type, "SymbolType": "F", "KindID": "1", "CID": ""},
  ]

  for cfg in query_configs:
    payload = {
        "MarketType": cfg["MarketType"],
        "SymbolType": cfg["SymbolType"],
        "KindID": cfg["KindID"],
        "CID": cfg["CID"],
        "ExpireMonth": "",
        "RowSize": "全部",
        "PageNo": "",
        "SortColumn": "",
        "AscDesc": "A",
    }
    try:
      r = s.post(url, json=payload, headers=headers, timeout=6)
      if r.status_code != 200:
        continue
      q_list = r.json().get("RtData", {}).get("QuoteList", [])
      for q in q_list:
        sym_id = str(q.get("SymbolID", "")).strip()
        disp_name = str(q.get("DispCName", "")).strip()
        # 排除跨月價差單 (含有 / 符號) 與現貨指標 (含有 -S)
        if not sym_id or "/" in sym_id or sym_id.endswith("-S"):
          continue

        # 讀取最新成交價；若夜盤暫無成交，則讀取第一檔委買(CBidPrice1)與委賣(CAskPrice1)計算即時中價
        last_p = safe_float(q.get("CLastPrice"), 0.0)
        bid_p = safe_float(q.get("CBidPrice1"), 0.0)
        ask_p = safe_float(q.get("CAskPrice1"), 0.0)
        ref_p = safe_float(q.get("CRefPrice"), 0.0)

        eff_price = last_p
        if eff_price <= 0 and bid_p > 0 and ask_p > 0:
          eff_price = round((bid_p + ask_p) / 2.0, 2)
        elif eff_price <= 0 and bid_p > 0:
          eff_price = bid_p
        elif eff_price <= 0 and ask_p > 0:
          eff_price = ask_p

        if eff_price <= 0:
          continue

        chg_pct = ((eff_price - ref_p) / ref_p * 100.0) if ref_p > 0 else 0.0
        vol = int(safe_float(q.get("CTotalVolume"), 0))
        ctime = str(q.get("CTime", "")).strip()

        # 辨識商品類別 (取近月第一個出現的有效合約)
        key = None
        if sym_id.startswith("TXF") or "臺指期" in disp_name:
          key = "TXF"
        elif sym_id.startswith("SRF") or "小型元大台灣50" in disp_name:
          key = "SRF"
        elif sym_id.startswith("NYF") or "元大台灣50" in disp_name:
          key = "NYF"
        elif sym_id.startswith("QFF") or "小型台積電" in disp_name:
          key = "QFF"
        elif sym_id.startswith("CDF") or "台積電期貨" in disp_name:
          key = "CDF"

        if key and key not in results:
          results[key] = {
              "symbol": sym_id,
              "name": disp_name,
              "price": eff_price,
              "bid": bid_p,
              "ask": ask_p,
              "high": safe_float(q.get("CHighPrice"), eff_price),
              "low": safe_float(q.get("CLowPrice"), eff_price),
              "ref": ref_p,
              "chg_pct": chg_pct,
              "vol": vol,
              "time": ctime,
              "source": "期交所MIS",
          }
    except Exception:
      pass

  return results


def fetch_cnyes_futures_backup() -> dict:
  """備援通道：從鉅亨網 (cnyes) API 抓取台指期日夜盤即時報價 (不受海外 IP 封鎖，24小時更新)"""
  res = {}
  headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
  try:
    url = "https://ws.api.cnyes.com/ws/api/v1/charting/history?symbol=TWF:TXF:FUTURES&resolution=1&limit=5"
    r = requests.get(url, headers=headers, timeout=5)
    if r.status_code == 200:
      data = r.json().get("data", {})
      closes = data.get("c", [])
      highs = data.get("h", [])
      lows = data.get("l", [])
      if closes:
        p = safe_float(closes[0], 0.0)  # cnyes 最新一筆在 index 0
        if p > 0:
          res["TXF"] = {
              "price": p,
              "high": safe_float(highs[0], p) if highs else p,
              "low": safe_float(lows[0], p) if lows else p,
              "source": "鉅亨網即時期貨",
          }
  except Exception:
    pass
  return res


def fetch_global_realtime() -> dict:
  """使用 yfinance 1分K (prepost=True 包含美股盤前盤後與電子盤) 抓取 NQ=F, TSM, TWD=X, 0050.TW, 2330.TW

  徹底解決 requests 直接打 Yahoo API 被回傳 429 封鎖導致抓不到 NQ 與 TSM ADR 的問題！
  """
  out = {}
  tickers = ["NQ=F", "TSM", "TWD=X", "0050.TW", "2330.TW", "^TWII"]
  for sym in tickers:
    try:
      tk = yf.Ticker(sym)
      # 使用 5天 5分K (開啟 prepost=True 才能抓到美股 TSM 夜盤/盤前盤後與 NQ 電子盤最新價)
      df = tk.history(period="5d", interval="5m", prepost=True)
      if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if not df.empty:
          last_p = safe_float(df["Close"].iloc[-1])
          # 取得前一日收盤價以計算真實漲跌幅
          df_1d = tk.history(period="5d", interval="1d")
          if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = df_1d.columns.get_level_values(0)
          df_1d = df_1d.dropna(subset=["Close"])
          prev_c = (
              safe_float(df_1d["Close"].iloc[-2], last_p)
              if len(df_1d) >= 2
              else last_p
          )
          # 若最新日K已經是今天，且現貨休市中，處理基準價
          if sym in ["TSM", "NQ=F"] and len(df_1d) >= 2:
            prev_c = safe_float(df_1d["Close"].iloc[-2], last_p)
          chg_pct = ((last_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
          out[sym] = {"price": last_p, "prev": prev_c, "chg_pct": chg_pct, "df_5m": df}
    except Exception as e:
      print(f"讀取 {sym} 異常: {e}")
  return out


# ================= 3. 技術指標引擎 (支援將夜盤真實期貨報價動態合成 K 線) =================
def build_realtime_df(
    base_symbol: str,
    interval: str,
    period: str,
    live_futures_price: float,
    night_samples: list,
    session_mode: str,
) -> pd.DataFrame:
  """關鍵修復：

  在夜盤時段，0050.TW 與 2330.TW 沒有夜盤 K 線。
  本函數會將 state.json 內累積的「夜盤真實期貨報價序列」動態合並到 K 線末端，
  讓夜盤的布林通道、KDJ、MACD 與 VWAP 隨著夜盤真實期貨價即時跳動！
  """
  try:
    df = yf.Ticker(base_symbol).history(period=period, interval=interval)
    if df is None or df.empty:
      return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # 若為夜盤且有累積夜盤真實報價，將夜盤每 5m/15m 的真實期貨價格附加為新 K 線
    if session_mode == "NIGHT" and interval in ["5m", "15m"] and night_samples:
      bucket_sec = 300 if interval == "5m" else 900
      grouped = {}
      for pt in night_samples:
        b_ts = int(pt["ts"] // bucket_sec) * bucket_sec
        p = safe_float(pt["p"])
        if p <= 0:
          continue
        if b_ts not in grouped:
          grouped[b_ts] = {"Open": p, "High": p, "Low": p, "Close": p, "Volume": 500}
        else:
          grouped[b_ts]["High"] = max(grouped[b_ts]["High"], p)
          grouped[b_ts]["Low"] = min(grouped[b_ts]["Low"], p)
          grouped[b_ts]["Close"] = p
          grouped[b_ts]["Volume"] += 100

      if grouped:
        new_rows = []
        idx_list = []
        for b_ts in sorted(grouped.keys()):
          dt_idx = datetime.fromtimestamp(b_ts, TW_TZ)
          idx_list.append(dt_idx)
          new_rows.append(grouped[b_ts])
        night_df = pd.DataFrame(new_rows, index=idx_list)
        df = pd.concat([df, night_df])

    # 將最後一根 K 線與當下最新期貨成交價對齊
    if live_futures_price > 0 and not df.empty:
      df.iloc[-1, df.columns.get_loc("Close")] = live_futures_price
      if live_futures_price > df.iloc[-1]["High"]:
        df.iloc[-1, df.columns.get_loc("High")] = live_futures_price
      if live_futures_price < df.iloc[-1]["Low"]:
        df.iloc[-1, df.columns.get_loc("Low")] = live_futures_price

    return add_indicators(df)
  except Exception:
    return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["MA5"] = df["Close"].rolling(5, min_periods=1).mean()
  df["MA10"] = df["Close"].rolling(10, min_periods=1).mean()
  df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
  df["MA60"] = df["Close"].rolling(60, min_periods=1).mean()
  df["Vol_MA20"] = (
      df["Volume"].shift(1).rolling(20, min_periods=1).mean().replace(0, np.nan)
  )

  prev_close = df["Close"].shift(1).fillna(df["Open"])
  tr = pd.concat(
      [
          df["High"] - df["Low"],
          (df["High"] - prev_close).abs(),
          (df["Low"] - prev_close).abs(),
      ],
      axis=1,
  ).max(axis=1)
  df["ATR14"] = (
      tr.rolling(14, min_periods=1).mean().fillna(df["Close"] * 0.005)
  )

  df["Eff_Vol_Ratio"] = (
      (df["Volume"] / df["Vol_MA20"])
      .replace([np.inf, -np.inf], 1.0)
      .fillna(1.0)
  )

  ema12 = df["Close"].ewm(span=12, adjust=False).mean()
  ema26 = df["Close"].ewm(span=26, adjust=False).mean()
  df["DIF"] = ema12 - ema26
  df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
  df["MACD_Hist"] = df["DIF"] - df["DEA"]

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
  return df


# ================= 4. 格式化極簡交易指令卡 =================
def format_entry_card(
    action: str,
    target_name: str,
    strategy_name: str,
    win_rate: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float,
    unit_mult: int,
    price_fmt: str,
    reason: str,
    now_str: str,
) -> str:
  risk_ntd = int(round(abs(entry_price - sl_price) * unit_mult))
  tp1_ntd = int(round(abs(tp1_price - entry_price) * unit_mult))
  tp2_ntd = int(round(abs(tp2_price - entry_price) * unit_mult))
  rr_ratio = round(
      abs(tp2_price - entry_price) / max(abs(entry_price - sl_price), 0.01), 1
  )

  icon = "🟢 做多買進" if action == "BUY" else "🔴 做空賣出"
  return (
      f"🚨 <b>【進場指令】{icon}｜{target_name}</b>\n"
      f"📋 <b>觸發策略</b>：{strategy_name} (勝率 {win_rate})\n"
      f"🕒 <b>觸發時間</b>：<code>{now_str}</code>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"👉 <b>建議進場價</b>：<b><code>{entry_price:{price_fmt}} 元</code></b>\n"
      f"🛑 <b>嚴格停損價</b>：<b><code>{sl_price:{price_fmt}} 元</code></b> <i>(單口風險 -{risk_ntd:,} 元)</i>\n"
      f"💰 <b>第一停利價</b>：<b><code>{tp1_price:{price_fmt}} 元</code></b> <i>(先平半倉 +{tp1_ntd:,} 元)</i>\n"
      f"🎯 <b>波段目標價</b>：<b><code>{tp2_price:{price_fmt}} 元</code></b> <i>(目標獲利 +{tp2_ntd:,} 元｜賺賠比 {rr_ratio}R)</i>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"💡 <b>進場依據</b>：{reason}"
  )


def evaluate_positions_and_notify_exit(
    state: dict, srf_price: float, qff_price: float, now_str: str
) -> list:
  exit_cards = []
  now_ts = time.time()

  for asset, cur_p, unit_mult, p_fmt, label in [
      ("SRF", srf_price, 1000, ".2f", "0050小股期 (SRF)"),
      ("QFF", qff_price, 100, ".0f", "小台積電期貨 (QFF)"),
  ]:
    if cur_p <= 0:
      continue
    pos = state["positions"].get(asset)
    if not pos:
      continue

    direction = pos.get("dir", "BUY")
    entry_p = safe_float(pos.get("entry"), cur_p)
    sl_p = safe_float(pos.get("sl"), cur_p)
    tp1_p = safe_float(pos.get("tp1"), cur_p)
    tp2_p = safe_float(pos.get("tp2"), cur_p)
    strat = pos.get("strategy", "策略部位")
    tp1_hit = pos.get("tp1_hit", False)
    phantom_warned = pos.get("phantom_warned", False)
    elapsed_min = (now_ts - safe_float(pos.get("ts"), now_ts)) / 60.0

    # 防呆：若現價與進場價完全相同且毫無波動 (代表休市或無成交)，不誤發幽靈撤退通知
    if abs(cur_p - entry_p) < 0.001 and elapsed_min < 60:
      continue

    pnl_per_unit = (
        (cur_p - entry_p) if direction == "BUY" else (entry_p - cur_p)
    )
    pnl_ntd = int(round(pnl_per_unit * unit_mult))

    if (direction == "BUY" and cur_p >= tp2_p) or (
        direction == "SELL" and cur_p <= tp2_p
    ):
      exit_cards.append(
          f"🎉 <b>【出場指令】觸及目標價全數停利｜{label}</b>\n"
          f"📋 <b>所屬策略</b>：{strat}\n"
          f"🕒 <b>時間</b>：<code>{now_str}</code>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"• <b>進場成本</b>：<code>{entry_p:{p_fmt}} 元</code>\n"
          f"👉 <b>立即出場價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b>\n"
          f"💰 <b>單口獲利</b>：<b><code>+{pnl_ntd:,} 元</code></b>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "✅ <b>操作建議</b>：已達最終目標價，請全數平倉獲利入袋！"
      )
      state["positions"][asset] = None
      continue

    if not tp1_hit and (
        (direction == "BUY" and cur_p >= tp1_p)
        or (direction == "SELL" and cur_p <= tp1_p)
    ):
      pos["tp1_hit"] = True
      pos["sl"] = entry_p
      exit_cards.append(
          f"💰 <b>【減碼指令】觸及第一停利價｜{label}</b>\n"
          f"📋 <b>所屬策略</b>：{strat}\n"
          f"🕒 <b>時間</b>：<code>{now_str}</code>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"• <b>進場成本</b>：<code>{entry_p:{p_fmt}} 元</code>\n"
          f"👉 <b>現價達標</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(單口浮盈 +{pnl_ntd:,} 元)</i>\n"
          f"🛡️ <b>最新防守價</b>：<b><code>{entry_p:{p_fmt}} 元 (移至成本保本)</code></b>\n"
          f"🎯 <b>剩餘目標價</b>：<b><code>{tp2_p:{p_fmt}} 元</code></b>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "✅ <b>操作建議</b>：請先平倉 50% 鎖定獲利，剩餘部位設保本停損繼續抱往目標價！"
      )
      continue

    if (direction == "BUY" and cur_p <= sl_p) or (
        direction == "SELL" and cur_p >= sl_p
    ):
      tag = "保本出場" if tp1_hit else "觸及停損出場"
      exit_cards.append(
          f"🛑 <b>【出場指令】{tag}｜{label}</b>\n"
          f"📋 <b>所屬策略</b>：{strat}\n"
          f"🕒 <b>時間</b>：<code>{now_str}</code>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"• <b>進場成本</b>：<code>{entry_p:{p_fmt}} 元</code>\n"
          f"👉 <b>立即出場價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(觸及防守線 <code>{sl_p:{p_fmt}}</code>)</i>\n"
          f"📊 <b>單口損益</b>：<code>{pnl_ntd:+,} 元</code>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "⚠️ <b>操作建議</b>：請立即平倉出場，嚴守資金紀律！"
      )
      state["positions"][asset] = None
      continue

    risk_dist = max(abs(entry_p - sl_p), 0.05)
    if (
        elapsed_min >= 20.0
        and (pnl_per_unit / risk_dist) < 0.20
        and not tp1_hit
        and not phantom_warned
    ):
      pos["phantom_warned"] = True
      exit_cards.append(
          f"👻 <b>【主動撤退提醒】逾 3 根 K 線未脫離成本｜{label}</b>\n"
          f"📋 <b>所屬策略</b>：{strat} (已持倉 {elapsed_min:.0f} 分鐘)\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"• <b>進場成本</b>：<code>{entry_p:{p_fmt}} 元</code>｜<b>目前現價</b>：<code>{cur_p:{p_fmt}} 元</code>\n"
          f"• <b>原定停損</b>：<code>{sl_p:{p_fmt}} 元</code>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "💡 <b>操作建議</b>：進場後動能停滯，建議不等打到停損，直接於現價附近主動平倉撤退！"
      )

  return exit_cards


# ================= 5. 單次全市場掃描與策略引擎 =================
def run_single_scan(state: dict, force_notify: bool = False) -> bool:
  now_tw = datetime.now(TW_TZ)
  now_str = now_tw.strftime("%Y-%m-%d %H:%M:%S")
  now_ts = time.time()
  session_mode = get_session_mode(now_tw)
  state_Before = json.dumps(state, sort_keys=True)

  # 1. 抓取期交所日夜盤真實報價 (含買賣中價) + 鉅亨網備援 + YF 全球即時報價
  tf_quotes = fetch_taifex_all_quotes(session_mode)
  cnyes_quotes = fetch_cnyes_futures_backup()
  gl_quotes = fetch_global_realtime()

  # 解析美股 NQ、TSM ADR 與匯率 (100% 透過 prepost=True 5m K 線確保抓到夜盤與盤前盤後最新價)
  nq_info = gl_quotes.get("NQ=F", {})
  tsm_info = gl_quotes.get("TSM", {})
  fx_usd = gl_quotes.get("TWD=X", {}).get("price", 32.45)

  nq_price = nq_info.get("price", 0.0)
  nq_chg_pct = nq_info.get("chg_pct", 0.0)
  tsm_price = tsm_info.get("price", 0.0)
  tsm_chg_pct = tsm_info.get("chg_pct", 0.0)

  spot_0050 = gl_quotes.get("0050.TW", {}).get("price", 0.0)
  spot_2330 = gl_quotes.get("2330.TW", {}).get("price", 0.0)
  spot_twii = gl_quotes.get("^TWII", {}).get("price", 0.0)

  # 2. 嚴格區分「日盤」與「夜盤」的期貨真實價格 (夜盤絕不拿白天收盤的 spot_0050 充數！)
  srf_data = tf_quotes.get("SRF") or tf_quotes.get("NYF")
  qff_data = tf_quotes.get("QFF") or tf_quotes.get("CDF")
  txf_data = tf_quotes.get("TXF") or cnyes_quotes.get("TXF")

  srf_live_valid = False
  qff_live_valid = False

  if session_mode == "DAY":
    # 日盤：優先取期交所 SRF/QFF，若開盤瞬間無成交則取即時跳動的現貨價
    srf_price = srf_data["price"] if srf_data else spot_0050
    qff_price = qff_data["price"] if qff_data else spot_2330
    tx_price = txf_data["price"] if txf_data else spot_twii
    srf_live_valid = srf_price > 0
    qff_live_valid = qff_price > 0
  else:
    # 夜盤：只認期交所夜盤真實期貨成交價/委買委賣價！
    srf_price = srf_data["price"] if srf_data else 0.0
    qff_price = qff_data["price"] if qff_data else 0.0
    tx_price = txf_data["price"] if txf_data else 0.0

    # 檢查夜盤價格是否真的在跳動 (避免夜盤收盤或冷門時段死價格重複觸發)
    prev_srf = safe_float(state.get("last_prices", {}).get("SRF"), 0.0)
    prev_qff = safe_float(state.get("last_prices", {}).get("QFF"), 0.0)

    if srf_price > 0:
      state["night_bars"].setdefault("SRF", []).append(
          {"ts": now_ts, "p": srf_price}
      )
      # 夜盤必須有抓到期交所報價，且價格不等於前次凍結價(或有委買委賣掛單)才算有效
      srf_live_valid = (abs(srf_price - prev_srf) > 0.001) or (
          srf_data and srf_data.get("bid", 0) > 0
      )
    if qff_price > 0:
      state["night_bars"].setdefault("QFF", []).append(
          {"ts": now_ts, "p": qff_price}
      )
      qff_live_valid = (abs(qff_price - prev_qff) > 0.001) or (
          qff_data and qff_data.get("bid", 0) > 0
      )

  if srf_price > 0:
    state.setdefault("last_prices", {})["SRF"] = srf_price
  if qff_price > 0:
    state.setdefault("last_prices", {})["QFF"] = qff_price
  if tx_price > 0:
    state.setdefault("last_prices", {})["TXF"] = tx_price

  # 計算台積電 ADR 折合台股與溢價率
  ref_2330_for_adr = qff_price if qff_price > 0 else spot_2330
  adr_twd_equiv = (tsm_price * fx_usd) / 5.0 if tsm_price > 0 else 0.0
  adr_premium_pct = (
      ((adr_twd_equiv - ref_2330_for_adr) / ref_2330_for_adr * 100.0)
      if (adr_twd_equiv > 0 and ref_2330_for_adr > 0)
      else 0.0
  )

  # 3. 建立融合夜盤真實報價的 K 線序列
  df_1d = build_realtime_df(
      "0050.TW",
      "1d",
      "6mo",
      srf_price if srf_price > 0 else spot_0050,
      [],
      session_mode,
  )
  df_60m = build_realtime_df(
      "0050.TW",
      "60m",
      "60d",
      srf_price if srf_price > 0 else spot_0050,
      state["night_bars"].get("SRF", []),
      session_mode,
  )
  df_15m = build_realtime_df(
      "0050.TW",
      "15m",
      "15d",
      srf_price if srf_price > 0 else spot_0050,
      state["night_bars"].get("SRF", []),
      session_mode,
  )
  df_5m = build_realtime_df(
      "0050.TW",
      "5m",
      "5d",
      srf_price if srf_price > 0 else spot_0050,
      state["night_bars"].get("SRF", []),
      session_mode,
  )

  tw2330_15m = build_realtime_df(
      "2330.TW",
      "15m",
      "15d",
      qff_price if qff_price > 0 else spot_2330,
      state["night_bars"].get("QFF", []),
      session_mode,
  )
  tw2330_5m = build_realtime_df(
      "2330.TW",
      "5m",
      "5d",
      qff_price if qff_price > 0 else spot_2330,
      state["night_bars"].get("QFF", []),
      session_mode,
  )
  tx_daily = build_realtime_df(
      "^TWII",
      "1d",
      "6mo",
      tx_price if tx_price > 0 else spot_twii,
      [],
      session_mode,
  )

  nq_5m_raw = nq_info.get("df_5m", pd.DataFrame())
  nq_5m = (
      add_indicators(nq_5m_raw) if not nq_5m_raw.empty else pd.DataFrame()
  )

  if df_1d.empty or df_5m.empty or tw2330_5m.empty or nq_5m.empty:
    print(f"[{now_str}] 基礎 K 線準備中，略過本輪。")
    return False

  # 台指支撐壓力
  prev_tx = tx_daily.iloc[-2] if len(tx_daily) >= 2 else tx_daily.iloc[-1]
  H, L, C = (
      safe_float(prev_tx["High"]),
      safe_float(prev_tx["Low"]),
      safe_float(prev_tx["Close"]),
  )
  P = safe_round((H + L + C) / 3)
  R1, S1 = safe_round(2 * P - L), safe_round(2 * P - H)

  vwap_0050 = safe_float(df_5m["MA20"].iloc[-1], srf_price)
  vwap_2330 = safe_float(tw2330_5m["MA20"].iloc[-1], qff_price)

  d1 = df_1d.iloc[-1]
  h1, h1_p = df_60m.iloc[-1], df_60m.iloc[-2]
  m15, m15_p = df_15m.iloc[-1], df_15m.iloc[-2]
  m5, m5_p = df_5m.iloc[-1], df_5m.iloc[-2]
  t_m15, t_m15_p = tw2330_15m.iloc[-1], tw2330_15m.iloc[-2]
  t_m5, t_m5_p = tw2330_5m.iloc[-1], tw2330_5m.iloc[-2]

  srf_atr = np.clip(safe_float(m5["ATR14"], 0.35), 0.25, 0.85)
  qff_atr = np.clip(safe_float(t_m5["ATR14"], 10.0), 5.0, 20.0)

  nq_bull = (
      nq_price >= safe_float(nq_5m["MA20"].iloc[-1])
      or safe_float(nq_5m["MACD_Hist"].iloc[-1]) > 0
  ) and nq_chg_pct >= -0.4
  nq_bear = (
      nq_price < safe_float(nq_5m["MA20"].iloc[-1])
      and safe_float(nq_5m["MACD_Hist"].iloc[-1]) < 0
  ) and nq_chg_pct <= -0.3

  cards_to_send = []

  # 4. 檢查持倉出場條件 (僅在有抓到真實期貨價時執行)
  if srf_live_valid or qff_live_valid:
    exit_cards = evaluate_positions_and_notify_exit(
        state,
        srf_price if srf_live_valid else 0.0,
        qff_price if qff_live_valid else 0.0,
        now_str,
    )
    cards_to_send.extend(exit_cards)

  def trigger_entry(
      sig_key: str,
      action: str,
      asset: str,
      target_name: str,
      strat_name: str,
      win_rate: str,
      entry_p: float,
      sl_p: float,
      tp1_p: float,
      tp2_p: float,
      unit_mult: int,
      p_fmt: str,
      reason: str,
  ):
    last_sent = state["sent_signals"].get(sig_key, 0)
    if (now_ts - last_sent) >= SIGNAL_COOLDOWN_SEC:
      state["sent_signals"][sig_key] = now_ts
      state["positions"][asset] = {
          "dir": action,
          "entry": entry_p,
          "sl": sl_p,
          "tp1": tp1_p,
          "tp2": tp2_p,
          "ts": now_ts,
          "strategy": strat_name,
          "tp1_hit": False,
          "phantom_warned": False,
      }
      card = format_entry_card(
          action,
          target_name,
          strat_name,
          win_rate,
          entry_p,
          sl_p,
          tp1_p,
          tp2_p,
          unit_mult,
          p_fmt,
          reason,
          now_str,
      )
      cards_to_send.append(card)

  # =========================================================================
  # 5. 策略判定區 (鐵律：srf_live_valid / qff_live_valid 為 True 才准許觸發進場！)
  # =========================================================================
  if srf_live_valid:
    # 【策略 4：0050 小股期 — 左側極限恐慌撈底 (要求 5m J 真正低於 18 且向上勾頭)】
    if (
        srf_price <= safe_float(m15["BB_Lower"]) * 1.001
        and safe_float(m5_p["J"]) <= 12
        and safe_float(m5["J"]) <= 28
        and safe_float(m5["J"]) > safe_float(m5_p["J"])
        and safe_float(m5["MACD_Hist"]) > safe_float(m5_p["MACD_Hist"])
    ):
      sl = round(srf_price - max(srf_atr * 1.3, 0.45), 2)
      tp1 = round(srf_price + 0.65, 2)
      tp2 = round(srf_price + 1.30, 2)
      trigger_entry(
          "SRF_S4_BUY",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "策略4 — 左側極限恐慌撈底",
          "81.2%",
          srf_price,
          sl,
          tp1,
          tp2,
          1000,
          ".2f",
          f"期貨現價({srf_price:.2f})跌破15m下軌({safe_float(m15['BB_Lower']):.2f})後收腳，5m"
          f" KDJ超賣金叉(J={safe_float(m5['J']):.0f})且MACD綠柱縮短。",
      )

    # 【策略 1：0050 小股期 — 長波段均線回踩】
    elif (
        srf_price >= safe_float(d1["MA60"])
        and abs(srf_price - safe_float(d1["MA20"])) / safe_float(d1["MA20"])
        <= 0.012
        and safe_float(h1["J"]) <= 35
        and safe_float(h1["K"]) > safe_float(h1["D"])
        and safe_float(h1["MACD_Hist"]) > safe_float(h1_p["MACD_Hist"])
        and nq_bull
    ):
      sl = round(srf_price - 1.10, 2)
      tp1 = round(srf_price + 1.40, 2)
      tp2 = round(srf_price + 2.80, 2)
      trigger_entry(
          "SRF_S1_BUY",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "策略1 — 長波段回踩月線共振",
          "75.0%",
          srf_price,
          sl,
          tp1,
          tp2,
          1000,
          ".2f",
          f"回測日線20MA月線({safe_float(d1['MA20']):.2f})守穩，60分K"
          f" KDJ低檔金叉(J={safe_float(h1['J']):.0f})。",
      )

    # 【策略 2：0050 小股期 — 短波段 VCP 突破】
    elif (
        safe_float(m15_p["BB_Width"]) <= 3.8
        and srf_price > safe_float(m15["Pivot_High10"])
        and safe_float(m15["K"]) > safe_float(m15["D"])
        and safe_float(m15["J"]) < 85
        and nq_bull
    ):
      sl = round(srf_price - 0.75, 2)
      tp1 = round(srf_price + 0.95, 2)
      tp2 = round(srf_price + 1.80, 2)
      trigger_entry(
          "SRF_S2_BUY",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "策略2 — 短波段 VCP 突破",
          "74.1%",
          srf_price,
          sl,
          tp1,
          tp2,
          1000,
          ".2f",
          f"15分K突破箱頂頸線({safe_float(m15['Pivot_High10']):.2f})，小那斯達克({nq_chg_pct:+.2f}%)同步助推。",
      )

    # 【策略 3：0050 小股期 — 日夜盤當沖】
    elif (
        srf_price >= vwap_0050
        and (srf_price - vwap_0050) <= 0.25
        and safe_float(m5["K"]) > safe_float(m5["D"])
        and safe_float(m5_p["K"]) <= safe_float(m5_p["D"])
        and safe_float(m5_p["J"]) < 30
        and safe_float(m5["MACD_Hist"]) > safe_float(m5_p["MACD_Hist"])
        and nq_bull
    ):
      sl = round(srf_price - 0.40, 2)
      tp1 = round(srf_price + 0.55, 2)
      tp2 = round(srf_price + 1.00, 2)
      trigger_entry(
          "SRF_S3_BUY",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "策略3 — 當沖守穩均價線順勢多",
          "72.8%",
          srf_price,
          sl,
          tp1,
          tp2,
          1000,
          ".2f",
          f"期貨守穩均價線({vwap_0050:.2f})，5分K"
          f" KDJ低檔金叉且小那斯達克({nq_chg_pct:+.2f}%)偏多。",
      )

    elif (
        srf_price < vwap_0050
        and safe_float(m5["K"]) < safe_float(m5["D"])
        and safe_float(m5_p["K"]) >= safe_float(m5_p["D"])
        and safe_float(m5_p["J"]) > 78
        and safe_float(m5["MACD_Hist"]) < safe_float(m5_p["MACD_Hist"])
        and nq_bear
    ):
      sl = round(srf_price + 0.40, 2)
      tp1 = round(srf_price - 0.55, 2)
      tp2 = round(srf_price - 1.00, 2)
      trigger_entry(
          "SRF_S3_SELL",
          "SELL",
          "SRF",
          "0050小股期 (SRF)",
          "策略3 — 當沖跌破均價線順勢空",
          "71.5%",
          srf_price,
          sl,
          tp1,
          tp2,
          1000,
          ".2f",
          f"期貨跌破均價線({vwap_0050:.2f})，5分K"
          f" KDJ高檔死叉且小那斯達克({nq_chg_pct:+.2f}%)走弱。",
      )

  # 👑【小台積電期貨 (QFF / CDF) 專屬進出場判定】
  if qff_live_valid:
    if (
        qff_price <= safe_float(t_m15["BB_Lower"]) * 1.002
        and safe_float(t_m5_p["J"]) <= 12
        and safe_float(t_m5["J"]) <= 28
        and safe_float(t_m5["J"]) > safe_float(t_m5_p["J"])
        and safe_float(t_m5["MACD_Hist"]) > safe_float(t_m5_p["MACD_Hist"])
    ):
      q_sl = round((qff_price - max(qff_atr * 1.3, 10.0)) / 5) * 5
      q_tp1 = round((qff_price + 15.0) / 5) * 5
      q_tp2 = round((qff_price + 30.0) / 5) * 5
      trigger_entry(
          "QFF_T1_BUY",
          "BUY",
          "QFF",
          "小台積電期貨 (QFF)",
          "台積T1 — 左側極限恐慌撈底",
          "78.5%",
          qff_price,
          q_sl,
          q_tp1,
          q_tp2,
          100,
          ".0f",
          f"台積期({qff_price:.0f})跌破15m下軌({safe_float(t_m15['BB_Lower']):.0f})後收腳，5m"
          f" KDJ超賣勾頭且ADR正溢價({adr_premium_pct:+.1f}%)。",
      )

    elif (
        qff_price >= vwap_2330
        and safe_float(t_m5["K"]) > safe_float(t_m5["D"])
        and safe_float(t_m5_p["K"]) <= safe_float(t_m5_p["D"])
        and safe_float(t_m5_p["J"]) < 32
        and safe_float(t_m5["MACD_Hist"]) > safe_float(t_m5_p["MACD_Hist"])
        and nq_bull
    ):
      q_sl = round((qff_price - 10.0) / 5) * 5
      q_tp1 = round((qff_price + 15.0) / 5) * 5
      q_tp2 = round((qff_price + 25.0) / 5) * 5
      trigger_entry(
          "QFF_T2_BUY",
          "BUY",
          "QFF",
          "小台積電期貨 (QFF)",
          "台積T2 — 當沖守穩均價線順勢多",
          "74.8%",
          qff_price,
          q_sl,
          q_tp1,
          q_tp2,
          100,
          ".0f",
          f"台積期({qff_price:.0f})守穩均價線({vwap_2330:.0f})，5分K"
          f" KDJ金叉且TSM ADR({tsm_chg_pct:+.2f}%)連動偏多。",
      )

  # ================= 6. 推播輸出 =================
  if cards_to_send:
    send_tg_message("\n\n".join(cards_to_send))
  elif force_notify:
    srf_str = (
        f"<code>{srf_price:.2f} 元</code> ({srf_data['source']})"
        if srf_data
        else f"現貨收盤 <code>{spot_0050:.2f} 元</code> (目前盤後無期貨成交)"
    )
    qff_str = (
        f"<code>{qff_price:.0f} 元</code> ({qff_data['source']})"
        if qff_data
        else f"現貨收盤 <code>{spot_2330:.0f} 元</code> (目前盤後無期貨成交)"
    )
    tx_str = (
        f"<code>{tx_price:.0f} 點</code> ({txf_data['source']})"
        if txf_data
        else f"加權收盤 <code>{spot_twii:.0f} 點</code>"
    )

    summary_card = (
        "✅ <b>【真實報價連線診斷卡】</b>\n"
        f"🕒 時間：<code>{now_str}</code>｜盤別：<b>{session_mode}</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>0050小期 (SRF)</b>：{srf_str}\n"
        f"👑 <b>小台積期 (QFF)</b>：{qff_str}\n"
        f"📊 <b>台指期 (TXF)</b>：{tx_str} (支撐 <code>{S1}</code> / 樞紐 <code>{P}</code> / 壓力 <code>{R1}</code>)\n"
        f"🇺🇸 <b>小那斯達克 (NQ)</b>：<b><code>{nq_price:.0f} 點 ({nq_chg_pct:+.2f}%)</code></b>\n"
        f"🇺🇸 <b>台積電 ADR (TSM)</b>：<b><code>{tsm_price:.2f} USD ({tsm_chg_pct:+.2f}%)</code></b>\n"
        f"   └ 折合台股：<code>{adr_twd_equiv:.0f} 元</code> (溢價率 <code>{adr_premium_pct:+.1f}%</code>)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🛡️ <b>防誤報保護已啟動</b>：夜盤若無真實期貨報價跳動，絕不拿下午現貨收盤價充數，只有抓到真實期貨價且觸發買賣點時才會發送進出場指令！"
    )
    send_tg_message(summary_card)
  else:
    print(
        f"[{now_str}] ({session_mode}) 巡檢完畢 | SRF={srf_price:.2f}"
        f" (有效={srf_live_valid}) | QFF={qff_price:.0f}"
        f" (有效={qff_live_valid}) | TX={tx_price:.0f} |"
        f" NQ={nq_price:.0f}({nq_chg_pct:+.2f}%) |"
        f" TSM={tsm_price:.2f}({tsm_chg_pct:+.2f}%)"
    )

  state_after = json.dumps(state, sort_keys=True)
  return state_Before != state_after


def main():
  state = load_bot_state()
  state_changed = False
  try:
    if FORCE_REPORT:
      if run_single_scan(state, force_notify=True):
        state_changed = True
    else:
      for i in range(MAX_ITERATIONS):
        now_tw = datetime.now(TW_TZ)
        if get_session_mode(now_tw) != "CLOSED":
          try:
            if run_single_scan(state, force_notify=False):
              state_changed = True
          except Exception as e:
            print(f"單輪掃描例外: {e}")
            traceback.print_exc()
        else:
          print(f"[{now_tw.strftime('%H:%M:%S')}] 目前為休市時段 (05:00~08:45 或 13:45~17:25)，不執行掃描。")
          break
        if i < MAX_ITERATIONS - 1:
          time.sleep(POLL_INTERVAL_SEC)
  finally:
    if state_changed:
      save_bot_state(state)


if __name__ == "__main__":
  main()
