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

# ================= 1. 環境變數與每 30 秒極速迴圈設定 =================
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()
FORCE_REPORT = (
    os.environ.get("FORCE_REPORT") or "false"
).strip().lower() == "true"

# 每 30 秒運算一次，單次啟動連續執行 9 次 (涵蓋 4分30秒)，無縫銜接每 5 分鐘排程
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "30"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "9"))

TW_TZ = pytz.timezone("Asia/Taipei")
STATE_FILE = "state.json"
SIGNAL_COOLDOWN_SEC = 1800  # 同一策略訊號 30 分鐘內不重複洗版


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
      "intraday_bars": {"SRF": [], "QFF": [], "TXF": []},
      "last_valid_quotes": {},
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
    for k in ["SRF", "QFF", "TXF"]:
      if k in state.get("intraday_bars", {}):
        state["intraday_bars"][k] = state["intraday_bars"][k][-180:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
  except Exception as e:
    print(f"保存狀態檔異常: {e}")


# ================= 2. 嚴格時段控制 (含提早 30 分鐘開機等待模式) =================
def get_session_status(now_tw: datetime) -> dict:
  """早盤交易：08:45 - 13:45 (提早 30 分鐘於 08:15 開機等待)

  夜盤交易：15:00 - 05:00 (提早 30 分鐘於 14:30 開機等待)
  """
  weekday = now_tw.weekday()  # 0=週一 ... 4=週五, 5=週六, 6=週日
  hm = now_tw.hour * 100 + now_tw.minute

  # 週末休市判斷 (週六 05:02 後至週一 08:15 前休市)
  if (
      weekday == 6
      or (weekday == 5 and hm > 502)
      or (weekday == 0 and hm < 815)
  ):
    return {
        "mode": "CLOSED",
        "label": "🛑 休市中",
        "can_run": False,
        "can_signal": False,
    }

  # 早盤提早 30 分鐘開機等待 (08:15 ~ 08:44)
  if 815 <= hm < 845:
    return {
        "mode": "STANDBY_DAY",
        "label": "⏳ 早盤開機待命中 (08:45 開盤)",
        "can_run": True,
        "can_signal": False,
    }

  # 早盤正式交易 (08:45 ~ 13:45)
  if 845 <= hm <= 1345:
    return {
        "mode": "DAY",
        "label": "☀️ 早盤實時監控中 (08:45-13:45)",
        "can_run": True,
        "can_signal": True,
    }

  # 夜盤提早 30 分鐘開機等待 (14:30 ~ 14:59)
  if 1430 <= hm < 1500:
    return {
        "mode": "STANDBY_NIGHT",
        "label": "⏳ 夜盤開機待命中 (15:00 開盤)",
        "can_run": True,
        "can_signal": False,
    }

  # 夜盤正式交易 (15:00 ~ 次日 05:00)
  if hm >= 1500 or hm <= 500:
    return {
        "mode": "NIGHT",
        "label": "🌙 夜盤實時監控中 (15:00-05:00)",
        "can_run": True,
        "can_signal": True,
    }

  return {
      "mode": "CLOSED",
      "label": "🛑 盤間休息 (13:46-14:29 / 05:01-08:14)",
      "can_run": False,
      "can_signal": False,
  }


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


# ================= 3. 核心除錯：期交所「股票期貨(S) + 指數期貨(F)」雙盤全掃描引擎 =================
def fetch_taifex_realtime_quotes(session_mode: str) -> dict:
  """關鍵除錯：

  1. 小0050(SRF) 與 小台積電(QFF/CDF) 在期交所 API 的 SymbolType 是 "S" (股票期貨)，也有部分接口歸類於 "F"！
  2. 同時掃描日盤 (MarketType="0") 與夜盤 (MarketType="1")： - 在日盤 (DAY) 優先取日盤成交價，若剛開盤無成交則取夜盤最新收盤價作為基準。 - 在夜盤 (NIGHT) 或早盤待命 (STANDBY_DAY, 如 08:15) 優先取夜盤 (MarketType="1") 的真實收盤/成交價 (如 SRF 108.65、QFF 2426)！
  3. 同一商品若有多個月份合約 (如結算日同時有 10月與11月)，自動挑選「有最新成交價且成交量最大」的主力合約。
  """
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

  # 決定查詢盤別優先順序：早盤正式交易優先查 "0" (日盤)，其餘時段 (夜盤、早盤08:15待命) 優先查 "1" (夜盤)
  m_types = ["0", "1"] if session_mode == "DAY" else ["1", "0"]

  # 涵蓋股票期貨 (SymbolType="S") 與指數期貨 (SymbolType="F") 及指定 CID
  query_list = []
  for mt in m_types:
    query_list.extend([
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": ""},  # 抓全部股票期貨 (含 SRF, QFF, CDF, NYF)
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": ""},  # 抓全部指數期貨 (含 TXF, MXF, TMF)
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": "SRF"},
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": "QFF"},
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": "CDF"},
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": "TXF"},
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": "SRF"},
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": "QFF"},
    ])

  candidates = {"SRF": [], "QFF": [], "CDF": [], "NYF": [], "TXF": []}

  for cfg in query_list:
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
      r = s.post(url, json=payload, headers=headers, timeout=5)
      if r.status_code != 200:
        continue
      q_list = r.json().get("RtData", {}).get("QuoteList", [])
      for q in q_list:
        sym_id = str(q.get("SymbolID", "")).strip()
        disp_name = str(q.get("DispCName", "")).strip()
        # 排除跨月價差單 (含 /) 與現貨標的 (含 -S)
        if not sym_id or "/" in sym_id or sym_id.endswith("-S"):
          continue

        last_p = safe_float(q.get("CLastPrice"), 0.0)
        bid_p = safe_float(q.get("CBidPrice1"), 0.0)
        ask_p = safe_float(q.get("CAskPrice1"), 0.0)
        ref_p = safe_float(q.get("CRefPrice"), 0.0)
        vol = int(safe_float(q.get("CTotalVolume"), 0))

        # 優先取最新成交價；若無成交但有委買委賣掛單，取買賣中價
        eff_price = last_p
        if eff_price <= 0 and bid_p > 0 and ask_p > 0:
          eff_price = round((bid_p + ask_p) / 2.0, 2)
        if eff_price <= 0:
          continue

        key = None
        if sym_id.startswith("SRF") or "小型元大台灣50" in disp_name:
          key = "SRF"
        elif sym_id.startswith("QFF") or "小型台積電" in disp_name:
          key = "QFF"
        elif sym_id.startswith("CDF") or (
            "台積電期貨" in disp_name and "小型" not in disp_name
        ):
          key = "CDF"
        elif sym_id.startswith("NYF") or (
            "元大台灣50" in disp_name and "小型" not in disp_name
        ):
          key = "NYF"
        elif sym_id.startswith("TXF") or "臺指期" in disp_name:
          key = "TXF"

        if key:
          chg_pct = ((eff_price - ref_p) / ref_p * 100.0) if ref_p > 0 else 0.0
          m_label = "日盤" if cfg["MarketType"] == "0" else "夜盤"
          # 給予優先級權重：符合當下優先盤別 + 有成交量者優先
          priority = (
              1000000 if cfg["MarketType"] == m_types[0] else 0
          ) + (100000 if last_p > 0 else 0) + vol
          candidates[key].append({
              "symbol": sym_id,
              "name": disp_name,
              "price": eff_price,
              "last": last_p,
              "bid": bid_p,
              "ask": ask_p,
              "high": safe_float(q.get("CHighPrice"), eff_price),
              "low": safe_float(q.get("CLowPrice"), eff_price),
              "ref": ref_p,
              "chg_pct": chg_pct,
              "vol": vol,
              "time": str(q.get("CTime", "")).strip(),
              "market_type": cfg["MarketType"],
              "source": f"期交所MIS({m_label})",
              "priority": priority,
          })
    except Exception:
      pass

  # 備援：同步從 Yahoo 奇摩台灣期貨即時 API 抓取 (WTSRF&, WTQFF&, WTCDF&, WTX&)
  try:
    y_syms = "WTSRF%26,WTQFF%26,WTCDF%26,WTNYF%26,WTX%26"
    y_url = f"https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.stockList;symbols={y_syms}"
    ry = requests.get(
        y_url, headers={"User-Agent": headers["User-Agent"]}, timeout=5
    )
    if ry.status_code == 200:
      items = ry.json()
      if isinstance(items, dict):
        items = items.get("list", [])
      map_sym = {
          "WTSRF&": "SRF",
          "WTQFF&": "QFF",
          "WTCDF&": "CDF",
          "WTNYF&": "NYF",
          "WTX&": "TXF",
      }
      for it in items:
        k = map_sym.get(it.get("symbol", ""))
        p = safe_float(it.get("price"), 0.0)
        if k and p > 0:
          ref = safe_float(it.get("regularMarketPreviousClose"), p)
          candidates[k].append({
              "symbol": it.get("symbol"),
              "name": it.get("symbolName", k),
              "price": p,
              "last": p,
              "bid": safe_float(it.get("bid"), p),
              "ask": safe_float(it.get("ask"), p),
              "high": safe_float(it.get("regularMarketDayHigh"), p),
              "low": safe_float(it.get("regularMarketDayLow"), p),
              "ref": ref,
              "chg_pct": safe_float(it.get("changePercent"), 0.0),
              "vol": int(safe_float(it.get("regularMarketVolume"), 100)),
              "time": "即時",
              "market_type": m_types[0],
              "source": "奇摩即時期貨",
              "priority": 500000,
          })
  except Exception:
    pass

  results = {}
  for k, cand_list in candidates.items():
    if cand_list:
      cand_list.sort(key=lambda x: x["priority"], reverse=True)
      results[k] = cand_list[0]

  return results


def fetch_global_realtime() -> dict:
  """抓取美股 NQ=F, TSM (台積電ADR), TWD=X, 0050.TW, 2330.TW"""
  out = {}
  for sym in ["NQ=F", "TSM", "TWD=X", "0050.TW", "2330.TW", "^TWII"]:
    try:
      tk = yf.Ticker(sym)
      df = tk.history(period="5d", interval="5m", prepost=True)
      if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
          df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if not df.empty:
          last_p = safe_float(df["Close"].iloc[-1])
          df_1d = tk.history(period="5d", interval="1d")
          if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = df_1d.columns.get_level_values(0)
          df_1d = df_1d.dropna(subset=["Close"])
          prev_c = (
              safe_float(df_1d["Close"].iloc[-2], last_p)
              if len(df_1d) >= 2
              else last_p
          )
          chg_pct = ((last_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
          out[sym] = {
              "price": last_p,
              "prev": prev_c,
              "chg_pct": chg_pct,
              "df_5m": df,
          }
    except Exception:
      pass
  return out


# ================= 4. 動態合成日夜盤實時 K 線與指標引擎 =================
def build_realtime_df(
    base_symbol: str,
    interval: str,
    period: str,
    live_futures_price: float,
    intraday_samples: list,
) -> pd.DataFrame:
  try:
    df = yf.Ticker(base_symbol).history(period=period, interval=interval)
    if df is None or df.empty:
      return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # 將實時採樣的期貨報價合並至 5m / 15m K 線末端，確保日夜盤指標完全跟隨期貨實時跳動
    if interval in ["5m", "15m"] and intraday_samples:
      bucket_sec = 300 if interval == "5m" else 900
      last_df_ts = df.index[-1].timestamp() if not df.empty else 0
      grouped = {}
      for pt in intraday_samples:
        if pt["ts"] <= last_df_ts:
          continue
        b_ts = int(pt["ts"] // bucket_sec) * bucket_sec
        p = safe_float(pt["p"])
        if p <= 0:
          continue
        if b_ts not in grouped:
          grouped[b_ts] = {
              "Open": p,
              "High": p,
              "Low": p,
              "Close": p,
              "Volume": 500,
          }
        else:
          grouped[b_ts]["High"] = max(grouped[b_ts]["High"], p)
          grouped[b_ts]["Low"] = min(grouped[b_ts]["Low"], p)
          grouped[b_ts]["Close"] = p
          grouped[b_ts]["Volume"] += 150

      if grouped:
        idx_list = [
            datetime.fromtimestamp(b_ts, TW_TZ)
            for b_ts in sorted(grouped.keys())
        ]
        new_rows = [grouped[b_ts] for b_ts in sorted(grouped.keys())]
        df = pd.concat([df, pd.DataFrame(new_rows, index=idx_list)])

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


# ================= 5. 極簡進出場指令卡 =================
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


# ================= 6. 單次全市場掃描與策略引擎 =================
def run_single_scan(state: dict, force_notify: bool = False) -> bool:
  now_tw = datetime.now(TW_TZ)
  now_str = now_tw.strftime("%Y-%m-%d %H:%M:%S")
  now_ts = time.time()
  sess = get_session_status(now_tw)
  session_mode = sess["mode"]
  state_Before = json.dumps(state, sort_keys=True)

  # 1. 抓取期交所「股票期貨(S) + 指數期貨(F)」日夜盤全掃描報價
  tf_quotes = fetch_taifex_realtime_quotes(session_mode)
  gl_quotes = fetch_global_realtime()

  nq_info = gl_quotes.get("NQ=F", {})
  tsm_info = gl_quotes.get("TSM", {})
  fx_usd = gl_quotes.get("TWD=X", {}).get("price", 32.45)

  nq_price = nq_info.get("price", 0.0)
  nq_chg_pct = nq_info.get("chg_pct", 0.0)
  tsm_price = tsm_info.get("price", 0.0)
  tsm_chg_pct = tsm_info.get("chg_pct", 0.0)

  # 2. 取得小0050 (SRF)、小台積電期貨 (QFF/CDF)、台指期 (TXF) 實時價格
  srf_data = tf_quotes.get("SRF") or tf_quotes.get("NYF")
  qff_data = tf_quotes.get("QFF") or tf_quotes.get("CDF")
  txf_data = tf_quotes.get("TXF")

  # 若成功從期交所/奇摩期貨抓到報價，寫入 last_valid_quotes 永久記憶 (供盤前待命時段顯示正確夜盤收盤價)
  if srf_data and srf_data["price"] > 0:
    state.setdefault("last_valid_quotes", {})["SRF"] = srf_data
  else:
    srf_data = state.get("last_valid_quotes", {}).get("SRF")

  if qff_data and qff_data["price"] > 0:
    state.setdefault("last_valid_quotes", {})["QFF"] = qff_data
  else:
    qff_data = state.get("last_valid_quotes", {}).get("QFF")

  if txf_data and txf_data["price"] > 0:
    state.setdefault("last_valid_quotes", {})["TXF"] = txf_data
  else:
    txf_data = state.get("last_valid_quotes", {}).get("TXF")

  srf_price = srf_data["price"] if srf_data else 0.0
  qff_price = qff_data["price"] if qff_data else 0.0
  tx_price = (
      txf_data["price"]
      if txf_data
      else gl_quotes.get("^TWII", {}).get("price", 0.0)
  )

  # 只有在正式交易時段 (DAY 或 NIGHT) 且有抓到真實期貨報價時，才記錄 K 線並允許觸發買賣訊號
  srf_can_trade = (
      sess["can_signal"] and (srf_data is not None) and (srf_price > 0)
  )
  qff_can_trade = (
      sess["can_signal"] and (qff_data is not None) and (qff_price > 0)
  )

  if srf_can_trade:
    state["intraday_bars"].setdefault("SRF", []).append(
        {"ts": now_ts, "p": srf_price}
    )
  if qff_can_trade:
    state["intraday_bars"].setdefault("QFF", []).append(
        {"ts": now_ts, "p": qff_price}
    )

  # 計算台積電 ADR 折合台股與溢價率
  adr_twd_equiv = (tsm_price * fx_usd) / 5.0 if tsm_price > 0 else 0.0
  adr_premium_pct = (
      ((adr_twd_equiv - qff_price) / qff_price * 100.0)
      if (adr_twd_equiv > 0 and qff_price > 0)
      else 0.0
  )

  # 3. 建立融合實時期貨報價的 K 線序列
  df_1d = build_realtime_df("0050.TW", "1d", "6mo", srf_price, [])
  df_60m = build_realtime_df(
      "0050.TW", "60m", "60d", srf_price, state["intraday_bars"].get("SRF", [])
  )
  df_15m = build_realtime_df(
      "0050.TW", "15m", "15d", srf_price, state["intraday_bars"].get("SRF", [])
  )
  df_5m = build_realtime_df(
      "0050.TW", "5m", "5d", srf_price, state["intraday_bars"].get("SRF", [])
  )

  tw2330_15m = build_realtime_df(
      "2330.TW", "15m", "15d", qff_price, state["intraday_bars"].get("QFF", [])
  )
  tw2330_5m = build_realtime_df(
      "2330.TW", "5m", "5d", qff_price, state["intraday_bars"].get("QFF", [])
  )
  tx_daily = build_realtime_df("^TWII", "1d", "6mo", tx_price, [])

  nq_5m_raw = nq_info.get("df_5m", pd.DataFrame())
  nq_5m = (
      add_indicators(nq_5m_raw) if not nq_5m_raw.empty else pd.DataFrame()
  )

  if df_1d.empty or df_5m.empty or tw2330_5m.empty or nq_5m.empty:
    print(f"[{now_str}] K 線初始化中，略過本輪。")
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

  # 4. 檢查持倉出場條件
  if srf_can_trade or qff_can_trade:
    exit_cards = evaluate_positions_and_notify_exit(
        state,
        srf_price if srf_can_trade else 0.0,
        qff_price if qff_can_trade else 0.0,
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
  # 5. 實時策略判定區 (僅在早盤 08:45-13:45 與夜盤 15:00-05:00 正式交易時段觸發)
  # =========================================================================
  if srf_can_trade:
    # 【策略 4：0050 小股期 — 左側極限恐慌撈底】
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
  if qff_can_trade:
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

  # ================= 7. 推播輸出 =================
  if cards_to_send:
    send_tg_message("\n\n".join(cards_to_send))
  elif force_notify:
    srf_str = (
        f"<b><code>{srf_price:.2f} 元</code></b> (參考價"
        f" <code>{srf_data.get('ref', 0):.2f}</code>｜{srf_data['source']})"
        if srf_data
        else "暫無期貨報價"
    )
    qff_str = (
        f"<b><code>{qff_price:.0f} 元</code></b> (參考價"
        f" <code>{qff_data.get('ref', 0):.0f}</code>｜{qff_data['source']})"
        if qff_data
        else "暫無期貨報價"
    )
    tx_str = (
        f"<b><code>{tx_price:.0f} 點</code></b> ({txf_data['source']})"
        if txf_data
        else f"<code>{tx_price:.0f} 點</code>"
    )

    summary_card = (
        "✅ <b>【實時報價連線診斷卡 (已修復股票期貨夜盤與日盤通道)】</b>\n"
        f"🕒 時間：<code>{now_str}</code>\n"
        f"🧭 狀態：<b>{sess['label']}</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>0050小期 (SRF)</b>：{srf_str}\n"
        f"👑 <b>小台積期 (QFF)</b>：{qff_str}\n"
        f"📊 <b>台指期 (TXF)</b>：{tx_str}\n"
        f"   └ 支撐 <code>{S1}</code> / 樞紐 <code>{P}</code> / 壓力 <code>{R1}</code>\n"
        f"🇺🇸 <b>小那斯達克 (NQ)</b>：<b><code>{nq_price:.0f} 點 ({nq_chg_pct:+.2f}%)</code></b>\n"
        f"🇺🇸 <b>台積電 ADR (TSM)</b>：<b><code>{tsm_price:.2f} USD ({tsm_chg_pct:+.2f}%)</code></b>\n"
        f"   └ 折合台股：<code>{adr_twd_equiv:.0f} 元</code> (溢價率 <code>{adr_premium_pct:+.1f}%</code>)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "⚡ <b>監控時段設定</b>：\n"
        "• <b>早盤</b>：08:15 提早開機待命 ➔ <b>08:45 ~ 13:45 實時訊號推播</b>\n"
        "• <b>夜盤</b>：14:30 提早開機待命 ➔ <b>15:00 ~ 05:00 實時訊號推播</b>"
    )
    send_tg_message(summary_card)
  else:
    print(
        f"[{now_str}] {sess['label']} | SRF={srf_price:.2f} |"
        f" QFF={qff_price:.0f} | TX={tx_price:.0f} |"
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
        sess = get_session_status(now_tw)
        if sess["can_run"]:
          try:
            if run_single_scan(state, force_notify=False):
              state_changed = True
          except Exception as e:
            print(f"單輪掃描例外: {e}")
            traceback.print_exc()
        else:
          print(f"[{now_tw.strftime('%H:%M:%S')}] {sess['label']}，不執行掃描。")
          break
        if i < MAX_ITERATIONS - 1:
          time.sleep(POLL_INTERVAL_SEC)
  finally:
    if state_changed:
      save_bot_state(state)


if __name__ == "__main__":
  main()
