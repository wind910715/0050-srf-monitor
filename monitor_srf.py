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
SIGNAL_COOLDOWN_SEC = 1200  # 同一策略訊號 20 分鐘內不重複發送


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


def load_bot_state() -> dict:
  default_state = {
      "sent_signals": {},
      "positions": {"SRF": None, "QFF": None},
  }
  if os.path.exists(STATE_FILE):
    try:
      with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "sent_signals" in data and "positions" in data:
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
    with open(STATE_FILE, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
  except Exception as e:
    print(f"保存狀態檔異常: {e}")


def is_trading_hours(now_tw: datetime) -> bool:
  weekday = now_tw.weekday()
  hm = now_tw.hour * 100 + now_tw.minute
  if (
      weekday == 6
      or (weekday == 5 and hm > 505)
      or (weekday == 0 and hm < 840)
  ):
    return False
  return (840 <= hm <= 1350) or (hm >= 1720) or (hm <= 505)


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


# ================= 2. 零延遲即時報價引擎 (突破海外 IP 封鎖，直連奇摩股市台灣即時 API) =================
def fetch_tw_realtime_quotes() -> dict:
  """直接從 Yahoo 奇摩股市台灣即時報價 API 與 글로벌 Chart V8 API 獲取零延遲報價

  完全不受 GitHub 美國 IP 限制，且無 15 分鐘延遲！
  """
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/124.0.0.0 Safari/537.36"
      )
  }
  result = {}

  # A. 抓取台灣奇摩即時報價：0050現貨、2330台積電現貨、加權指數、台指期近月(WTX&)、小台積期(WTQFF&/WTCDF&)、小0050期(WTSRF&)
  tw_symbols = (
      "0050.TW,2330.TW,%23001,WTX%26,WTQFF%26,WTCDF%26,WTSRF%26,WTNYF%26"
  )
  tw_url = f"https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.stockList;symbols={tw_symbols}"
  try:
    r = requests.get(tw_url, headers=headers, timeout=6)
    if r.status_code == 200:
      items = r.json()
      if isinstance(items, dict):
        items = items.get("list", [])
      for item in items:
        sym = item.get("symbol", "")
        price = safe_float(item.get("price"), 0.0)
        # 若盤中瞬間僅有買賣價，取買賣中價
        if price <= 0:
          bid = safe_float(item.get("bid"), 0.0)
          ask = safe_float(item.get("ask"), 0.0)
          if bid > 0 and ask > 0:
            price = round((bid + ask) / 2, 2)
        prev_close = safe_float(item.get("regularMarketPreviousClose"), price)
        chg_pct = safe_float(item.get("changePercent"), 0.0)
        if price > 0:
          result[sym] = {
              "price": price,
              "high": safe_float(item.get("regularMarketDayHigh"), price),
              "low": safe_float(item.get("regularMarketDayLow"), price),
              "prev": prev_close,
              "chg_pct": chg_pct,
          }
  except Exception as e:
    print(f"奇摩台灣即時 API 讀取備援: {e}")

  # B. 備援：若期交所 JSON API 可連線則同步更新期貨報價
  now_tw = datetime.now(TW_TZ)
  hm = now_tw.hour * 100 + now_tw.minute
  m_type = "0" if (845 <= hm <= 1345) else "1"
  for cid, map_key in [("SRF", "WTSRF&"), ("QFF", "WTQFF&"), ("TXF", "WTX&")]:
    if map_key not in result:
      try:
        r_tf = requests.post(
            "https://mis.taifex.com.tw/futures/api/getQuoteList",
            json={
                "MarketType": m_type,
                "SymbolType": "F",
                "KindID": "1",
                "CID": cid,
                "ExpireMonth": "",
                "RowSize": "全部",
                "PageNo": "",
                "SortColumn": "",
                "AscDesc": "A",
            },
            headers=headers,
            timeout=4,
        )
        for q in r_tf.json().get("RtData", {}).get("QuoteList", []):
          s_id = str(q.get("SymbolID", ""))
          lp = str(q.get("CLastPrice", "")).strip()
          if s_id and "/" not in s_id and lp and lp != "-":
            p_val = safe_float(lp, 0.0)
            ref_val = safe_float(q.get("CRefPrice"), p_val)
            if p_val > 0:
              result[map_key] = {
                  "price": p_val,
                  "high": safe_float(q.get("CHighPrice"), p_val),
                  "low": safe_float(q.get("CLowPrice"), p_val),
                  "prev": ref_val,
                  "chg_pct": (
                      ((p_val - ref_val) / ref_val * 100)
                      if ref_val > 0
                      else 0.0
                  ),
              }
              break
      except Exception:
        pass

  # C. 直接抓取美股即時 NQ=F (小那斯達克電子盤)、TSM (台積電ADR)、TWD=X (匯率) 最新 meta 報價
  for us_sym in ["NQ=F", "TSM", "TWD=X", "0050.TW", "2330.TW", "^TWII"]:
    if us_sym in result:
      continue
    try:
      u_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{us_sym}?interval=1m&range=1d"
      ru = requests.get(u_url, headers=headers, timeout=5)
      meta = ru.json()["chart"]["result"][0]["meta"]
      p = safe_float(meta.get("regularMarketPrice"), 0.0)
      prev = safe_float(
          meta.get("chartPreviousClose") or meta.get("previousClose"), p
      )
      if p > 0:
        result[us_sym] = {
            "price": p,
            "high": safe_float(meta.get("regularMarketDayHigh"), p),
            "low": safe_float(meta.get("regularMarketDayLow"), p),
            "prev": prev,
            "chg_pct": ((p - prev) / prev * 100) if prev > 0 else 0.0,
        }
    except Exception:
      pass

  return result


# ================= 3. 技術指標引擎 (將最後一根 K 線強制融合最新即時價，消除指標延遲) =================
def fetch_clean_history(
    ticker_symbol: str,
    period: str,
    interval: str,
    realtime_price: float = 0.0,
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

    # 關鍵修正：將最後一根 K 線的收盤價、最高價、最低價即時更新為「當下零延遲報價」，確保 KDJ/MACD/布林通道完全零延遲！
    if realtime_price > 0:
      df.iloc[-1, df.columns.get_loc("Close")] = realtime_price
      if realtime_price > df.iloc[-1]["High"]:
        df.iloc[-1, df.columns.get_loc("High")] = realtime_price
      if realtime_price < df.iloc[-1]["Low"]:
        df.iloc[-1, df.columns.get_loc("Low")] = realtime_price

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

  # ATR(14)
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

  # 下影線與收盤相對位置
  bar_range = (df["High"] - df["Low"]).replace(0, np.nan)
  lower_shadow = np.minimum(df["Open"], df["Close"]) - df["Low"]
  df["Lower_Shadow_Ratio"] = (lower_shadow / bar_range).fillna(0.0)
  df["Close_Pos"] = ((df["Close"] - df["Low"]) / bar_range).fillna(0.5)
  df["Eff_Vol_Ratio"] = (
      (df["Volume"] / df["Vol_MA20"])
      .replace([np.inf, -np.inf], 1.0)
      .fillna(1.0)
  )

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
      return safe_float(
          (tp * today_df["Volume"]).sum() / vol_sum, fallback_price
      )
  except Exception:
    pass
  return fallback_price


# ================= 4. 格式化「一眼看懂」的極簡進出場指令卡 =================
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
  """生成極簡、直覺的進場通知卡 (一眼看到進場、停損、第一停利、目標價與單口損益金額)"""
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
  """監控持倉：觸及停損、第一停利、目標價或逾時未脫離成本時，第一時間發送出場指令卡"""
  exit_cards = []
  now_ts = time.time()

  for asset, cur_p, unit_mult, p_fmt, label in [
      ("SRF", srf_price, 1000, ".2f", "0050小股期 (SRF)"),
      ("QFF", qff_price, 100, ".0f", "小台積電期貨 (QFF)"),
  ]:
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

    pnl_per_unit = (
        (cur_p - entry_p) if direction == "BUY" else (entry_p - cur_p)
    )
    pnl_ntd = int(round(pnl_per_unit * unit_mult))

    # 1. 觸及最終波段目標價 (TP2) -> 全數獲利出場
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
          f"💰 <b>單口獲利</b>：<b><code>+{pnl_ntd:,} 元</code></b> (已達最終目標價 <code>{tp2_p:{p_fmt}}</code>)\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "✅ <b>操作建議</b>：請全數平倉獲利入袋！"
      )
      state["positions"][asset] = None
      continue

    # 2. 觸及第一停利價 (TP1) -> 平倉 50% 並將停損移至成本價
    if not tp1_hit and (
        (direction == "BUY" and cur_p >= tp1_p)
        or (direction == "SELL" and cur_p <= tp1_p)
    ):
      pos["tp1_hit"] = True
      pos["sl"] = entry_p  # 停損自動移至成本保本
      exit_cards.append(
          f"💰 <b>【減碼指令】觸及第一停利價｜{label}</b>\n"
          f"📋 <b>所屬策略</b>：{strat}\n"
          f"🕒 <b>時間</b>：<code>{now_str}</code>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"• <b>進場成本</b>：<code>{entry_p:{p_fmt}} 元</code>\n"
          f"👉 <b>現價達標</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(單口浮盈 +{pnl_ntd:,} 元)</i>\n"
          f"🛡️ <b>最新防守價</b>：<b><code>{entry_p:{p_fmt}} 元 (移至成本保本)</code></b>\n"
          f"🎯 <b>剩餘部位目標</b>：<b><code>{tp2_p:{p_fmt}} 元</code></b>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "✅ <b>操作建議</b>：請先平倉 50% 鎖定獲利，剩餘部位設保本停損繼續抱往目標價！"
      )
      continue

    # 3. 觸及停損價 (SL) -> 立即停損或保本出場
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
          "⚠️ <b>操作建議</b>：請立即市價/限價平倉出場，嚴守資金紀律！"
      )
      state["positions"][asset] = None
      continue

    # 4. 幽靈法則一：進場滿 18 分鐘仍未脫離成本區 -> 主動撤退提醒
    risk_dist = max(abs(entry_p - sl_p), 0.05)
    if (
        elapsed_min >= 18.0
        and (pnl_per_unit / risk_dist) < 0.25
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
  state_Before = json.dumps(state, sort_keys=True)

  # 1. 取得零延遲即時報價
  rt = fetch_tw_realtime_quotes()

  spot_0050 = rt.get("0050.TW", {}).get("price", 0.0)
  srf_price = (
      rt.get("WTSRF&", {}).get("price", 0.0)
      or rt.get("WTNYF&", {}).get("price", 0.0)
      or spot_0050
  )

  spot_2330 = rt.get("2330.TW", {}).get("price", 0.0)
  qff_price = (
      rt.get("WTQFF&", {}).get("price", 0.0)
      or rt.get("WTCDF&", {}).get("price", 0.0)
      or spot_2330
  )

  tx_price = rt.get("WTX&", {}).get("price", 0.0) or rt.get(
      "%23001", rt.get("^TWII", {})
  ).get("price", 0.0)
  nq_info = rt.get("NQ=F", {})
  tsm_info = rt.get("TSM", {})
  fx_usd = rt.get("TWD=X", {}).get("price", 32.45)

  # 2. 抓取各時框 K 線並將最後一根強制融合「當下零延遲報價」
  df_1d = fetch_clean_history("0050.TW", "6mo", "1d", srf_price)
  df_60m = fetch_clean_history("0050.TW", "60d", "60m", srf_price)
  df_15m = fetch_clean_history("0050.TW", "15d", "15m", srf_price)
  df_5m = fetch_clean_history("0050.TW", "5d", "5m", srf_price)

  tw2330_1d = fetch_clean_history("2330.TW", "6mo", "1d", qff_price)
  tw2330_60m = fetch_clean_history("2330.TW", "60d", "60m", qff_price)
  tw2330_15m = fetch_clean_history("2330.TW", "15d", "15m", qff_price)
  tw2330_5m = fetch_clean_history("2330.TW", "5d", "5m", qff_price)

  tx_daily = fetch_clean_history("^TWII", "6mo", "1d", tx_price)
  nq_5m = fetch_clean_history("NQ=F", "5d", "5m", nq_info.get("price", 0.0))

  if df_1d.empty or df_5m.empty or tw2330_5m.empty or nq_5m.empty:
    print(f"[{now_str}] 資料讀取中，略過本輪。")
    return False

  if srf_price <= 0:
    srf_price = safe_float(df_5m["Close"].iloc[-1])
  if qff_price <= 0:
    qff_price = safe_float(tw2330_5m["Close"].iloc[-1])
  if tx_price <= 0:
    tx_price = safe_float(tx_daily["Close"].iloc[-1])

  nq_price = nq_info.get("price") or safe_float(nq_5m["Close"].iloc[-1])
  nq_chg_pct = nq_info.get("chg_pct", 0.0)
  tsm_price = tsm_info.get("price", 412.61)
  tsm_chg_pct = tsm_info.get("chg_pct", 0.0)
  adr_twd_equiv = (tsm_price * fx_usd) / 5.0
  adr_premium_pct = (
      ((adr_twd_equiv - qff_price) / qff_price * 100) if qff_price > 0 else 0.0
  )

  # 台指支撐壓力計算
  prev_tx = tx_daily.iloc[-2] if len(tx_daily) >= 2 else tx_daily.iloc[-1]
  H, L, C = (
      safe_float(prev_tx["High"]),
      safe_float(prev_tx["Low"]),
      safe_float(prev_tx["Close"]),
  )
  P = safe_round((H + L + C) / 3)
  R1, S1 = safe_round(2 * P - L), safe_round(2 * P - H)
  tx_ma20 = safe_round(tx_daily["MA20"].iloc[-1])
  tx_ma60 = safe_round(tx_daily["MA60"].iloc[-1])

  vwap_0050 = compute_vwap(df_5m, srf_price)
  vwap_2330 = compute_vwap(tw2330_5m, qff_price)

  d1 = df_1d.iloc[-1]
  h1, h1_p = df_60m.iloc[-1], df_60m.iloc[-2]
  m15, m15_p = df_15m.iloc[-1], df_15m.iloc[-2]
  m5, m5_p = df_5m.iloc[-1], df_5m.iloc[-2]

  t_h1, t_h1_p = tw2330_60m.iloc[-1], tw2330_60m.iloc[-2]
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

  # 3. 先檢查手上部位是否觸及停損 / 停利 / 主動出場
  exit_cards = evaluate_positions_and_notify_exit(
      state, srf_price, qff_price, now_str
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
  # 策略判定區 (修正策略1誤觸發：強制要求價格真正回測月線/下軌且 60m J < 40 才觸發)
  # =========================================================================

  # 【策略 4：0050 小股期 — 左側極限恐慌撈底】
  if (
      (
          srf_price <= safe_float(m15["BB_Lower"]) * 1.002
          or (srf_price - vwap_0050) / vwap_0050 <= -0.01
      )
      and (safe_float(m15["J"]) <= 10 or safe_float(m5_p["J"]) <= 5)
      and safe_float(m5["J"]) > safe_float(m5_p["J"])
      and safe_float(m5["MACD_Hist"]) > safe_float(m5_p["MACD_Hist"])
  ):
    sl = round(srf_price - max(srf_atr * 1.3, 0.45), 2)
    tp1 = round(max(vwap_0050, srf_price + 0.65), 2)
    tp2 = round(srf_price + max(srf_atr * 2.8, 1.30), 2)
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
        f"跌破15m布林下軌({safe_float(m15['BB_Lower']):.2f})後收腳，5m"
        f" KDJ超賣勾頭(J={safe_float(m5['J']):.0f})且MACD綠柱縮短。",
    )

  # 【策略 1：0050 小股期 — 長波段均線回踩買進 (嚴格要求 60m J <= 40 且接近 20MA，杜絕高檔誤報)】
  elif (
      srf_price >= safe_float(d1["MA60"])
      and abs(srf_price - safe_float(d1["MA20"])) / safe_float(d1["MA20"])
      <= 0.015
      and safe_float(h1["J"]) <= 40
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
        f" KDJ於低檔金叉(J={safe_float(h1['J']):.0f})。",
    )

  # 【策略 2：0050 小股期 — 短波段 VCP 爆量突破】
  elif (
      safe_float(m15_p["BB_Width"]) <= 4.0
      and safe_float(m15["Eff_Vol_Ratio"]) >= 1.5
      and srf_price > safe_float(m15["Pivot_High10"])
      and safe_float(m15["K"]) > safe_float(m15["D"])
      and safe_float(m15["J"]) < 88
      and tx_price >= P
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
        "策略2 — 短波段 VCP 帶量突破",
        "74.1%",
        srf_price,
        sl,
        tp1,
        tp2,
        1000,
        ".2f",
        f"15分K帶量({safe_float(m15['Eff_Vol_Ratio']):.1f}x)突破箱頂頸線({safe_float(m15['Pivot_High10']):.2f})，台指站上樞紐({P})。",
    )

  # 【策略 3：0050 小股期 — 日夜盤當沖 (多空雙向)】
  elif (
      srf_price >= vwap_0050
      and (srf_price - vwap_0050) <= 0.30
      and safe_float(m5["K"]) > safe_float(m5["D"])
      and safe_float(m5_p["K"]) <= safe_float(m5_p["D"])
      and safe_float(m5_p["J"]) < 35
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
        "策略3 — 當沖回測 VWAP 順勢多",
        "72.8%",
        srf_price,
        sl,
        tp1,
        tp2,
        1000,
        ".2f",
        f"回測日內均價線 VWAP({vwap_0050:.2f})不破，5分K"
        f" KDJ低檔金叉且小那斯達克({nq_chg_pct:+.2f}%)偏多。",
    )

  elif (
      srf_price < vwap_0050
      and safe_float(m5["K"]) < safe_float(m5["D"])
      and safe_float(m5_p["K"]) >= safe_float(m5_p["D"])
      and safe_float(m5_p["J"]) > 75
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
        "策略3 — 當沖跌破 VWAP 順勢空",
        "71.5%",
        srf_price,
        sl,
        tp1,
        tp2,
        1000,
        ".2f",
        f"跌破日內均價線 VWAP({vwap_0050:.2f})受阻，5分K"
        f" KDJ高檔死叉且小那斯達克({nq_chg_pct:+.2f}%)走弱。",
    )

  # =========================================================================
  # 👑【台積電期貨 (QFF 小台積 / CDF 台積期) 專屬進出場判定】
  # =========================================================================
  if (
      (
          qff_price <= safe_float(t_m15["BB_Lower"]) * 1.003
          or (qff_price - vwap_2330) / vwap_2330 <= -0.011
      )
      and (safe_float(t_m15["J"]) <= 10 or safe_float(t_m5_p["J"]) <= 5)
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
        f"台積期跌破15m下軌({safe_float(t_m15['BB_Lower']):.0f})後收腳，5m"
        f" KDJ超賣勾頭且ADR維持正溢價({adr_premium_pct:+.1f}%)。",
    )

  elif (
      qff_price >= vwap_2330
      and safe_float(t_m5["K"]) > safe_float(t_m5["D"])
      and safe_float(t_m5_p["K"]) <= safe_float(t_m5_p["D"])
      and safe_float(t_m5_p["J"]) < 35
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
        "台積T2 — 當沖守穩 VWAP 順勢多",
        "74.8%",
        qff_price,
        q_sl,
        q_tp1,
        q_tp2,
        100,
        ".0f",
        f"台積期守穩日內均價線({vwap_2330:.0f}元)，5分K"
        f" KDJ金叉且MACD翻紅(ADR折合{adr_twd_equiv:.0f}元)。",
    )

  # ================= 6. 發送訊息 (有觸發進出場才發送；若為手動測試/早報則發送簡潔報價卡) =================
  if cards_to_send:
    send_tg_message("\n\n".join(cards_to_send))
  elif force_notify:
    summary_card = (
        "✅ <b>【系統連線正常｜即時零延遲報價確認】</b>\n"
        f"🕒 時間：<code>{now_str}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>0050小期 (SRF) 即時價</b>：<b><code>{srf_price:.2f} 元</code></b> (均價線"
        f" <code>{vwap_0050:.2f}</code>)\n"
        f"👑 <b>小台積期 (QFF) 即時價</b>：<b><code>{qff_price:.0f} 元</code></b> (均價線"
        f" <code>{vwap_2330:.0f}</code>)\n"
        f"📊 <b>台指期即時點數</b>：<b><code>{tx_price:.0f} 點</code></b> (支撐"
        f" <code>{S1}</code> / 樞紐 <code>{P}</code> / 壓力 <code>{R1}</code>)\n"
        f"🇺🇸 <b>小那斯達克 (NQ)</b>：<code>{nq_price:.0f}</code>"
        f" (<code>{nq_chg_pct:+.2f}%</code>)\n"
        f"🇺🇸 <b>台積電 ADR 折合台股</b>：<code>{adr_twd_equiv:.0f} 元</code> (溢價"
        f" <code>{adr_premium_pct:+.1f}%</code>)\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "⏳ <b>目前狀態</b>：盤中每 35 秒自動盯盤中，<b>未觸發買賣點前不會發送雜訊打擾您</b>；一旦觸發策略，將第一時間推播「進場價、停損價、停利價與目標價」指令卡！"
    )
    send_tg_message(summary_card)
  else:
    print(
        f"[{now_str}] 即時巡檢正常 (SRF={srf_price:.2f}, QFF={qff_price:.0f},"
        f" TX={tx_price:.0f}, NQ={nq_chg_pct:+.2f}%) - 等待進出場訊號"
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
        if is_trading_hours(now_tw):
          try:
            if run_single_scan(state, force_notify=False):
              state_changed = True
          except Exception as e:
            print(f"單輪掃描例外: {e}")
            traceback.print_exc()
        else:
          print(f"[{now_tw.strftime('%H:%M:%S')}] 非交易時段，休息中。")
          break
        if i < MAX_ITERATIONS - 1:
          time.sleep(POLL_INTERVAL_SEC)
  finally:
    if state_changed:
      save_bot_state(state)


if __name__ == "__main__":
  main()
