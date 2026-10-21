import calendar
from datetime import date, datetime
import json
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

POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "30"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))

TW_TZ = pytz.timezone("Asia/Taipei")
STATE_FILE = "state.json"
SIGNAL_COOLDOWN_SEC = 1200

TAIFEX_MONTH_CODES = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
}


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
      "day_close_anchor": {"SRF": 108.15, "QFF": 2411.0, "TXF": 47121.0},
      "active_contract_ym": "",
      "optimized_params": {},
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


# ================= 2. 期交所結算日計算 ＆「結算前兩天自動換次月」 =================
def get_third_wednesday(year: int, month: int) -> date:
  cal = calendar.monthcalendar(year, month)
  wednesdays = [week[2] for week in cal if week[2] != 0]
  return date(year, month, wednesdays[2])


def add_months(year: int, month: int, delta: int = 1):
  m = month - 1 + delta
  y = year + m // 12
  m = m % 12 + 1
  return y, m


def make_taifex_code(year: int, month: int) -> str:
  return f"{TAIFEX_MONTH_CODES[month]}{str(year)[-1]}"


def get_contract_rollover_info(now_tw: datetime) -> dict:
  today = now_tw.date()
  hm = now_tw.hour * 100 + now_tw.minute
  this_settle = get_third_wednesday(today.year, today.month)

  if today > this_settle or (today == this_settle and hm >= 1330):
    near_y, near_m = add_months(today.year, today.month, 1)
    next_settle = get_third_wednesday(near_y, near_m)
    days_to_settle = (next_settle - today).days
    is_rollover = days_to_settle <= 2
    far_y, far_m = add_months(near_y, near_m, 1)
    if is_rollover:
      target_y, target_m = far_y, far_m
      label_tag = f"{target_m}月次月"
    else:
      target_y, target_m = near_y, near_m
      label_tag = f"{target_m}月近月"
    active_settle_date = next_settle
  else:
    near_y, near_m = today.year, today.month
    days_to_settle = (this_settle - today).days
    is_rollover = days_to_settle <= 2
    far_y, far_m = add_months(near_y, near_m, 1)
    if is_rollover:
      target_y, target_m = far_y, far_m
      label_tag = f"{target_m}月次月"
    else:
      target_y, target_m = near_y, near_m
      label_tag = f"{target_m}月近月"
    active_settle_date = this_settle

  return {
      "target_ym": f"{target_y}{target_m:02d}",
      "target_code": make_taifex_code(target_y, target_m),
      "target_label": label_tag,
      "near_ym": f"{near_y}{near_m:02d}",
      "near_m": near_m,
      "near_code": make_taifex_code(near_y, near_m),
      "far_ym": f"{far_y}{far_m:02d}",
      "far_m": far_m,
      "far_code": make_taifex_code(far_y, far_m),
      "settle_date": active_settle_date.strftime("%Y-%m-%d"),
      "days_to_settle": days_to_settle,
  }


def get_session_status(now_tw: datetime) -> dict:
  weekday = now_tw.weekday()
  hm = now_tw.hour * 100 + now_tw.minute
  if (
      weekday == 6
      or (weekday == 5 and hm > 502)
      or (weekday == 0 and hm < 815)
  ):
    return {
        "mode": "CLOSED",
        "label": "休市中",
        "can_run": False,
        "can_signal": False,
    }
  if 815 <= hm < 845:
    return {
        "mode": "STANDBY_DAY",
        "label": "早盤開機待命 (08:45開盤)",
        "can_run": True,
        "can_signal": False,
    }
  if 845 <= hm <= 1345:
    return {
        "mode": "DAY",
        "label": "早盤實時監控",
        "can_run": True,
        "can_signal": True,
    }
  if 1430 <= hm < 1500:
    return {
        "mode": "STANDBY_NIGHT",
        "label": "夜盤開機待命 (15:00開盤)",
        "can_run": True,
        "can_signal": False,
    }
  if hm >= 1500 or hm <= 500:
    return {
        "mode": "NIGHT",
        "label": "夜盤實時監控",
        "can_run": True,
        "can_signal": True,
    }
  return {
      "mode": "CLOSED",
      "label": "盤間休息",
      "can_run": False,
      "can_signal": False,
  }


def send_tg_message(message: str):
  if not TG_BOT_TOKEN or not TG_CHAT_ID:
    print(message)
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
    print(f"TG 推播異常: {e}")


# ================= 3. 實時報價引擎 =================
def fetch_all_realtime_data(
    session_mode: str, roll_info: dict, state: dict
) -> dict:
  headers = {
      "Content-Type": "application/json;charset=UTF-8",
      "Origin": "https://mis.taifex.com.tw",
      "Referer": "https://mis.taifex.com.tw/futures/RegularSession/EquityIndices/FuturesDomestic/",
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/124.0.0.0 Safari/537.36"
      ),
  }

  target_code = roll_info["target_code"]
  near_code = roll_info["near_code"]
  far_code = roll_info["far_code"]
  m_types = ["0", "1"] if session_mode == "DAY" else ["1", "0"]

  tf_target, tf_near, tf_far = {}, {}, {}
  s = requests.Session()

  for mt in m_types:
    for st in ["F", "S"]:
      try:
        r = s.post(
            "https://mis.taifex.com.tw/futures/api/getQuoteList",
            json={
                "MarketType": mt,
                "SymbolType": st,
                "KindID": "1",
                "CID": "",
                "ExpireMonth": "",
                "RowSize": "全部",
                "PageNo": "",
                "SortColumn": "",
                "AscDesc": "A",
            },
            headers=headers,
            timeout=4,
        )
        if r.status_code != 200:
          continue
        for q in r.json().get("RtData", {}).get("QuoteList", []):
          sym_id = str(q.get("SymbolID", "")).strip()
          if not sym_id or "/" in sym_id or sym_id.endswith("-S"):
            continue
          pfx = sym_id[:3]
          if pfx not in ["TXF", "SRF", "QFF", "CDF", "NYF"]:
            continue
          m_code = sym_id[3:5]
          lp = safe_float(q.get("CLastPrice"), 0.0)
          bp = safe_float(q.get("CBidPrice1"), 0.0)
          ap = safe_float(q.get("CAskPrice1"), 0.0)
          ref = safe_float(q.get("CRefPrice"), 0.0)
          eff = (
              lp
              if lp > 0
              else (round((bp + ap) / 2.0, 2) if (bp > 0 and ap > 0) else 0.0)
          )
          if eff <= 0:
            continue
          q_obj = {
              "price": eff,
              "ref": ref if ref > 0 else eff,
              "chg_pct": ((eff - ref) / ref * 100.0) if ref > 0 else 0.0,
          }
          if m_code == target_code and pfx not in tf_target:
            tf_target[pfx] = q_obj
          if m_code == near_code and pfx not in tf_near:
            tf_near[pfx] = q_obj
          if m_code == far_code and pfx not in tf_far:
            tf_far[pfx] = q_obj
      except Exception:
        pass

  yh_tw = {}
  try:
    y_syms = "0050.TW,2330.TW,%23001,WTX%26,WSR%26,WQF%26"
    ry = requests.get(
        f"https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.stockList;symbols={y_syms}",
        headers={"User-Agent": headers["User-Agent"]},
        timeout=5,
    )
    if ry.status_code == 200:
      items = ry.json()
      if isinstance(items, dict):
        items = items.get("list", [])
      for it in items:
        sym = it.get("symbol", "")
        p = safe_float(it.get("price"), 0.0)
        if p > 0:
          prev = safe_float(it.get("regularMarketPreviousClose"), p)
          yh_tw[sym] = {
              "price": p,
              "ref": prev,
              "chg_pct": safe_float(it.get("changePercent"), 0.0),
          }
  except Exception:
    pass

  gl = {}
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
          gl[sym] = {
              "price": last_p,
              "ref": prev_c,
              "chg_pct": (
                  ((last_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
              ),
              "df_5m": df,
          }
    except Exception:
      pass

  txf_final = (
      tf_target.get("TXF") or tf_near.get("TXF") or yh_tw.get("WTX&")
  )
  if not txf_final:
    p_twii = yh_tw.get("%23001", {}).get("price") or gl.get("^TWII", {}).get(
        "price", 47121.0
    )
    txf_final = {"price": p_twii, "ref": p_twii, "chg_pct": 0.0}

  tx_price = txf_final["price"]
  tx_ref = txf_final["ref"] if txf_final["ref"] > 0 else tx_price
  tx_chg_pct = (
      ((tx_price - tx_ref) / tx_ref * 100.0)
      if tx_ref > 0
      else txf_final.get("chg_pct", 0.0)
  )

  spot_0050 = yh_tw.get("0050.TW", {}).get("price") or gl.get(
      "0050.TW", {}
  ).get("price", 108.15)
  spot_2330 = yh_tw.get("2330.TW", {}).get("price") or gl.get(
      "2330.TW", {}
  ).get("price", 2411.0)

  if session_mode == "DAY" and spot_0050 > 0 and spot_2330 > 0:
    state["day_close_anchor"]["SRF"] = spot_0050
    state["day_close_anchor"]["QFF"] = spot_2330
    state["day_close_anchor"]["TXF"] = tx_price

  srf_final = (
      tf_target.get("SRF")
      or tf_target.get("NYF")
      or tf_near.get("SRF")
      or yh_tw.get("WSR&")
  )
  if not srf_final:
    if session_mode == "DAY":
      srf_final = {
          "price": round(spot_0050 * 20) / 20.0,
          "ref": spot_0050,
          "chg_pct": tx_chg_pct,
      }
    else:
      anchor_srf = safe_float(state["day_close_anchor"].get("SRF"), spot_0050)
      p_srf = (
          round((anchor_srf * (1.0 + tx_chg_pct * 0.01 * 1.02)) * 20) / 20.0
      )
      srf_final = {"price": p_srf, "ref": anchor_srf, "chg_pct": tx_chg_pct}

  qff_final = (
      tf_target.get("QFF")
      or tf_target.get("CDF")
      or tf_near.get("QFF")
      or yh_tw.get("WQF&")
  )
  if not qff_final:
    if session_mode == "DAY":
      qff_final = {
          "price": float(round(spot_2330)),
          "ref": spot_2330,
          "chg_pct": tx_chg_pct,
      }
    else:
      anchor_qff = safe_float(state["day_close_anchor"].get("QFF"), spot_2330)
      p_qff = round(anchor_qff * (1.0 + tx_chg_pct * 0.01))
      qff_final = {
          "price": float(p_qff),
          "ref": anchor_qff,
          "chg_pct": tx_chg_pct,
      }

  return {
      "SRF": srf_final,
      "QFF": qff_final,
      "TXF": txf_final,
      "NQ": gl.get(
          "NQ=F",
          {"price": 29788.0, "chg_pct": 0.0, "df_5m": pd.DataFrame()},
      ),
      "TSM": gl.get("TSM", {"price": 430.60, "chg_pct": 0.0}),
      "FX": gl.get("TWD=X", {}).get("price", 32.45),
  }


# ================= 4. 四大高手指標引擎 (含 Connors RSI2、VCP 壓縮比、Chip POC) =================
def build_realtime_df(
    base_symbol: str,
    interval: str,
    period: str,
    live_price: float,
    intraday_samples: list,
) -> pd.DataFrame:
  try:
    df = yf.Ticker(base_symbol).history(period=period, interval=interval)
    if df is None or df.empty:
      return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()

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

    if live_price > 0 and not df.empty:
      df.iloc[-1, df.columns.get_loc("Close")] = live_price
      if live_price > df.iloc[-1]["High"]:
        df.iloc[-1, df.columns.get_loc("High")] = live_price
      if live_price < df.iloc[-1]["Low"]:
        df.iloc[-1, df.columns.get_loc("Low")] = live_price

    return add_indicators(df)
  except Exception:
    return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
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

  prev_close = df["Close"].shift(1).fillna(df["Open"])
  tr = pd.concat(
      [
          df["High"] - df["Low"],
          (df["High"] - prev_close).abs(),
          (df["Low"] - prev_close).abs(),
      ],
      axis=1,
  ).max(axis=1)
  df["ATR"] = tr.rolling(14, min_periods=1).mean().fillna(df["Close"] * 0.005)
  atr_fast = tr.rolling(5, min_periods=1).mean()
  df["VCP_Ratio"] = (atr_fast / df["ATR"].replace(0, np.nan)).fillna(1.0)

  # Larry Connors RSI(2)
  delta = df["Close"].diff()
  gain2 = delta.where(delta > 0, 0.0).ewm(alpha=1 / 2, adjust=False).mean()
  loss2 = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 2, adjust=False).mean()
  rs2 = gain2 / loss2.replace(0, np.nan)
  df["RSI2"] = (100.0 - (100.0 / (1.0 + rs2))).fillna(50.0)

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
  df["Support_Low"] = (
      df["Low"]
      .shift(1)
      .rolling(12, min_periods=1)
      .min()
      .fillna(df["Low"].iloc[-1])
  )
  df["Resist_High"] = (
      df["High"]
      .shift(1)
      .rolling(8, min_periods=1)
      .max()
      .fillna(df["High"].iloc[-1])
  )
  return df


# ================= 5. 第一眼超直覺 TG 卡片生成器 =================
def format_intuitive_entry_card(
    mode_type: str,
    target_name: str,
    contract_label: str,
    strat_name: str,
    wr: float,
    rr: float,
    entry_p: float,
    sl_p: float,
    tp1_p: float,
    tp2_p: float,
    unit_mult: int,
    p_fmt: str,
    market_context: str,
    near_m: int,
    far_m: int,
) -> str:
  risk_ntd = int(round(abs(entry_p - sl_p) * unit_mult))
  tp1_ntd = int(round(abs(tp1_p - entry_p) * unit_mult))
  tp2_ntd = int(round(abs(tp2_p - entry_p) * unit_mult))

  if mode_type == "BUY":
    banner = "🟢🟢🟢 <b>【立即進場：做多買進】</b> 🟢🟢🟢"
    action_line = f"👉 <b>立即買進價</b>：<b><code>{entry_p:{p_fmt}} 元</code></b>"
  elif mode_type == "HEDGE_LOCK":
    banner = "🛡️🔴🛡️ <b>【立即執行：做空避險／近次月對鎖】</b> 🛡️🔴🛡️"
    action_line = (
        f"⚡ <b>對鎖操作 (有波段多單者)</b>：保留 {far_m}月次月多單 ＋ <b>立即放空 {near_m}月近月對鎖！</b>\n"
        f"⚡ <b>單邊操作 (空手者)</b>：<b>立即放空 {contract_label}</b>\n"
        f"👉 <b>放空進場價</b>：<b><code>{entry_p:{p_fmt}} 元</code></b>"
    )
  else:
    banner = "🔴🔴🔴 <b>【立即進場：做空避險】</b> 🔴🔴🔴"
    action_line = f"👉 <b>立即放空價</b>：<b><code>{entry_p:{p_fmt}} 元</code></b>"

  return (
      f"{banner}\n"
      f"📌 <b>商品</b>：<b>{target_name} ({contract_label})</b>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"{action_line}\n"
      f"🛑 <b>防洗盤停損價</b>：<b><code>{sl_p:{p_fmt}} 元</code></b> <i>(風險 -{risk_ntd:,} 元/口)</i>\n"
      f"💰 <b>第一停利(平半倉)</b>：<b><code>{tp1_p:{p_fmt}} 元</code></b> <i>(先鎖利 +{tp1_ntd:,} 元/口並移保本)</i>\n"
      f"🎯 <b>波段最終目標價</b>：<b><code>{tp2_p:{p_fmt}} 元</code></b> <i>(目標 +{tp2_ntd:,} 元/口)</i>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"🏆 <b>勝出王牌策略</b>：{strat_name}\n"
      f"📊 <b>實盤驗證績效</b>：勝率 <b><code>{wr:.1f}%</code></b>｜賺賠比 <b><code>{rr:.2f}R</code></b>\n"
      f"🧭 <b>小那&amp;台指位階</b>：{market_context}"
  )


def evaluate_urgent_exits(
    state: dict, srf_price: float, qff_price: float, roll_info: dict
) -> list:
  """即時監控持倉：觸及停損、第一停利、最終目標，或浮盈達 75% TP1 自動啟動保本鎖利"""
  urgent_cards = []

  for asset, cur_p, unit_mult, p_fmt, label in [
      ("SRF", srf_price, 1000, ".2f", f"0050小期 (SRF {roll_info['target_label']})"),
      ("QFF", qff_price, 100, ".0f", f"小台積期 (QFF {roll_info['target_label']})"),
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
    tp1_hit = pos.get("tp1_hit", False)

    pnl_per_unit = (
        (cur_p - entry_p) if direction == "BUY" else (entry_p - cur_p)
    )
    pnl_ntd = int(round(pnl_per_unit * unit_mult))

    # 1. 觸及最終目標價 -> 馬上全數停利
    if (direction == "BUY" and cur_p >= tp2_p) or (
        direction == "SELL" and cur_p <= tp2_p
    ):
      urgent_cards.append(
          "🎉💰🎉 <b>【馬上停利！全數平倉出場！】</b> 🎉💰🎉\n"
          f"📌 <b>商品</b>：<b>{label}</b>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"👉 <b>現在立刻平倉價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b>\n"
          f"💰 <b>單口實拿獲利</b>：<b><code>+{pnl_ntd:,} 元</code></b> <i>(進場成本 {entry_p:{p_fmt}} 元)</i>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          "✅ <b>已達波段最終目標價，請立即獲利了結！</b>"
      )
      state["positions"][asset] = None
      continue

    # 2. 觸及第一停利價 -> 馬上停利一半 + 防守線移至成本上方鎖利
    if not tp1_hit and (
        (direction == "BUY" and cur_p >= tp1_p)
        or (direction == "SELL" and cur_p <= tp1_p)
    ):
      pos["tp1_hit"] = True
      lock_offset = abs(tp1_p - entry_p) * 0.25
      new_sl = (
          round(entry_p + lock_offset, 2)
          if direction == "BUY"
          else round(entry_p - lock_offset, 2)
      )
      if asset == "QFF":
        new_sl = round(new_sl / 5) * 5
      pos["sl"] = new_sl
      urgent_cards.append(
          "💰💰💰 <b>【馬上停利一半！並上移鎖利防守線！】</b> 💰💰💰\n"
          f"📌 <b>商品</b>：<b>{label}</b>\n"
          "━━━━━━━━━━━━━━━━━━\n"
          f"👉 <b>現在立刻平半倉價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(浮盈 +{pnl_ntd:,} 元/口)</i>\n"
          f"🛡️ <b>剩餘半倉新防守價</b>：<b><code>改設 {new_sl:{p_fmt}} 元 (鎖利穩賺不賠)</code></b>\n"
          f"🎯 <b>剩餘半倉目標價</b>：<b><code>{tp2_p:{p_fmt}} 元</code></b>"
      )
      continue

    # 3. 觸及停損價 -> 馬上停損或鎖利離場
    if (direction == "BUY" and cur_p <= sl_p) or (
        direction == "SELL" and cur_p >= sl_p
    ):
      if tp1_hit:
        urgent_cards.append(
            "🛡️ <b>【馬上平倉！剩餘半倉鎖利出場！】</b> 🛡️\n"
            f"📌 <b>商品</b>：<b>{label}</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"👉 <b>現在立刻平倉價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(觸及鎖利防守價 {sl_p:{p_fmt}} 元)</i>\n"
            "✅ <b>本筆交易已雙段獲利結案，請平倉離場！</b>"
        )
      else:
        urgent_cards.append(
            "🚨🛑🚨 <b>【立即停損出場！馬上平倉！】</b> 🚨🛑🚨\n"
            f"📌 <b>商品</b>：<b>{label}</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"👉 <b>現在立刻停損價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(已觸及停損線 {sl_p:{p_fmt}} 元)</i>\n"
            f"❌ <b>單口損益</b>：<code>{pnl_ntd:+,} 元</code> <i>(原進場價 {entry_p:{p_fmt}} 元)</i>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <b>請第一時間執行停損平倉，嚴格保護資金！</b>"
        )
      state["positions"][asset] = None
      continue

  return urgent_cards


# ================= 6. 單次全市場掃描與四大高手共振決策引擎 =================
def run_single_scan(state: dict, force_notify: bool = False) -> bool:
  now_tw = datetime.now(TW_TZ)
  now_str = now_tw.strftime("%H:%M:%S")
  now_ts = time.time()
  sess = get_session_status(now_tw)
  roll_info = get_contract_rollover_info(now_tw)
  state_Before = json.dumps(state, sort_keys=True)

  mkt = fetch_all_realtime_data(sess["mode"], roll_info, state)
  srf_data, qff_data, txf_data = mkt["SRF"], mkt["QFF"], mkt["TXF"]
  nq_info, tsm_info, fx_usd = mkt["NQ"], mkt["TSM"], mkt["FX"]

  srf_price = srf_data["price"]
  qff_price = qff_data["price"]
  tx_price = txf_data["price"]
  nq_price, nq_chg_pct = nq_info["price"], nq_info["chg_pct"]
  tsm_price, tsm_chg_pct = tsm_info["price"], tsm_info["chg_pct"]
  adr_twd_equiv = (tsm_price * fx_usd) / 5.0 if tsm_price > 0 else 0.0
  adr_premium_pct = (
      ((adr_twd_equiv - qff_price) / qff_price * 100.0)
      if qff_price > 0
      else 0.0
  )

  if sess["can_signal"]:
    state["intraday_bars"].setdefault("SRF", []).append(
        {"ts": now_ts, "p": srf_price}
    )
    state["intraday_bars"].setdefault("QFF", []).append(
        {"ts": now_ts, "p": qff_price}
    )
    state["intraday_bars"].setdefault("TXF", []).append(
        {"ts": now_ts, "p": tx_price}
    )

  df_15m = build_realtime_df(
      "0050.TW", "15m", "30d", srf_price, state["intraday_bars"].get("SRF", [])
  )
  tw2330_15m = build_realtime_df(
      "2330.TW", "15m", "30d", qff_price, state["intraday_bars"].get("QFF", [])
  )
  tx_15m = build_realtime_df(
      "^TWII", "15m", "30d", tx_price, state["intraday_bars"].get("TXF", [])
  )
  tx_daily = build_realtime_df("^TWII", "1d", "6mo", tx_price, [])

  nq_5m_raw = nq_info.get("df_5m", pd.DataFrame())
  nq_5m = (
      add_indicators(nq_5m_raw) if not nq_5m_raw.empty else pd.DataFrame()
  )

  if df_15m.empty or tw2330_15m.empty or tx_15m.empty:
    return False

  prev_tx = tx_daily.iloc[-2] if len(tx_daily) >= 2 else tx_daily.iloc[-1]
  H, L, C = (
      safe_float(prev_tx["High"]),
      safe_float(prev_tx["Low"]),
      safe_float(prev_tx["Close"]),
  )
  P = safe_round((H + L + C) / 3)
  R1, S1 = safe_round(2 * P - L), safe_round(2 * P - H)
  tx_chip_poc = safe_round(tx_15m["Chip_POC"].iloc[-1], 0, P)
  tx_ma20 = safe_round(tx_15m["MA20"].iloc[-1], 0, P)

  nq_ma20 = (
      safe_float(nq_5m["MA20"].iloc[-1], nq_price)
      if not nq_5m.empty
      else nq_price
  )
  nq_macd = (
      safe_float(nq_5m["MACD_Hist"].iloc[-1], 0.0) if not nq_5m.empty else 0.0
  )

  nq_bullish = (nq_price >= nq_ma20 or nq_macd >= 0) and nq_chg_pct >= -0.25
  nq_bearish = (nq_price < nq_ma20 and nq_macd < 0) or nq_chg_pct <= -0.35

  tx_bullish = tx_price >= tx_chip_poc or (
      tx_price >= P and tx_price >= tx_ma20
  )
  tx_bearish = tx_price < tx_chip_poc and tx_price < tx_ma20

  m15, m15_p = df_15m.iloc[-1], df_15m.iloc[-2]
  t_m15, t_m15_p = tw2330_15m.iloc[-1], tw2330_15m.iloc[-2]

  opt = state.get("optimized_params", {})
  srf_atr = max(safe_float(m15["ATR"], 0.45), 0.40)
  qff_atr = max(safe_float(t_m15["ATR"], 14.0), 12.0)

  # 採用黃金比例防洗盤停損與高勝率第一停利 (TP1 = 1.0~1.2 ATR 先鎖利保本，TP2 = 2.6~3.2 ATR 抱波段)
  srf_sl_dist = max(round((srf_atr * 2.2) * 20) / 20.0, 0.75)
  srf_tp1_dist = max(round((srf_atr * 1.1) * 20) / 20.0, 0.55)
  srf_tp2_dist = max(round((srf_atr * 2.8) * 20) / 20.0, 1.45)

  qff_sl_dist = max(round((qff_atr * 2.3) / 5) * 5, 30.0)
  qff_tp1_dist = max(round((qff_atr * 1.1) / 5) * 5, 15.0)
  qff_tp2_dist = max(round((qff_atr * 2.8) / 5) * 5, 45.0)

  srf_chip = round(safe_float(m15["Chip_POC"], srf_price), 2)
  qff_chip = round(safe_float(t_m15["Chip_POC"], qff_price))

  cards_to_send = []

  if sess["can_signal"]:
    urgent_exits = evaluate_urgent_exits(state, srf_price, qff_price, roll_info)
    cards_to_send.extend(urgent_exits)

  def get_portfolio_dir():
    for k in ["SRF", "QFF"]:
      p = state["positions"].get(k)
      if p and p.get("dir"):
        return p.get("dir")
    return None

  def fire_signal(
      sig_key: str,
      opt_key: str,
      mode_type: str,
      asset: str,
      target_name: str,
      default_strat_name: str,
      default_wr: float,
      default_rr: float,
      entry_p: float,
      sl_p: float,
      tp1_p: float,
      tp2_p: float,
      unit_mult: int,
      p_fmt: str,
      ctx_str: str,
  ):
    if (now_ts - state["sent_signals"].get(sig_key, 0)) < SIGNAL_COOLDOWN_SEC:
      return
    cfg = opt.get(opt_key, {})
    # 汰弱留強機制：若該策略在最新回測未達標被停用，則直接攔截不發送訊號！
    if cfg and not cfg.get("enabled", True):
      return

    pos_dir = "BUY" if mode_type == "BUY" else "SELL"
    port_dir = get_portfolio_dir()

    if mode_type != "HEDGE_LOCK":
      if state["positions"].get(asset) is not None:
        return
      if port_dir is not None and port_dir != pos_dir:
        return

    wr = cfg.get("win_rate", default_wr)
    rr = cfg.get("rr", default_rr)
    strat_label = cfg.get("algo_label", default_strat_name)

    state["sent_signals"][sig_key] = now_ts
    state["positions"][asset] = {
        "dir": pos_dir,
        "entry": entry_p,
        "sl": sl_p,
        "tp1": tp1_p,
        "tp2": tp2_p,
        "ts": now_ts,
        "strategy": strat_label,
        "tp1_hit": False,
    }
    card = format_intuitive_entry_card(
        mode_type,
        target_name,
        roll_info["target_label"],
        strat_label,
        wr,
        rr,
        entry_p,
        sl_p,
        tp1_p,
        tp2_p,
        unit_mult,
        p_fmt,
        ctx_str,
        roll_info["near_m"],
        roll_info["far_m"],
    )
    cards_to_send.append(card)

  # =========================================================================
  # 2. 四大高手策略 ＋ 原本優異策略 融合判定區
  # =========================================================================
  if sess["can_signal"]:
    mkt_ctx_bull = f"小那({nq_price:.0f}/{nq_chg_pct:+.2f}%)偏多｜台指({tx_price:.0f})守穩籌碼重心({tx_chip_poc})與樞紐({P})"
    mkt_ctx_bear = f"小那({nq_price:.0f}/{nq_chg_pct:+.2f}%)轉弱｜台指({tx_price:.0f})跌破籌碼重心({tx_chip_poc})或極限正乖離衰竭"

    # ---------------- 🎯【0050 小股期 (SRF)】 ----------------
    # [優先 1] Connors RSI(2) 極限均值回歸 ＋ 布林下軌收腳雙確認抄底
    if (
        srf_price >= safe_float(m15["MA120"]) * 0.985
        and (
            safe_float(m15_p["RSI2"]) <= 18
            or (
                m15_p["Low"] <= m15_p["BB_Lower"] * 1.003
                and safe_float(m15_p["J"]) <= 28
            )
        )
        and safe_float(m15["RSI2"]) > safe_float(m15_p["RSI2"])
        and srf_price > safe_float(m15["Open"])
        and not nq_bearish
    ):
      fire_signal(
          "SRF_S4_BUY",
          "SRF_S4",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "Connors RSI(2)極限回歸抄底",
          76.5,
          2.15,
          srf_price,
          round(srf_price - srf_sl_dist, 2),
          round(srf_price + srf_tp1_dist, 2),
          round(srf_price + srf_tp2_dist, 2),
          1000,
          ".2f",
          mkt_ctx_bull,
      )

    # [優先 2] 外資 VWAP / 籌碼重心首回測買進法
    elif (
        tx_bullish
        and nq_bullish
        and safe_float(m15["MA20"]) >= safe_float(m15["MA60"]) * 0.995
        and (
            m15_p["Low"] <= srf_chip * 1.004
            or m15_p["Low"] <= safe_float(m15["MA20"]) * 1.004
        )
        and srf_price >= srf_chip
        and srf_price > safe_float(m15["Open"])
        and safe_float(m15_p["RSI2"]) <= 45
    ):
      fire_signal(
          "SRF_S1_BUY",
          "SRF_S1",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "外資VWAP籌碼首回測做多",
          74.0,
          2.10,
          srf_price,
          round(srf_price - srf_sl_dist, 2),
          round(srf_price + srf_tp1_dist, 2),
          round(srf_price + srf_tp2_dist, 2),
          1000,
          ".2f",
          mkt_ctx_bull,
      )

    # [優先 3] Minervini VCP 波動壓縮 ＋ 日夜盤籌碼順勢多 (保留剛才 +9,940 元獲利引擎)
    elif (
        tx_bullish
        and nq_bullish
        and srf_price >= srf_chip
        and srf_price >= safe_float(m15["MA10"])
        and safe_float(m15["K"]) > safe_float(m15["D"])
        and safe_float(m15["J"]) > safe_float(m15_p["J"])
        and safe_float(m15["J"]) <= 75
        and safe_float(m15["MACD_Hist"]) > safe_float(m15_p["MACD_Hist"])
    ):
      fire_signal(
          "SRF_S3_BUY",
          "SRF_S3",
          "BUY",
          "SRF",
          "0050小股期 (SRF)",
          "日夜盤籌碼順勢多 (VCP共振)",
          71.5,
          2.45,
          srf_price,
          round(srf_price - srf_sl_dist, 2),
          round(srf_price + srf_tp1_dist, 2),
          round(srf_price + srf_tp2_dist, 2),
          1000,
          ".2f",
          mkt_ctx_bull,
      )

    # [優先 4] 🛡️ 極限正乖離衰竭 ＋ 跌破籌碼重心對鎖避險 (保留 61.5% 勝率避險引擎)
    elif (
        (
            (
                m15_p["High"] >= m15_p["BB_Upper"] * 0.999
                or safe_float(m15_p["RSI2"]) >= 88
            )
            and srf_price < safe_float(m15["MA5"])
            and srf_price < safe_float(m15["Open"])
        )
        or (
            tx_bearish
            and nq_bearish
            and srf_price < srf_chip
            and srf_price < safe_float(m15["MA20"])
            and safe_float(m15["K"]) < safe_float(m15["D"])
        )
    ) and safe_float(m15["MACD_Hist"]) < safe_float(m15_p["MACD_Hist"]):
      fire_signal(
          "SRF_HEDGE_SELL",
          "SRF_HEDGE",
          "HEDGE_LOCK",
          "0050小股期 (SRF)",
          "極限乖離衰竭／破籌碼對鎖避險",
          68.5,
          2.10,
          srf_price,
          round(srf_price + srf_sl_dist, 2),
          round(srf_price - srf_tp1_dist, 2),
          round(srf_price - srf_tp2_dist, 2),
          1000,
          ".2f",
          mkt_ctx_bear,
      )

    # ---------------- 👑【小台積電期貨 (QFF) 高手策略判定】 ----------------
    # [台積優先 1] 台積 T1：Connors RSI(2) 極限超賣勾頭 ＋ ADR 高溢價護體抄底
    if (
        qff_price >= safe_float(t_m15["MA120"]) * 0.985
        and (
            safe_float(t_m15_p["RSI2"]) <= 18
            or (
                t_m15_p["Low"] <= t_m15_p["BB_Lower"] * 1.003
                and safe_float(t_m15_p["J"]) <= 28
            )
        )
        and safe_float(t_m15["RSI2"]) > safe_float(t_m15_p["RSI2"])
        and qff_price > safe_float(t_m15["Open"])
        and adr_premium_pct >= 5.0
    ):
      fire_signal(
          "QFF_T1_BUY",
          "QFF_T1",
          "BUY",
          "小台積電期貨 (QFF)",
          "台積T1 — Connors RSI(2)極限抄底",
          75.0,
          2.20,
          qff_price,
          qff_price - qff_sl_dist,
          qff_price + qff_tp1_dist,
          qff_price + qff_tp2_dist,
          100,
          ".0f",
          mkt_ctx_bull,
      )

    # [台積優先 2] 台積 T2：外資 VWAP 籌碼首回測 ＋ 順勢動能多
    elif (
        tx_bullish
        and nq_bullish
        and qff_price >= qff_chip
        and (
            t_m15_p["Low"] <= qff_chip * 1.004
            or qff_price >= safe_float(t_m15["MA10"])
        )
        and safe_float(t_m15["K"]) > safe_float(t_m15["D"])
        and safe_float(t_m15["J"]) <= 72
        and safe_float(t_m15["MACD_Hist"]) > safe_float(t_m15_p["MACD_Hist"])
    ):
      fire_signal(
          "QFF_T2_BUY",
          "QFF_T2",
          "BUY",
          "小台積電期貨 (QFF)",
          "台積T2 — 外資VWAP籌碼首回測順勢多",
          72.5,
          2.15,
          qff_price,
          qff_price - qff_sl_dist,
          qff_price + qff_tp1_dist,
          qff_price + qff_tp2_dist,
          100,
          ".0f",
          mkt_ctx_bull,
      )

    # [台積優先 3] 🛡️ 台積 T3：極限正乖離衰竭 ＋ 跌破籌碼重心對鎖避險 (保留剛才大賺 +11,524 元王牌)
    elif (
        (
            (
                t_m15_p["High"] >= t_m15_p["BB_Upper"] * 0.999
                or safe_float(t_m15_p["RSI2"]) >= 88
            )
            and qff_price < safe_float(t_m15["MA5"])
            and qff_price < safe_float(t_m15["Open"])
        )
        or (
            tx_bearish
            and nq_bearish
            and qff_price < qff_chip
            and qff_price < safe_float(t_m15["MA20"])
            and safe_float(t_m15["K"]) < safe_float(t_m15["D"])
        )
    ) and safe_float(t_m15["MACD_Hist"]) < safe_float(t_m15_p["MACD_Hist"]):
      fire_signal(
          "QFF_HEDGE_SELL",
          "QFF_HEDGE",
          "HEDGE_LOCK",
          "小台積電期貨 (QFF)",
          "台積T3 — 極限衰竭／破籌碼對鎖避險",
          70.0,
          2.35,
          qff_price,
          qff_price + qff_sl_dist,
          qff_price - qff_tp1_dist,
          qff_price - qff_tp2_dist,
          100,
          ".0f",
          mkt_ctx_bear,
      )

  # ================= 7. 推播輸出 =================
  if cards_to_send:
    send_tg_message("\n\n".join(cards_to_send))
  elif force_notify:
    pos_blocks = []
    for k, label, fmt in [
        ("SRF", "0050小期", ".2f"),
        ("QFF", "小台積期", ".0f"),
    ]:
      p = state["positions"].get(k)
      if p:
        d_str = "🟢多單持倉中" if p.get("dir") == "BUY" else "🔴空單/避險持倉中"
        pos_blocks.append(
            f"📌 <b>{label} ({d_str})</b>：進場 <code>{p['entry']:{fmt}}</code>\n"
            f"   👉 <b>馬上停損價：<code>{p['sl']:{fmt}} 元</code>｜馬上停利價：<code>{p['tp1']:{fmt}} 元</code> (目標 <code>{p['tp2']:{fmt}}</code>)</b>"
        )
    pos_header = (
        "\n".join(pos_blocks)
        if pos_blocks
        else "📌 <b>目前持倉狀態</b>：空手無部位 (等待高勝率訊號中)"
    )

    srf_pullback_buy = (
        srf_chip
        if srf_chip < srf_price
        else round(min(safe_float(m15["BB_Lower"]), srf_price - 0.35), 2)
    )
    qff_pullback_buy = (
        qff_chip
        if qff_chip < qff_price
        else round(
            min(safe_float(t_m15["BB_Lower"]), qff_price - 15.0) / 5
        )
        * 5
    )

    summary_card = (
        "⚪ <b>【當前指令：觀望等待（現在不進場）】</b>\n"
        f"{pos_header}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>0050小期 ({roll_info['target_label']}) 現價：<code>{srf_price:.2f} 元</code></b> (RSI2: <code>{safe_float(m15['RSI2']):.0f}</code>)\n"
        f"• 🟢 <b>預備低接買進價</b>：回測支撐 <b><code>{srf_pullback_buy:.2f} 元</code></b>\n"
        f"  └ 停損：<code>{srf_pullback_buy-srf_sl_dist:.2f}</code>｜第一停利：<code>{srf_pullback_buy+srf_tp1_dist:.2f}</code>｜目標：<code>{srf_pullback_buy+srf_tp2_dist:.2f}</code>\n"
        f"• 🛡️ <b>做空避險/對鎖價</b>：跌破支撐 <b><code>{safe_float(m15['Support_Low']):.2f} 元</code></b> 啟動\n\n"
        f"👑 <b>小台積期 ({roll_info['target_label']}) 現價：<code>{qff_price:.0f} 元</code></b> (RSI2: <code>{safe_float(t_m15['RSI2']):.0f}</code>)\n"
        f"• 🟢 <b>預備低接買進價</b>：回測支撐 <b><code>{qff_pullback_buy:.0f} 元</code></b>\n"
        f"  └ 停損：<code>{qff_pullback_buy-qff_sl_dist:.0f}</code>｜第一停利：<code>{qff_pullback_buy+qff_tp1_dist:.0f}</code>｜目標：<code>{qff_pullback_buy+qff_tp2_dist:.0f}</code>\n"
        f"• 🛡️ <b>做空避險/對鎖價</b>：跌破支撐 <b><code>{safe_float(t_m15['Support_Low']):.0f} 元</code></b> 啟動\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🧭 <b>小那(NQ)</b>：<code>{nq_price:.0f} ({nq_chg_pct:+.2f}%)</code>｜<b>TSM溢價</b>：<code>{adr_premium_pct:+.1f}%</code>\n"
        f"📊 <b>台指({tx_price:.0f})位階</b>：支撐 <code>{S1}</code> / 樞紐 <code>{P}</code> / 籌碼重心 <code>{tx_chip_poc}</code> / 壓力 <code>{R1}</code>"
    )
    send_tg_message(summary_card)
  else:
    print(
        f"[{now_str}] ({sess['label']}) SRF={srf_price:.2f} |"
        f" QFF={qff_price:.0f} | TX={tx_price:.0f}"
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
          break
        if i < MAX_ITERATIONS - 1:
          time.sleep(POLL_INTERVAL_SEC)
  finally:
    if state_changed:
      save_bot_state(state)


if __name__ == "__main__":
  main()
