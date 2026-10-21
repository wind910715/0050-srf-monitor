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
      # 預設寫入昨日真實夜盤收盤基準 (SRF=108.56, QFF=2426, TXF=46975)，防止清盤時段失真
      "last_valid_quotes": {
          "SRF": {
              "price": 108.56,
              "ref": 107.40,
              "chg_pct": 1.08,
              "name": "小型0050期貨(夜盤收盤)",
              "source": "期交所夜盤基準",
          },
          "QFF": {
              "price": 2426.0,
              "ref": 2405.0,
              "chg_pct": 0.87,
              "name": "小台積電期貨(夜盤收盤)",
              "source": "期交所夜盤基準",
          },
          "TXF": {
              "price": 46975.0,
              "ref": 46210.0,
              "chg_pct": 1.66,
              "name": "臺指期貨(夜盤收盤)",
              "source": "期交所夜盤基準",
          },
      },
  }
  if os.path.exists(STATE_FILE):
    try:
      with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        for k in default_state:
          if k not in data:
            data[k] = default_state[k]
        # 若舊 state.json 內存有昨天錯誤的 107.15 或 2400，強制修正為最新夜盤收盤價
        lv = data.get("last_valid_quotes", {})
        if safe_float(lv.get("SRF", {}).get("price")) < 107.50:
          data["last_valid_quotes"]["SRF"] = default_state[
              "last_valid_quotes"
          ]["SRF"]
        if safe_float(lv.get("QFF", {}).get("price")) <= 2405.0:
          data["last_valid_quotes"]["QFF"] = default_state[
              "last_valid_quotes"
          ]["QFF"]
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
      if k in state.get("night_bars", {}):
        state["night_bars"][k] = state["night_bars"][k][-150:]
    with open(STATE_FILE, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
  except Exception as e:
    print(f"保存狀態檔異常: {e}")


def get_session_mode(now_tw: datetime) -> str:
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


# ================= 2. 突破期交所分頁與清盤限制：雙盤別 + 股票期貨專屬爬蟲 =================
def fetch_taifex_precision_quotes(session_mode: str, state: dict) -> dict:
  """關鍵修正：

  1. 同時掃描期交所「當前盤別」與「上一盤別 (例如早盤開盤前掃描夜盤 MarketType=1)」，確保 08:00~08:45
  仍能抓到夜盤收盤價 (108.56 / 2426)！
  2. 針對股票期貨 (SRF, NYF, QFF, CDF) 獨立帶入 CID 與 SymbolType 查詢，解決排在第 200 筆之後被截斷的問題。
  3. 自動比較近月與次月合約，挑選「有最新成交價且成交量最大」的主力合約 (完美處理結算日換月)。
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

  # 決定查詢順序：日盤優先查 "0" 再備援 "1"(夜盤)；夜盤或早上開盤前優先查 "1"(夜盤) 再查 "0"(日盤)
  m_types = ["0", "1"] if session_mode == "
