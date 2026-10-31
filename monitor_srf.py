import calendar
from datetime import date, datetime
import json
import os
import re
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ================= 1. 環境變數與長駐高頻迴圈設定 =================
TG_BOT_TOKEN = (os.environ.get("TG_BOT_TOKEN") or "").strip()
TG_CHAT_ID = (os.environ.get("TG_CHAT_ID") or "").strip()
FORCE_REPORT = (
    os.environ.get("FORCE_REPORT") or "false"
).strip().lower() == "true"

POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "25"))
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "115"))

TW_TZ = pytz.timezone("Asia/Taipei")
STATE_FILE = "state.json"
SIGNAL_COOLDOWN_SEC = 900  # 同一策略 15 分鐘冷卻

# ================= 2026-10-27 重構：分層＋分欄位＋優先序（REFACTOR_PLAN_2026-10-27.md） =================
# 部位結構：positions[asset] = {"swing": pos|None, "daytrade": pos|None, "hedge": pos|None}
# 波段進場優先序（初版，待回測驗證）：SRF: S3→S1→S2→S4；QFF: T2→T1；對鎖獨立欄位。
# 出場優先序：硬停損 > 當沖收盤強制出場 > 波段 TP2 全出 > TP1 分批＋移動停利 > 對鎖。
# 波段：SL 1.25 ATR、TP1 2.0 ATR（出一半＋剩餘移保本）、TP2 3.5 ATR。
# 當沖：SL 1.0 ATR、TP1 1.5 ATR 全出、TP2 2.5 ATR、收盤強制出場為硬規則。
# 開盤緩衝（提案二選一，選「第一根 5m 收盤確認」）：開盤後 5 分鐘內（日盤 08:45-08:50、
# 夜盤 17:25-17:30）不觸發硬停損，TP／強制出場不受影響。
SWING_PRIORITY = {
    "SRF": ["SRF_S3_BUY", "SRF_S1_BUY", "SRF_S2_BUY", "SRF_S4_BUY"],
    "QFF": ["QFF_T2_BUY", "QFF_T1_BUY"],
}
SWING_PRIORITY_NUM = {
    "SRF_S3_BUY": "P1", "SRF_S1_BUY": "P2", "SRF_S2_BUY": "P3", "SRF_S4_BUY": "P4",
    "QFF_T2_BUY": "P1", "QFF_T1_BUY": "P2",
}
SWING_TP1_MULT = 2.0
DAYTRADE_SL_MULT = 1.0
DAYTRADE_TP1_MULT = 1.5
DAYTRADE_TP2_MULT = 2.5
OPEN_BUFFER_WINDOWS = [(845, 850), (1725, 1730)]  # (hm 起, hm 迄)：開盤後 5 分鐘不觸發硬停損
DAYTRADE_FORCE_CLOSE_DAY_HM = 1325   # 日盤當沖單 13:25 起強制出場（13:30 前出清）
DAYTRADE_FORCE_CLOSE_NIGHT_HM = 455  # 夜盤當沖單 04:55 起強制出場（05:00 前出清）

# 更新版本號：自動清除 105.05 假持倉，重新以 100,000 元乾淨帳本啟動！
ACCOUNT_RESET_VERSION = "2026-10-22_1815_PURE_TAIFEX_MIS"
INITIAL_CAPITAL = 100000

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


def migrate_positions(positions) -> dict:
  """舊格式 positions[asset] 為單一部位 dict 時，自動轉換為三欄位格式。

  規則：策略名含「對鎖」者進 hedge 欄位，其餘進 swing 欄位；
  保留原有 entry/sl/tp1/tp2/tp1_hit，並補 pos_type/priority/session/
  sl_dist/tp1_dist/tp2_dist/closed_lots/realized_partial 欄位。
  已是新格式者只補預設欄位，不動既有部位。
  """
  new_positions = {}
  for asset in ["SRF", "QFF"]:
    old = positions.get(asset) if isinstance(positions, dict) else None
    slots = {"swing": None, "daytrade": None, "hedge": None}
    if isinstance(old, dict) and old.get("dir"):
      strat = str(old.get("strategy", ""))
      slot = "hedge" if "對鎖" in strat else "swing"
      pos = dict(old)
      pos["pos_type"] = slot
      pos["priority"] = "H" if slot == "hedge" else None
      pos["session"] = None
      pos.setdefault("closed_lots", 0)
      pos.setdefault("realized_partial", 0)
      entry_p = safe_float(pos.get("entry"), 0.0)
      pos["sl_dist"] = abs(entry_p - safe_float(pos.get("sl"), entry_p))
      pos["tp1_dist"] = abs(safe_float(pos.get("tp1"), entry_p) - entry_p)
      pos["tp2_dist"] = abs(safe_float(pos.get("tp2"), entry_p) - entry_p)
      slots[slot] = pos
    elif isinstance(old, dict):
      for s in ["swing", "daytrade", "hedge"]:
        p = old.get(s)
        if isinstance(p, dict) and p.get("dir"):
          p.setdefault("pos_type", s)
          p.setdefault("priority", "H" if s == "hedge" else None)
          p.setdefault("session", None)
          p.setdefault("closed_lots", 0)
          p.setdefault("realized_partial", 0)
          entry_p = safe_float(p.get("entry"), 0.0)
          p.setdefault("sl_dist", abs(entry_p - safe_float(p.get("sl"), entry_p)))
          p.setdefault("tp1_dist", abs(safe_float(p.get("tp1"), entry_p) - entry_p))
          p.setdefault("tp2_dist", abs(safe_float(p.get("tp2"), entry_p) - entry_p))
          slots[s] = p
    new_positions[asset] = slots
  return new_positions


def load_bot_state() -> dict:
  default_state = {
      "account_version": ACCOUNT_RESET_VERSION,
      "sent_signals": {},
      "positions": {"SRF": {"swing": None, "daytrade": None, "hedge": None},
                    "QFF": {"swing": None, "daytrade": None, "hedge": None}},
      "account": {
          "initial_capital": INITIAL_CAPITAL,
          "realized_pnl": 0,
          "today_realized_pnl": 0,
          "today_date": datetime.now(TW_TZ).strftime("%Y-%m-%d"),
          "closed_trades_today": [],
          "total_trades_count": 0,
          "win_trades_count": 0,
      },
      "daily_flags": {
          "day_standby_date": "",
          "day_open_0845_date": "",
          "spot_open_0900_date": "",
          "report_1400_date": "",
          "night_standby_date": "",
          "night_open_1500_date": "",
          "night_srf_open_1725_date": "",
          "last_hourly_heartbeat": "",
          "last_startup_ping_ts": 0,
      },
      "intraday_bars": {"SRF": [], "QFF": [], "TXF": []},
      "last_real_quotes": {},
      "active_contract_ym": "",
      "optimized_params": {},
  }
  if os.path.exists(STATE_FILE):
    try:
      with open(STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        if data.get("account_version") != ACCOUNT_RESET_VERSION:
          opt_keep = data.get("optimized_params", {})
          data = default_state.copy()
          data["optimized_params"] = opt_keep
          print("🔄 已清除錯誤連動持倉，重置為 100,000 元純期交所真實報價帳本！")
        else:
          for k in default_state:
            if k not in data:
              data[k] = default_state[k]
          for fk in default_state["daily_flags"]:
            if fk not in data["daily_flags"]:
              data["daily_flags"][fk] = default_state["daily_flags"][fk]
          # 舊格式部位自動遷移為三欄位（對鎖→hedge、其餘→swing），現有模擬倉不得無聲消失
          data["positions"] = migrate_positions(data.get("positions", {}))
        today_str = datetime.now(TW_TZ).strftime("%Y-%m-%d")
        if data["account"].get("today_date") != today_str:
          data["account"]["today_date"] = today_str
          data["account"]["today_realized_pnl"] = 0
          data["account"]["closed_trades_today"] = []
        return data
    except Exception as e:
      # 損壞的 state.json 絕不可無聲重置 (會讓整本帳本/持倉憑空消失):
      # 先把損壞檔備份起來供人工還原, 再以預設狀態啟動並大聲告警。
      print(f"⚠️ state.json 讀取失敗 ({e})！已備份損壞檔, 以預設狀態啟動, 請人工檢查帳本！")
      try:
        os.replace(STATE_FILE, STATE_FILE + ".corrupt")
      except Exception:
        pass
  return default_state


def save_bot_state(state: dict, git_sync: bool = False):
  try:
    now_ts = time.time()
    state["sent_signals"] = {
        k: v for k, v in state["sent_signals"].items() if (now_ts - v) < 7200
    }
    for k in ["SRF", "QFF", "TXF"]:
      if k in state.get("intraday_bars", {}):
        state["intraday_bars"][k] = state["intraday_bars"][k][-180:]
    # 原子寫入: 先寫暫存檔再 rename, 避免程序中途被殺造成 state.json 截斷損毀。
    tmp_path = STATE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
      json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, STATE_FILE)
    if git_sync:
      subprocess.run(["git", "add", STATE_FILE], check=False)
      res = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
      if res.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", "chore: sync realtime bot state"],
            check=False,
        )
        # Actions 環境是 detached HEAD, 裸 pull/push 一定失敗 (不在分支上);
        # 有 GITHUB_REF_NAME 時明確指定遠端分支, 否則維持舊行為。
        ref = (os.environ.get("GITHUB_REF_NAME") or "").strip()
        if ref:
          subprocess.run(["git", "pull", "--rebase", "origin", ref], check=False)
          subprocess.run(["git", "push", "origin", f"HEAD:{ref}"], check=False)
        else:
          subprocess.run(["git", "pull", "--rebase"], check=False)
          subprocess.run(["git", "push"], check=False)
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
      "target_m": target_m,
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
  """精準區分：

  - 早盤：08:10~08:44 備戰，08:45~13:45 實時監控
  - 收盤報帳：13:46~14:24 結算報表
  - 夜盤暖機：14:25~14:59 備戰，15:00~17:24 台指夜盤先行(股期17:25開盤待命)
  - 夜盤全開：17:25~05:00 小0050(SRF)與小台積(QFF)夜盤正式交易！
  """
  weekday = now_tw.weekday()
  hm = now_tw.hour * 100 + now_tw.minute
  if (
      weekday == 6
      or (weekday == 5 and hm > 502)
      or (weekday == 0 and hm < 810)
  ):
    return {
        "mode": "CLOSED",
        "label": "休市中",
        "can_run": False,
        "can_signal": False,
    }
  if 810 <= hm < 845:
    return {
        "mode": "STANDBY_DAY",
        "label": "⏳ 早盤開機備戰中 (08:45開盤)",
        "can_run": True,
        "can_signal": False,
    }
  if 845 <= hm <= 1345:
    return {
        "mode": "DAY",
        "label": "☀️ 早盤實時監控中 (08:45-13:45)",
        "can_run": True,
        "can_signal": True,
    }
  if 1346 <= hm <= 1424:
    return {
        "mode": "POST_DAY_REPORT",
        "label": "📊 早盤收盤結算時段",
        "can_run": True,
        "can_signal": False,
    }
  if 1425 <= hm < 1500:
    return {
        "mode": "STANDBY_NIGHT",
        "label": "⏳ 夜盤開機備戰中 (台指15:00/股期17:25開盤)",
        "can_run": True,
        "can_signal": False,
    }
  if 1500 <= hm < 1725:
    return {
        "mode": "NIGHT_TXF_ONLY",
        "label": "🌙 台指夜盤監控中 (小0050/小台積 17:25 開盤待命)",
        "can_run": True,
        "can_signal": False,
    }
  if hm >= 1725 or hm <= 500:
    return {
        "mode": "NIGHT",
        "label": "🌙 夜盤全商品實時監控中 (17:25-05:00)",
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


# ================= 3. 純期交所 MIS 真實報價引擎 (還原 KindID="1" + 徹底廢除假連動公式) =================
def extract_taifex_item(q: dict, target_code: str, target_m: int):
  raw_id = str(q.get("SymbolID", "")).strip()
  disp_name = str(q.get("DispCName", "")).strip()
  if not raw_id or "/" in raw_id or raw_id.endswith("-S"):
    return None, None, None

  clean_id = re.sub(r"[\s_]+", "", raw_id)
  pfx = clean_id[:3]
  if pfx not in ["TXF", "SRF", "QFF", "CDF", "NYF", "MXF", "TMF"]:
    if "小型元大台灣50" in disp_name:
      pfx = "SRF"
    elif "小型台積電" in disp_name:
      pfx = "QFF"
    elif "台積電期貨" in disp_name and "小型" not in disp_name:
      pfx = "CDF"
    elif "元大台灣50" in disp_name and "小型" not in disp_name:
      pfx = "NYF"
    elif "臺指期" in disp_name:
      pfx = "TXF"
    else:
      return None, None, None

  m_match = re.search(r"([A-L]\d)", clean_id[3:])
  m_code = m_match.group(1) if m_match else ""
  is_target_month = (
      (m_code == target_code)
      or (f"{target_m:02d}" in disp_name)
      or (f"{target_m}月" in disp_name)
      or (f"26{target_m:02d}" in disp_name)
  )

  lp = safe_float(q.get("CLastPrice"), 0.0)
  bp = safe_float(q.get("CBidPrice1"), 0.0)
  ap = safe_float(q.get("CAskPrice1"), 0.0)
  ref = safe_float(q.get("CRefPrice"), 0.0)
  vol = int(safe_float(q.get("CTotalVolume"), 0))

  # 優先取最新成交價 (例如 106.60)；若該分鐘尚無成交但有買賣掛單，取委買委賣中價；若為開盤前待命則取平盤參考價 (106.85)
  has_live_trade = False
  if lp > 0:
    eff = lp
    has_live_trade = True
  elif bp > 0 and ap > 0:
    eff = round((bp + ap) / 2.0, 2)
    has_live_trade = True
  elif ref > 0:
    eff = ref
    has_live_trade = False
  else:
    return None, None, None

  return (
      pfx,
      is_target_month,
      {
          "symbol": clean_id,
          "name": disp_name,
          "price": eff,
          "last": lp,
          "bid": bp,
          "ask": ap,
          "ref": ref if ref > 0 else eff,
          "high": safe_float(q.get("CHighPrice"), eff),
          "low": safe_float(q.get("CLowPrice"), eff),
          "vol": vol,
          "has_live": has_live_trade,
          "chg_pct": ((eff - ref) / ref * 100.0) if ref > 0 else 0.0,
      },
  )


def fetch_all_realtime_data(
    session_mode: str, roll_info: dict, state: dict
) -> dict:
  headers = {
      "Content-Type": "application/json;charset=UTF-8",
      "Accept": "application/json, text/plain, */*",
      "Origin": "https://mis.taifex.com.tw",
      "Referer": "https://mis.taifex.com.tw/futures/AfterHoursSession/EquityFutures/",
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/124.0.0.0 Safari/537.36"
      ),
  }

  target_code = roll_info["target_code"]
  target_m = roll_info["target_m"]
  m_types = ["0", "1"] if session_mode == "DAY" else ["1", "0"]

  cands = {"TXF": [], "SRF": [], "QFF": [], "CDF": [], "NYF": []}
  s = requests.Session()

  # 先輕量取得期交所首頁 Cookie，避免 WAF 阻擋
  try:
    s.get(
        "https://mis.taifex.com.tw/futures/",
        headers={"User-Agent": headers["User-Agent"]},
        timeout=4,
    )
  except Exception:
    pass

  # 關鍵修復：全部使用早上 08:02 / 08:38 驗證成功的 KindID="1" 參數！
  for mt in m_types:
    queries = [
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": "SRF"},
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": "SRF"},
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": "QFF"},
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": "QFF"},
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": "TXF"},
        {"MarketType": mt, "SymbolType": "S", "KindID": "1", "CID": ""},
        {"MarketType": mt, "SymbolType": "F", "KindID": "1", "CID": ""},
    ]
    for q_cfg in queries:
      # 若當前盤別已成功抓到 TXF, SRF, QFF 的實時成交價，立即提早結束迴圈，不浪費請求！
      if (
          mt == m_types[0]
          and cands["TXF"]
          and cands["SRF"]
          and cands["QFF"]
      ):
        break
      try:
        r = s.post(
            "https://mis.taifex.com.tw/futures/api/getQuoteList",
            json={
                "MarketType": q_cfg["MarketType"],
                "SymbolType": q_cfg["SymbolType"],
                "KindID": q_cfg["KindID"],
                "CID": q_cfg["CID"],
                "ExpireMonth": "",
                "RowSize": "全部",
                "PageNo": "",
                "SortColumn": "",
                "AscDesc": "A",
            },
            headers=headers,
            timeout=6,
        )
        if r.status_code != 200:
          continue
        for q in r.json().get("RtData", {}).get("QuoteList", []):
          pfx, is_target, item = extract_taifex_item(q, target_code, target_m)
          if pfx and item:
            # 優先權：當下盤別 + 有實時成交/掛單 + 目標月份(2611/K6) + 成交量
            prio = (
                (4000000 if q_cfg["MarketType"] == m_types[0] else 0)
                + (2000000 if item["has_live"] else 0)
                + (1000000 if is_target else 0)
                + item["vol"]
            )
            item["priority"] = prio
            m_str = "日盤" if q_cfg["MarketType"] == "0" else "夜盤"
            item["source"] = (
                f"期交所{m_str}MIS"
                if item["has_live"]
                else f"期交所{m_str}參考價"
            )
            cands[pfx].append(item)
        time.sleep(0.15)
      except Exception:
        pass

  # B. 同步使用 getQuoteDetail 直連查詢 2611 (K6) 合約
  for mt_sfx, m_label in [
      ("-M" if session_mode != "DAY" else "-F", "夜盤" if session_mode != "DAY" else "日盤"),
      ("-F" if session_mode != "DAY" else "-M", "日盤" if session_mode != "DAY" else "夜盤"),
  ]:
    if cands["TXF"] and cands["SRF"] and cands["QFF"]:
      break
    detail_ids = [
        f"SRF{target_code}{mt_sfx}",
        f"QFF{target_code}{mt_sfx}",
        f"CDF{target_code}{mt_sfx}",
        f"NYF{target_code}{mt_sfx}",
        f"TXF{target_code}{mt_sfx}",
    ]
    try:
      rd = s.post(
          "https://mis.taifex.com.tw/futures/api/getQuoteDetail",
          json={"SymbolID": detail_ids},
          headers=headers,
          timeout=6,
      )
      if rd.status_code == 200:
        for q in rd.json().get("RtData", {}).get("QuoteList", []):
          pfx, is_target, item = extract_taifex_item(q, target_code, target_m)
          if pfx and item:
            item["priority"] = (
                5000000
                + (2000000 if item["has_live"] else 0)
                + item["vol"]
            )
            item["source"] = f"期交所{m_label}MIS"
            cands[pfx].append(item)
    except Exception:
      pass

  # C. 鉅亨網 (cnyes) 台指期日夜盤 24 小時即時 API (作為 TXF 零延遲第二來源)
  cnyes_txf = None
  try:
    rc = requests.get(
        "https://ws.api.cnyes.com/ws/api/v1/charting/history?symbol=TWF:TXF:FUTURES&resolution=1&limit=2",
        headers={"User-Agent": headers["User-Agent"]},
        timeout=5,
    )
    if rc.status_code == 200:
      d_cn = rc.json().get("data", {})
      c_list = d_cn.get("c", [])
      if c_list and safe_float(c_list[0]) > 0:
        p_cn = safe_float(c_list[0])
        cnyes_txf = {
            "price": p_cn,
            "ref": p_cn,
            "chg_pct": 0.0,
            "has_live": True,
            "source": "鉅亨網台指即時",
        }
  except Exception:
    pass

  # D. 美股 NQ=F (小那斯達克)、TSM (台積電ADR)、TWD=X 實時報價 (採用 Yahoo Chart meta 確保漲跌幅為當日真實盤別漲跌)
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
          # 優先從 fast_info 讀取真實當盤 previous_close，避免跨兩日重複計算跌幅
          try:
            prev_c = safe_float(tk.fast_info.get("previous_close"), 0.0)
          except Exception:
            prev_c = 0.0
          if prev_c <= 0:
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

  best_tf = {}
  for k, lst in cands.items():
    if lst:
      lst.sort(key=lambda x: x["priority"], reverse=True)
      best_tf[k] = lst[0]

  # 將成功抓到的期交所真實報價存入 last_real_quotes 記憶庫
  last_saved = state.setdefault("last_real_quotes", {})

  # 1. 決定 SRF (小型 0050 期貨) 報價 — 絕不使用假連動公式！
  srf_final = best_tf.get("SRF") or best_tf.get("NYF")
  if srf_final:
    last_saved["SRF"] = srf_final
  elif "SRF" in last_saved:
    srf_final = last_saved["SRF"].copy()
    srf_final["has_live"] = False
    srf_final["source"] = "期交所MIS(前次快取)"
  else:
    spot_0050 = gl.get("0050.TW", {}).get("price", 106.85)
    srf_final = {
        "price": spot_0050,
        "ref": spot_0050,
        "chg_pct": 0.0,
        "has_live": False,
        "source": "日盤收盤基準價",
    }

  # 2. 決定 QFF (小型台積電期貨) 報價 — 絕不使用假連動公式！
  qff_final = best_tf.get("QFF") or best_tf.get("CDF")
  if qff_final:
    last_saved["QFF"] = qff_final
  elif "QFF" in last_saved:
    qff_final = last_saved["QFF"].copy()
    qff_final["has_live"] = False
    qff_final["source"] = "期交所MIS(前次快取)"
  else:
    spot_2330 = gl.get("2330.TW", {}).get("price", 2390.0)
    qff_final = {
        "price": spot_2330,
        "ref": spot_2330,
        "chg_pct": 0.0,
        "has_live": False,
        "source": "日盤收盤基準價",
    }

  # 3. 決定 TXF (台指期) 報價
  txf_final = best_tf.get("TXF") or cnyes_txf
  if txf_final:
    last_saved["TXF"] = txf_final
  elif "TXF" in last_saved:
    txf_final = last_saved["TXF"].copy()
  else:
    p_twii = gl.get("^TWII", {}).get("price", 46041.0)
    txf_final = {
        "price": p_twii,
        "ref": p_twii,
        "chg_pct": 0.0,
        "has_live": False,
        "source": "加權收盤基準",
    }

  return {
      "SRF": srf_final,
      "QFF": qff_final,
      "TXF": txf_final,
      "NQ": gl.get(
          "NQ=F",
          {"price": 29444.0, "chg_pct": 0.0, "df_5m": pd.DataFrame()},
      ),
      "TSM": gl.get("TSM", {"price": 430.60, "chg_pct": 0.0}),
      "FX": gl.get("TWD=X", {}).get("price", 32.45),
  }


# ================= 4. K 線基差平移與指標計算引擎 =================
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

    last_df_ts = df.index[-1].timestamp() if not df.empty else 0
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # 將現貨 K 線基準平移對齊至期貨平盤/首筆真實價，消除期現貨價差斷層
    if live_price > 0 and not df.empty:
      last_spot_close = safe_float(df["Close"].iloc[-1], live_price)
      anchor_first_live = (
          safe_float(intraday_samples[0]["p"], live_price)
          if intraday_samples
          else live_price
      )
      basis_shift = anchor_first_live - last_spot_close
      if abs(basis_shift) / max(last_spot_close, 1.0) >= 0.0015:
        for col in ["Open", "High", "Low", "Close"]:
          df[col] = df[col] + basis_shift

    if interval in ["5m", "15m"] and intraday_samples:
      bucket_sec = 300 if interval == "5m" else 900
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
            datetime.fromtimestamp(b_ts, TW_TZ).replace(tzinfo=None)
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
  except Exception as e:
    print(f"build_realtime_df ({base_symbol}) 異常: {e}")
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


# ================= 5. 10萬保證金模擬帳本核算與每日 14:00 報表生成 =================
def compute_account_summary(
    state: dict, srf_price: float, qff_price: float
) -> dict:
  acc = state["account"]
  init_cap = acc.get("initial_capital", INITIAL_CAPITAL)
  realized_total = acc.get("realized_pnl", 0)
  realized_today = acc.get("today_realized_pnl", 0)

  unrealized_total = 0
  open_details = []

  for asset, cur_p, mult, fmt, name in [
      ("SRF", srf_price, 1000, ".2f", "0050小期(SRF)"),
      ("QFF", qff_price, 100, ".0f", "小台積期(QFF)"),
  ]:
    slots = state["positions"].get(asset) or {}
    if isinstance(slots, dict) and slots.get("dir"):
      slots = migrate_positions(state["positions"]).get(asset, {})
    for slot_name, slot_label in [
        ("swing", "波段"), ("daytrade", "當沖"), ("hedge", "對鎖"),
    ]:
      pos = slots.get(slot_name) if isinstance(slots, dict) else None
      if pos and cur_p > 0:
        lots = int(pos.get("lots", 1))
        entry_p = safe_float(pos.get("entry"), cur_p)
        d_type = pos.get("dir", "BUY")
        diff = (cur_p - entry_p) if d_type == "BUY" else (entry_p - cur_p)
        u_pnl = int(round(diff * mult * lots))
        unrealized_total += u_pnl
        d_str = "🟢多單" if d_type == "BUY" else "🔴空單/避險"
        open_details.append(
            f"• <b>{name}</b>【{slot_label}】：{d_str} <b><code>{lots} 口</code></b>｜進場 <code>{entry_p:{fmt}}</code> ➔ 現價 <code>{cur_p:{fmt}}</code> (浮動損益 <b><code>{u_pnl:+,} 元</code></b>)"
        )

  total_equity = init_cap + realized_total + unrealized_total
  roi_pct = (total_equity - init_cap) / init_cap * 100.0
  return {
      "init_cap": init_cap,
      "realized_total": realized_total,
      "realized_today": realized_today,
      "unrealized_total": unrealized_total,
      "total_equity": total_equity,
      "roi_pct": roi_pct,
      "open_details": open_details,
      "closed_today": acc.get("closed_trades_today", []),
  }


def format_1400_daily_report(
    state: dict,
    srf_price: float,
    qff_price: float,
    roll_info: dict,
    now_str: str,
) -> str:
  summary = compute_account_summary(state, srf_price, qff_price)
  open_str = (
      "\n".join(summary["open_details"])
      if summary["open_details"]
      else "• 目前空手無未平倉部位（0 口）"
  )
  closed_list = summary["closed_today"]
  closed_str = (
      "\n".join(closed_list) if closed_list else "• 今日尚無已平倉交易（0 筆）"
  )

  return (
      "📊 <b>【每日 14:00 早盤收盤帳務與持倉損益總結】</b>\n"
      f"🕒 結算時間：<code>{now_str}</code>｜合約：<code>{roll_info['target_label']}</code>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "🏦 <b>【保證金帳戶總覽 (初始本金 100,000 元)】</b>\n"
      f"• <b>今日已平倉損益</b>：<b><code>{summary['realized_today']:+,} 元</code></b>\n"
      f"• <b>累計已平倉損益</b>：<b><code>{summary['realized_total']:+,} 元</code></b>\n"
      f"• <b>目前持倉浮動損益</b>：<b><code>{summary['unrealized_total']:+,} 元</code></b>\n"
      f"💰 <b>目前帳戶總金額 (總權益)</b>：<b><code>{summary['total_equity']:,} 元 ({summary['roi_pct']:+.2f}%)</code></b>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "📌 <b>【目前持倉明細 (共幾口與浮動損益)】</b>\n"
      f"{open_str}\n"
      "━━━━━━━━━━━━━━━━━━\n"
      "🧾 <b>【今日已平倉交易明細】</b>\n"
      f"{closed_str}"
  )


# ================= 6. 第一眼超直覺進出場指令卡 =================
def format_intuitive_entry_card(
    mode_type: str,
    target_name: str,
    contract_label: str,
    lots: int,
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
    total_equity: int,
) -> str:
  risk_ntd = int(round(abs(entry_p - sl_p) * unit_mult * lots))
  tp1_ntd = int(round(abs(tp1_p - entry_p) * unit_mult * lots))
  tp2_ntd = int(round(abs(tp2_p - entry_p) * unit_mult * lots))
  rr_calc = round(
      (0.20 * abs(tp1_p - entry_p) + 0.80 * abs(tp2_p - entry_p))
      / max(abs(entry_p - sl_p), 0.01),
      2,
  )
  disp_rr = max(rr, rr_calc)

  if mode_type == "BUY":
    banner = "🟢🟢🟢 <b>【立即進場：做多買進】</b> 🟢🟢🟢"
    action_line = f"👉 <b>立即買進價</b>：<b><code>{entry_p:{p_fmt}} 元</code></b>｜<b>建議下單：<code>{lots} 口</code></b>"
  elif mode_type == "HEDGE_LOCK":
    banner = "🛡️🔴🛡️ <b>【立即執行：做空避險／近次月對鎖】</b> 🛡️🔴🛡️"
    action_line = (
        f"⚡ <b>對鎖操作 (有波段多單者)</b>：保留 {far_m}月次月多單 ＋ <b>立即放空 {near_m}月近月 <code>{lots} 口</code> 對鎖！</b>\n"
        f"⚡ <b>單邊操作 (空手者)</b>：<b>立即放空 {contract_label} <code>{lots} 口</code></b>\n"
        f"👉 <b>放空進場價</b>：<b><code>{entry_p:{p_fmt}} 元</code></b>"
    )
  else:
    banner = "🔴🔴🔴 <b>【立即進場：做空避險】</b> 🔴🔴🔴"
    action_line = f"👉 <b>立即放空價</b>：<b><code>{entry_p:{p_fmt}} 元</code></b>｜<b>建議下單：<code>{lots} 口</code></b>"

  return (
      f"{banner}\n"
      f"📌 <b>商品</b>：<b>{target_name} ({contract_label})</b>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"{action_line}\n"
      f"🛑 <b>嚴格停損價</b>：<b><code>{sl_p:{p_fmt}} 元</code></b> <i>({lots}口風險 -{risk_ntd:,} 元)</i>\n"
      f"💰 <b>第一停利價</b>：<b><code>{tp1_p:{p_fmt}} 元</code></b> <i>(先鎖利 +{tp1_ntd:,} 元並移保本)</i>\n"
      f"🎯 <b>波段目標價</b>：<b><code>{tp2_p:{p_fmt}} 元</code></b> <i>({lots}口目標 +{tp2_ntd:,} 元｜賺賠比 <b>{disp_rr:.2f}R</b>)</i>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"🏆 <b>策略</b>：{strat_name} (實測勝率 <b>{wr:.1f}%</b>)\n"
      f"🏦 <b>當前帳戶總資金</b>：<code>{total_equity:,} 元</code>\n"
      f"🧭 <b>小那&amp;台指位階</b>：{market_context}"
  )





def _pos_attr(pos: dict, p_fmt: str) -> str:
  """平倉記錄歸因欄：[策略|類型|優先序|SL距|TP1距|TP2距]。"""
  strat = pos.get("strategy", "")
  ptype = pos.get("pos_type", "")
  pri = pos.get("priority") or "—"
  sl_d = safe_float(pos.get("sl_dist"), 0.0)
  t1_d = safe_float(pos.get("tp1_dist"), 0.0)
  t2_d = safe_float(pos.get("tp2_dist"), 0.0)
  return (
      f"[{strat}|{ptype}|{pri}|SL{sl_d:{p_fmt}}|"
      f"TP1 {t1_d:{p_fmt}}|TP2 {t2_d:{p_fmt}}]"
  )


def _book_full_close(
    state: dict,
    asset: str,
    slot_name: str,
    pos: dict,
    cur_p: float,
    unit_mult: int,
    p_fmt: str,
    label: str,
    tag: str,
    now_str: str,
) -> int:
  """全數平倉剩餘口數：結算損益（已含 TP1 分批已實現）、計數、寫平倉記錄、清欄位。

  回傳本筆總淨損益（含 TP1 分批部分）。一部位只計一筆交易，勝負以總淨損益判定。
  """
  acc = state["account"]
  lots = int(pos.get("lots", 1))
  entry_p = safe_float(pos.get("entry"), cur_p)
  direction = pos.get("dir", "BUY")
  diff = (cur_p - entry_p) if direction == "BUY" else (entry_p - cur_p)
  final_pnl = int(round(diff * unit_mult * lots)) - 50 * lots
  total_net = int(pos.get("realized_partial", 0)) + final_pnl
  acc["realized_pnl"] += final_pnl
  acc["today_realized_pnl"] += final_pnl
  acc["total_trades_count"] += 1
  if total_net > 0:
    acc["win_trades_count"] += 1
  total_lots = int(pos.get("closed_lots", 0)) + lots
  partial_note = "（含 TP1 分批）" if pos.get("realized_partial") else ""
  acc["closed_trades_today"].append(
      f"• [{now_str}] <b>{label}</b> ({total_lots}口) {tag}{partial_note} {_pos_attr(pos, p_fmt)}："
      f"<code>{entry_p:{p_fmt}} ➔ {cur_p:{p_fmt}}</code> (損益 <b><code>{total_net:+,} 元</code></b>)"
  )
  state["positions"][asset][slot_name] = None
  return total_net


def _daytrade_close_card(title: str, label: str, lots: int, cur_p: float,
                         p_fmt: str, entry_p: float, total_net: int,
                         new_eq: int) -> str:
  return (
      f"{title}\n"
      f"📌 <b>商品</b>：<b>{label} (共 {lots} 口)</b>\n"
      "━━━━━━━━━━━━━━━━━━\n"
      f"👉 <b>現在立刻平倉價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(進場成本 {entry_p:{p_fmt}} 元)</i>\n"
      f"💰 <b>本筆平倉損益</b>：<b><code>{total_net:+,} 元</code></b>\n"
      f"🏦 <b>平倉後帳戶總金額</b>：<b><code>{new_eq:,} 元</code></b>"
  )


def _parse_hm(now_str: str) -> int:
  try:
    parts = now_str.split(":")
    return int(parts[0]) * 100 + int(parts[1])
  except Exception:
    return -1


def evaluate_urgent_exits(
    state: dict,
    srf_price: float,
    qff_price: float,
    roll_info: dict,
    now_str: str,
) -> list:
  """出場引擎（2026-10-27 重構版）。

  出場優先序（同一價位多條件同時觸發時）：硬停損 > 當沖收盤強制出場 >
  波段 TP2 全出 > TP1 分批＋移動停利（波段出一半＋剩餘移保本；當沖 TP1 全出）。
  開盤緩衝：開盤後 5 分鐘內（日盤 08:45-08:50、夜盤 17:25-17:30）不觸發硬停損，
  TP／強制出場不受影響。對鎖欄位出場規則與波段相同。
  """
  urgent_cards = []
  acc = state["account"]
  hm = _parse_hm(now_str)
  open_buffer = any(s <= hm < e for s, e in OPEN_BUFFER_WINDOWS)
  if 845 <= hm <= 1345:
    cur_session = "day"
  elif hm >= 1725 or hm <= 500:
    cur_session = "night"
  else:
    cur_session = None

  for asset, cur_p, unit_mult, p_fmt, label in [
      ("SRF", srf_price, 1000, ".2f", f"0050小期 (SRF {roll_info['target_label']})"),
      ("QFF", qff_price, 100, ".0f", f"小台積期 (QFF {roll_info['target_label']})"),
  ]:
    if cur_p <= 0:
      continue
    slots = state["positions"].get(asset) or {}
    if isinstance(slots, dict) and slots.get("dir"):
      slots = migrate_positions(state["positions"]).get(asset, {})
      state["positions"][asset] = slots
    for slot_name in ["swing", "daytrade", "hedge"]:
      pos = slots.get(slot_name) if isinstance(slots, dict) else None
      if not pos:
        continue
      pos_type = pos.get("pos_type", slot_name)
      direction = pos.get("dir", "BUY")
      lots = int(pos.get("lots", 1))
      entry_p = safe_float(pos.get("entry"), cur_p)
      sl_p = safe_float(pos.get("sl"), cur_p)
      tp1_p = safe_float(pos.get("tp1"), cur_p)
      tp2_p = safe_float(pos.get("tp2"), cur_p)
      tp1_hit = pos.get("tp1_hit", False)

      stop_hit = (direction == "BUY" and cur_p <= sl_p) or (
          direction == "SELL" and cur_p >= sl_p
      )
      tp2_hit = (direction == "BUY" and cur_p >= tp2_p) or (
          direction == "SELL" and cur_p <= tp2_p
      )
      tp1_hit_now = (not tp1_hit) and (
          (direction == "BUY" and cur_p >= tp1_p)
          or (direction == "SELL" and cur_p <= tp1_p)
      )

      # ---- 優先序 1：硬停損（開盤緩衝時段內不觸發） ----
      if stop_hit and not open_buffer:
        tag_str = "鎖利出場" if tp1_hit else "停損出場"
        total_net = _book_full_close(
            state, asset, slot_name, pos, cur_p, unit_mult, p_fmt,
            label, tag_str, now_str,
        )
        new_eq = acc["initial_capital"] + acc["realized_pnl"]
        if tp1_hit:
          urgent_cards.append(
              "🛡️ <b>【馬上平倉！觸及鎖利防守線出場！】</b> 🛡️\n"
              f"📌 <b>商品</b>：<b>{label} (共 {lots} 口)</b>\n"
              "━━━━━━━━━━━━━━━━━━\n"
              f"👉 <b>現在立刻平倉價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b>\n"
              f"💰 <b>本筆平倉損益</b>：<b><code>{total_net:+,} 元</code></b>\n"
              f"🏦 <b>平倉後帳戶總金額</b>：<b><code>{new_eq:,} 元</code></b>"
          )
        else:
          urgent_cards.append(
              "🚨🛑🚨 <b>【立即停損出場！馬上平倉！】</b> 🚨🛑🚨\n"
              f"📌 <b>商品</b>：<b>{label} (共 {lots} 口)</b>\n"
              "━━━━━━━━━━━━━━━━━━\n"
              f"👉 <b>現在立刻停損價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(已觸及停損線 {sl_p:{p_fmt}} 元)</i>\n"
              f"❌ <b>本筆平倉損益</b>：<b><code>{total_net:+,} 元</code></b> <i>(原進場 {entry_p:{p_fmt}} 元)</i>\n"
              f"🏦 <b>平倉後帳戶總金額</b>：<b><code>{new_eq:,} 元</code></b>\n"
              "━━━━━━━━━━━━━━━━━━\n"
              "⚠️ <b>請第一時間執行停損平倉，嚴格保護 10 萬保證金！</b>"
          )
        continue

      # ---- 優先序 2：當沖收盤強制出場（不論盈虧） ----
      forced = False
      if pos_type == "daytrade":
        sess_tag = pos.get("session")
        if sess_tag == "day":
          forced = (1325 <= hm <= 1345) or (cur_session == "night")
        elif sess_tag == "night":
          forced = (455 <= hm <= 500) or (cur_session == "day")
        else:
          forced = (1325 <= hm <= 1345) or (455 <= hm <= 500)
      if forced:
        total_net = _book_full_close(
            state, asset, slot_name, pos, cur_p, unit_mult, p_fmt,
            label, "當沖強制出場", now_str,
        )
        new_eq = acc["initial_capital"] + acc["realized_pnl"]
        urgent_cards.append(_daytrade_close_card(
            "⏰ <b>【當沖收盤強制出場】</b> ⏰", label, lots, cur_p,
            p_fmt, entry_p, total_net, new_eq,
        ))
        continue

      # ---- 優先序 3：波段／對鎖 TP2 全出 ----
      if tp2_hit:
        total_net = _book_full_close(
            state, asset, slot_name, pos, cur_p, unit_mult, p_fmt,
            label, "目標停利", now_str,
        )
        new_eq = acc["initial_capital"] + acc["realized_pnl"]
        urgent_cards.append(
            "🎉💰🎉 <b>【馬上停利！全數平倉出場！】</b> 🎉💰🎉\n"
            f"📌 <b>商品</b>：<b>{label} (共 {lots} 口)</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"👉 <b>現在立刻平倉價</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(進場成本 {entry_p:{p_fmt}} 元)</i>\n"
            f"💰 <b>本筆平倉實拿獲利</b>：<b><code>{total_net:+,} 元</code></b> <i>(已扣稅費)</i>\n"
            f"🏦 <b>平倉後帳戶總金額</b>：<b><code>{new_eq:,} 元</code></b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "✅ <b>已達波段最終目標價，請立即全數獲利了結！</b>"
        )
        continue

      # ---- 優先序 4：TP1（波段分批＋移保本；當沖全出） ----
      if tp1_hit_now:
        if pos_type == "daytrade":
          total_net = _book_full_close(
              state, asset, slot_name, pos, cur_p, unit_mult, p_fmt,
              label, "當沖TP1停利", now_str,
          )
          new_eq = acc["initial_capital"] + acc["realized_pnl"]
          urgent_cards.append(_daytrade_close_card(
              "💰 <b>【當沖 TP1 停利全出】</b> 💰", label, lots, cur_p,
              p_fmt, entry_p, total_net, new_eq,
          ))
          continue
        close_n = lots // 2
        if close_n > 0:
          if direction == "BUY":
            partial_pnl = int(round((tp1_p - entry_p) * unit_mult * close_n)) - 50 * close_n
          else:
            partial_pnl = int(round((entry_p - tp1_p) * unit_mult * close_n)) - 50 * close_n
          acc["realized_pnl"] += partial_pnl
          acc["today_realized_pnl"] += partial_pnl
          pos["realized_partial"] = int(pos.get("realized_partial", 0)) + partial_pnl
          pos["closed_lots"] = int(pos.get("closed_lots", 0)) + close_n
          pos["lots"] = lots - close_n
          acc["closed_trades_today"].append(
              f"• [{now_str}] <b>{label}</b> ({close_n}口) TP1分批出場 {_pos_attr(pos, p_fmt)}："
              f"<code>{entry_p:{p_fmt}} ➔ {tp1_p:{p_fmt}}</code> (損益 <b><code>{partial_pnl:+,} 元</code></b>，剩餘 {pos['lots']} 口停損移保本)"
          )
        pos["tp1_hit"] = True
        pos["sl"] = entry_p
        remain_lots = int(pos.get("lots", lots))
        if direction == "BUY":
          float_pnl = int(round((cur_p - entry_p) * unit_mult * remain_lots))
        else:
          float_pnl = int(round((entry_p - cur_p) * unit_mult * remain_lots))
        urgent_cards.append(
            "💰💰💰 <b>【馬上停利減碼！並將防守線上移保本！】</b> 💰💰💰\n"
            f"📌 <b>商品</b>：<b>{label} (持倉 {remain_lots} 口)</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"👉 <b>現在達標價位</b>：<b><code>{cur_p:{p_fmt}} 元</code></b> <i>(浮盈 {float_pnl:+,} 元)</i>\n"
            f"🛡️ <b>最新鎖利停損價</b>：<b><code>改設 {entry_p:{p_fmt}} 元 (穩賺不賠)</code></b>\n"
            f"🎯 <b>主力倉大目標價</b>：<b><code>{tp2_p:{p_fmt}} 元</code></b>"
        )
        continue

  return urgent_cards


# ================= 7. 單次全市場掃描與 5m/15m 雙時框實時決策引擎 =================
def run_single_scan(
    state: dict, force_notify: bool = False, iter_idx: int = 0
) -> bool:
  now_tw = datetime.now(TW_TZ)
  now_str = now_tw.strftime("%Y-%m-%d %H:%M:%S")
  today_str = now_tw.strftime("%Y-%m-%d")
  hour_key = now_tw.strftime("%Y-%m-%d_%H")
  hm = now_tw.hour * 100 + now_tw.minute
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

  # 只有真正從期交所 MIS 抓到實時成交/買賣掛單 (has_live=True) 且處於交易時段，才允許觸發訊號與寫入 K 線！
  srf_live_ready = sess["can_signal"] and srf_data.get("has_live", False)
  qff_live_ready = sess["can_signal"] and qff_data.get("has_live", False)

  flags = state.setdefault("daily_flags", {})
  heartbeat_title = None

  if iter_idx == 0 and (
      force_notify
      or (now_ts - safe_float(flags.get("last_startup_ping_ts"), 0)) >= 2700
  ):
    flags["last_startup_ping_ts"] = now_ts
    force_notify = True
    heartbeat_title = "🟢 <b>【常駐實時監控引擎已啟動！期交所MIS直連備戰中】</b>"

  if 1350 <= hm <= 1425 and flags.get("report_1400_date") != today_str:
    flags["report_1400_date"] = today_str
    report_msg = format_1400_daily_report(
        state, srf_price, qff_price, roll_info, now_str
    )
    send_tg_message(report_msg)

  if 810 <= hm < 845 and flags.get("day_standby_date") != today_str:
    flags["day_standby_date"] = today_str
    force_notify = True
    heartbeat_title = "🚀 <b>【08:15 早盤開機備戰完成｜10萬保證金待命中】</b>"
  elif 845 <= hm < 900 and flags.get("day_open_0845_date") != today_str:
    flags["day_open_0845_date"] = today_str
    force_notify = True
    heartbeat_title = "🔔 <b>【08:45 期貨早盤正式開盤！實時訊號掃描啟動】</b>"
  elif 900 <= hm <= 920 and flags.get("spot_open_0900_date") != today_str:
    flags["spot_open_0900_date"] = today_str
    force_notify = True
    heartbeat_title = "🔥 <b>【09:00 現貨開盤共振！基差對齊已完成】</b>"
  elif 1425 <= hm < 1500 and flags.get("night_standby_date") != today_str:
    flags["night_standby_date"] = today_str
    force_notify = True
    heartbeat_title = "🚀 <b>【14:30 夜盤開機備戰完成｜台指15:00/股期17:25開盤】</b>"
  elif 1500 <= hm <= 1520 and flags.get("night_open_1500_date") != today_str:
    flags["night_open_1500_date"] = today_str
    force_notify = True
    heartbeat_title = "🌙 <b>【15:00 台指夜盤開盤！小0050/小台積 17:25 開盤待命】</b>"
  elif 1725 <= hm <= 1745 and flags.get("night_srf_open_1725_date") != today_str:
    flags["night_srf_open_1725_date"] = today_str
    force_notify = True
    heartbeat_title = "🔥 <b>【17:25 小0050與小台積夜盤正式開盤！實時訊號全開】</b>"
  elif (
      sess["can_signal"]
      and (0 <= now_tw.minute <= 5)
      and flags.get("last_hourly_heartbeat") != hour_key
  ):
    flags["last_hourly_heartbeat"] = hour_key
    force_notify = True
    heartbeat_title = f"📡 <b>【{now_tw.hour:02d}:00 整點實時盯盤回報｜期交所MIS連線正常】</b>"

  if srf_live_ready:
    state["intraday_bars"].setdefault("SRF", []).append(
        {"ts": now_ts, "p": srf_price}
    )
  if qff_live_ready:
    state["intraday_bars"].setdefault("QFF", []).append(
        {"ts": now_ts, "p": qff_price}
    )
  if txf_data.get("has_live", False):
    state["intraday_bars"].setdefault("TXF", []).append(
        {"ts": now_ts, "p": tx_price}
    )

  # 以期交所平盤參考價 (例如 106.85) 作為 K 線平移錨點，確保盤中漲跌幅度與看盤軟體完全一致
  srf_ref_anchor = safe_float(srf_data.get("ref"), srf_price)
  qff_ref_anchor = safe_float(qff_data.get("ref"), qff_price)

  df_15m = build_realtime_df(
      "0050.TW",
      "15m",
      "30d",
      srf_price,
      state["intraday_bars"].get("SRF") or [{"ts": now_ts, "p": srf_ref_anchor}],
  )
  df_5m = build_realtime_df(
      "0050.TW",
      "5m",
      "5d",
      srf_price,
      state["intraday_bars"].get("SRF") or [{"ts": now_ts, "p": srf_ref_anchor}],
  )
  tw2330_15m = build_realtime_df(
      "2330.TW",
      "15m",
      "30d",
      qff_price,
      state["intraday_bars"].get("QFF") or [{"ts": now_ts, "p": qff_ref_anchor}],
  )
  tw2330_5m = build_realtime_df(
      "2330.TW",
      "5m",
      "5d",
      qff_price,
      state["intraday_bars"].get("QFF") or [{"ts": now_ts, "p": qff_ref_anchor}],
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
  if df_5m.empty:
    df_5m = df_15m
  if tw2330_5m.empty:
    tw2330_5m = tw2330_15m

  if tx_daily.empty:
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

  nq_bullish = (nq_price >= nq_ma20 or nq_macd >= 0) and nq_chg_pct >= -0.35
  nq_bearish = (nq_price < nq_ma20 and nq_macd < 0) or nq_chg_pct <= -0.45

  tx_bullish = tx_price >= tx_chip_poc or (
      tx_price >= P and tx_price >= tx_ma20
  )
  tx_bearish = tx_price < tx_chip_poc and tx_price < tx_ma20

  m15, m15_p = df_15m.iloc[-1], df_15m.iloc[-2]
  m5, m5_p = df_5m.iloc[-1], df_5m.iloc[-2]
  t_m15, t_m15_p = tw2330_15m.iloc[-1], tw2330_15m.iloc[-2]
  t_m5, t_m5_p = tw2330_5m.iloc[-1], tw2330_5m.iloc[-2]

  opt = state.get("optimized_params", {})
  srf_atr = max(safe_float(m15["ATR"], 0.40), 0.35)
  qff_atr = max(safe_float(t_m15["ATR"], 14.0), 12.0)

  srf_sl_dist = max(round((srf_atr * 1.25) * 20) / 20.0, 0.50)
  srf_tp1_dist = max(round((srf_atr * SWING_TP1_MULT) * 20) / 20.0, 0.45)
  srf_tp2_dist = max(round((srf_atr * 3.50) * 20) / 20.0, 1.50)

  qff_sl_dist = max(round((qff_atr * 1.25) / 5) * 5, 20.0)
  qff_tp1_dist = max(round((qff_atr * SWING_TP1_MULT) / 5) * 5, 20.0)
  qff_tp2_dist = max(round((qff_atr * 3.50) / 5) * 5, 55.0)

  # 當沖欄位距離（目前無當沖進場訊號，先備妥供未來接線用）
  srf_dt_sl_dist = max(round((srf_atr * DAYTRADE_SL_MULT) * 20) / 20.0, 0.50)
  srf_dt_tp1_dist = max(round((srf_atr * DAYTRADE_TP1_MULT) * 20) / 20.0, 0.45)
  srf_dt_tp2_dist = max(round((srf_atr * DAYTRADE_TP2_MULT) * 20) / 20.0, 1.50)
  qff_dt_sl_dist = max(round((qff_atr * DAYTRADE_SL_MULT) / 5) * 5, 20.0)
  qff_dt_tp1_dist = max(round((qff_atr * DAYTRADE_TP1_MULT) / 5) * 5, 20.0)
  qff_dt_tp2_dist = max(round((qff_atr * DAYTRADE_TP2_MULT) / 5) * 5, 55.0)

  srf_chip = round(safe_float(m15["Chip_POC"], srf_price), 2)
  qff_chip = round(safe_float(t_m15["Chip_POC"], qff_price))

  cards_to_send = []

  if srf_live_ready or qff_live_ready:
    urgent_exits = evaluate_urgent_exits(
        state,
        srf_price if srf_live_ready else 0.0,
        qff_price if qff_live_ready else 0.0,
        roll_info,
        now_tw.strftime("%H:%M:%S"),
    )
    cards_to_send.extend(urgent_exits)

  def fire_signal(
      sig_key: str,
      opt_key: str,
      mode_type: str,
      asset: str,
      target_name: str,
      lots: int,
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
      pos_type: str = "swing",
      priority=None,
  ) -> bool:
    """開倉（2026-10-27 重構版）。回傳 True = 本輪成功開倉或對鎖觸發出場。

    欄位規則：同商品同類型同時最多一個部位；不同類型可並存；全帳戶方向鎖已廢除。
    對鎖觸發時若已有反向波段部位，視為波段出場訊號：波段部位以現價平倉入帳，
    不開對鎖欄位（對鎖手動單仍照舊以警示卡通知使用者在另一月份執行）。
    """
    if (now_ts - state["sent_signals"].get(sig_key, 0)) < SIGNAL_COOLDOWN_SEC:
      return False
    cfg = opt.get(opt_key, {})
    pos_dir = "BUY" if mode_type == "BUY" else "SELL"
    slots = state["positions"].get(asset) or {}
    if isinstance(slots, dict) and slots.get("dir"):
      slots = migrate_positions(state["positions"]).get(asset, {})
      state["positions"][asset] = slots
    session_tag = "day" if sess.get("mode") == "DAY" else "night"

    wr = cfg.get("win_rate", default_wr)
    rr = max(cfg.get("rr", default_rr), 2.15)
    strat_label = cfg.get("algo_label", default_strat_name)

    if mode_type == "HEDGE_LOCK":
      swing_pos = slots.get("swing")
      if swing_pos and swing_pos.get("dir") != pos_dir:
        _book_full_close(
            state, asset, "swing", swing_pos, entry_p, unit_mult, p_fmt,
            f"{target_name} ({roll_info['target_label']})", "對鎖觸發出場",
            now_tw.strftime("%H:%M:%S"),
        )
        state["sent_signals"][sig_key] = now_ts
        acc_summary = compute_account_summary(state, srf_price, qff_price)
        card = format_intuitive_entry_card(
            mode_type,
            target_name,
            roll_info["target_label"],
            lots,
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
            acc_summary["total_equity"],
        )
        cards_to_send.append(card)
        return True
      if slots.get("hedge") is not None:
        return False
      slot_name = "hedge"
    else:
      if slots.get(pos_type) is not None:
        return False
      slot_name = pos_type

    state["sent_signals"][sig_key] = now_ts
    state["positions"][asset][slot_name] = {
        "dir": pos_dir,
        "lots": lots,
        "entry": entry_p,
        "sl": sl_p,
        "tp1": tp1_p,
        "tp2": tp2_p,
        "ts": now_ts,
        "strategy": strat_label,
        "tp1_hit": False,
        "pos_type": slot_name,
        "priority": priority,
        "session": session_tag,
        "sl_dist": abs(entry_p - sl_p),
        "tp1_dist": abs(tp1_p - entry_p),
        "tp2_dist": abs(tp2_p - entry_p),
        "closed_lots": 0,
        "realized_partial": 0,
    }
    acc_summary = compute_account_summary(state, srf_price, qff_price)
    card = format_intuitive_entry_card(
        mode_type,
        target_name,
        roll_info["target_label"],
        lots,
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
        acc_summary["total_equity"],
    )
    cards_to_send.append(card)
    return True

  mkt_ctx_bull = f"小那({nq_price:.0f}/{nq_chg_pct:+.2f}%)｜台指({tx_price:.0f})守穩籌碼重心({tx_chip_poc})與樞紐({P})"
  mkt_ctx_bear = f"小那({nq_price:.0f}/{nq_chg_pct:+.2f}%)｜台指({tx_price:.0f})跌破籌碼重心({tx_chip_poc})或極限乖離衰竭"

  # ---------------- 🎯【0050 小股期 (SRF) — 必須有期交所真實報價 srf_live_ready 才觸發】 ----------------
  # 波段進場優先序（2026-10-27 重構，明表）：S3 → S1 → S2 → S4；對鎖獨立於波段欄位，
  # 僅在波段未進場時評估（沿用舊 elif 鏈「波段優先於對鎖」的 precedence）。
  if srf_live_ready:
    prev_srf_mid = (
      safe_float(m15_p["Open"]) + safe_float(m15_p["Close"])
    ) / 2.0
    cond_srf_s4 = (
        srf_price >= safe_float(m15["MA120"]) * 0.985
        and (
            safe_float(m15_p["RSI2"]) <= 22
            or safe_float(m5_p["RSI2"]) <= 15
            or m15_p["Low"] <= m15_p["BB_Lower"] * 1.003
        )
        and (
            srf_price >= prev_srf_mid
            or srf_price >= safe_float(m5["MA5"])
        )
        and safe_float(m5["MACD_Hist"]) >= safe_float(m5_p["MACD_Hist"])
        and not nq_bearish
    )
    cond_srf_s1 = (
        tx_bullish
        and nq_bullish
        and srf_price >= safe_float(m15["MA60"])
        and (
            m15_p["Low"] <= srf_chip * 1.004
            or m5_p["Low"] <= srf_chip * 1.004
        )
        and srf_price >= srf_chip
        and (
            safe_float(m15["K"]) >= safe_float(m15["D"])
            or safe_float(m5["K"]) >= safe_float(m5["D"])
        )
    )
    cond_srf_s2 = (
        safe_float(m15_p["VCP_Ratio"]) <= 0.96
        and srf_price > safe_float(m15["MA10"])
        and srf_price > srf_chip
        and (
            srf_price >= safe_float(m15_p["High"])
            or srf_price >= safe_float(m5_p["High"])
        )
        and 30 <= safe_float(m15["J"]) <= 84
    )
    cond_srf_s3 = (
        tx_bullish
        and nq_bullish
        and srf_price >= srf_chip
        and srf_price >= safe_float(m15["MA10"])
        and (
            safe_float(m15["K"]) > safe_float(m15["D"])
            or safe_float(m5["K"]) > safe_float(m5["D"])
        )
        and 25 <= safe_float(m15["J"]) <= 78
        and safe_float(m15["MACD_Hist"]) >= 0
    )
    cond_srf_hedge = (
        (
            (
                m15_p["High"] >= m15_p["BB_Upper"] * 0.999
                or safe_float(m15_p["RSI2"]) >= 88
            )
            and srf_price < safe_float(m15["MA5"])
        )
        or (
            tx_bearish
            and nq_bearish
            and srf_price < srf_chip
            and srf_price < safe_float(m15["MA20"])
            and safe_float(m15["K"]) < safe_float(m15["D"])
        )
    ) and safe_float(m15["MACD_Hist"]) < safe_float(m15_p["MACD_Hist"])
    srf_swing_cands = [
        ("SRF_S3_BUY", "SRF_S3", "BUY", "日夜盤籌碼順勢共振多", 66.7, 3.99, cond_srf_s3, "P1"),
        ("SRF_S1_BUY", "SRF_S1", "BUY", "外資VWAP籌碼首回測確認", 75.0, 2.35, cond_srf_s1, "P2"),
        ("SRF_S2_BUY", "SRF_S2", "BUY", "Minervini VCP口袋支點", 70.0, 3.27, cond_srf_s2, "P3"),
        ("SRF_S4_BUY", "SRF_S4", "BUY", "布林下軌W底雙確認 (賺賠比升級)", 66.5, 2.35, cond_srf_s4, "P4"),
    ]
    fired_srf_swing = False
    for sig_key, opt_key, mode_type, strat_name, wr, rr, cond, pri in srf_swing_cands:
      if cond and fire_signal(
          sig_key, opt_key, mode_type, "SRF", "0050小股期 (SRF)", 2,
          strat_name, wr, rr, srf_price,
          round(srf_price - srf_sl_dist, 2),
          round(srf_price + srf_tp1_dist, 2),
          round(srf_price + srf_tp2_dist, 2),
          1000, ".2f", mkt_ctx_bull, pos_type="swing", priority=pri,
      ):
        fired_srf_swing = True
        break
    if not fired_srf_swing and cond_srf_hedge:
      fire_signal(
          "SRF_HEDGE_SELL", "SRF_HEDGE", "HEDGE_LOCK", "SRF",
          "0050小股期 (SRF)", 2, "極限乖離衰竭+破籌碼對鎖", 61.1, 2.08,
          srf_price,
          round(srf_price + srf_sl_dist, 2),
          round(srf_price - srf_tp1_dist, 2),
          round(srf_price - srf_tp2_dist, 2),
          1000, ".2f", mkt_ctx_bear, pos_type="hedge", priority="H",
      )

  # ---------------- 👑【小台積電期貨 (QFF) — 必須有期交所真實報價 qff_live_ready 才觸發】 ----------------
  # 波段進場優先序（2026-10-27 重構，明表）：T2 → T1；對鎖規則同 SRF。
  if qff_live_ready:
    prev_qff_mid = (
      safe_float(t_m15_p["Open"]) + safe_float(t_m15_p["Close"])
    ) / 2.0
    cond_qff_t1 = (
        qff_price >= safe_float(t_m15["MA120"]) * 0.985
        and (
            safe_float(t_m15_p["RSI2"]) <= 22
            or safe_float(t_m5_p["RSI2"]) <= 15
            or t_m15_p["Low"] <= t_m15_p["BB_Lower"] * 1.003
        )
        and (
            qff_price >= prev_qff_mid
            or qff_price >= safe_float(t_m5["MA5"])
        )
        and safe_float(t_m5["MACD_Hist"]) >= safe_float(t_m5_p["MACD_Hist"])
        and adr_premium_pct >= 5.0
    )
    cond_qff_t2 = (
        tx_bullish
        and nq_bullish
        and qff_price >= qff_chip
        and qff_price >= safe_float(t_m15["MA10"])
        and (
            safe_float(t_m15["K"]) > safe_float(t_m15["D"])
            or safe_float(t_m5["K"]) > safe_float(t_m5["D"])
        )
        and 25 <= safe_float(t_m15["J"]) <= 78
        and safe_float(t_m15["MACD_Hist"]) >= 0
    )
    cond_qff_hedge = (
        (
            (
                t_m15_p["High"] >= t_m15_p["BB_Upper"] * 0.999
                or safe_float(t_m15_p["RSI2"]) >= 86
                or safe_float(t_m15_p["J"]) >= 86
            )
            and qff_price < safe_float(t_m15["MA5"])
        )
        or (
            t_m15_p["Close"] >= qff_chip
            and qff_price < qff_chip
            and qff_price < safe_float(t_m15["MA20"])
            and safe_float(t_m15["K"]) < safe_float(t_m15["D"])
        )
    ) and safe_float(t_m15["MACD_Hist"]) < safe_float(t_m15_p["MACD_Hist"])
    qff_swing_cands = [
        ("QFF_T2_BUY", "QFF_T2", "BUY", "日夜盤籌碼順勢共振多", 83.3, 4.50, cond_qff_t2, "P1"),
        ("QFF_T1_BUY", "QFF_T1", "BUY", "Connors RSI(2)極限收腳 (賺賠比升級)", 68.0, 2.25, cond_qff_t1, "P2"),
    ]
    fired_qff_swing = False
    for sig_key, opt_key, mode_type, strat_name, wr, rr, cond, pri in qff_swing_cands:
      if cond and fire_signal(
          sig_key, opt_key, mode_type, "QFF", "小台積電期貨 (QFF)", 1,
          strat_name, wr, rr, qff_price,
          qff_price - qff_sl_dist,
          qff_price + qff_tp1_dist,
          qff_price + qff_tp2_dist,
          100, ".0f", mkt_ctx_bull, pos_type="swing", priority=pri,
      ):
        fired_qff_swing = True
        break
    if not fired_qff_swing and cond_qff_hedge:
      fire_signal(
          "QFF_HEDGE_SELL", "QFF_HEDGE", "HEDGE_LOCK", "QFF",
          "小台積電期貨 (QFF)", 1, "高檔極限衰竭+初破籌碼對鎖", 65.0, 2.20,
          qff_price,
          qff_price + qff_sl_dist,
          qff_price - qff_tp1_dist,
          qff_price - qff_tp2_dist,
          100, ".0f", mkt_ctx_bear, pos_type="hedge", priority="H",
      )

  # ================= 8. 推播輸出 =================
  if cards_to_send:
    send_tg_message("\n\n".join(cards_to_send))
    save_bot_state(state, git_sync=True)
  elif force_notify:
    acc_sum = compute_account_summary(state, srf_price, qff_price)
    pos_header = (
        "\n".join(acc_sum["open_details"])
        if acc_sum["open_details"]
        else "📌 <b>目前持倉狀態</b>：<b>空手無持倉 (0 口)</b>"
    )

    srf_pullback_buy = (
        srf_chip
        if srf_chip < srf_price
        else round(min(safe_float(m15["BB_Lower"]), srf_price - 0.30), 2)
    )
    qff_pullback_buy = (
        qff_chip
        if qff_chip < qff_price
        else round(
            min(safe_float(t_m15["BB_Lower"]), qff_price - 15.0) / 5
        )
        * 5
    )

    title_banner = (
        heartbeat_title
        or "⚪ <b>【當前指令：觀望等待（每25秒常駐監控中）】</b>"
    )

    summary_card = (
        f"{title_banner}\n"
        f"🕒 時間：<code>{now_str}</code>｜狀態：<b>{sess['label']}</b>\n"
        f"🏦 <b>帳戶總金額</b>：<b><code>{acc_sum['total_equity']:,} 元</code></b> (初始保證金 <code>100,000</code>｜今日已平倉 <code>{acc_sum['realized_today']:+,}</code>)\n"
        f"{pos_header}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>0050小期 ({roll_info['target_label']}) 現價：<code>{srf_price:.2f} 元 ({srf_data['chg_pct']:+.2f}%)</code></b>\n"
        f"• 報價來源：<b>{srf_data['source']}</b> (平盤參考價 <code>{srf_data.get('ref', srf_price):.2f}</code>)\n"
        f"• 指標位階：RSI(2) <code>{safe_float(m15['RSI2']):.0f}</code>｜15m J值 <code>{safe_float(m15['J']):.0f}</code>\n"
        f"• 🟢 <b>低接買點</b>：回測支撐 <b><code>{srf_pullback_buy:.2f} 元</code></b>｜🚀 <b>順勢買點</b>：站穩籌碼 <b><code>{srf_chip:.2f} 元</code></b>\n"
        f"  └ 對應停損：<code>{srf_price-srf_sl_dist:.2f}</code>｜第一停利：<code>{srf_price+srf_tp1_dist:.2f}</code>｜目標：<code>{srf_price+srf_tp2_dist:.2f}</code>\n"
        f"• 🛡️ <b>做空避險/對鎖價</b>：跌破支撐 <b><code>{safe_float(m15['Support_Low']):.2f} 元</code></b> 啟動\n\n"
        f"👑 <b>小台積期 ({roll_info['target_label']}) 現價：<code>{qff_price:.0f} 元 ({qff_data['chg_pct']:+.2f}%)</code></b>\n"
        f"• 報價來源：<b>{qff_data['source']}</b> (平盤參考價 <code>{qff_data.get('ref', qff_price):.0f}</code>)\n"
        f"• 指標位階：RSI(2) <code>{safe_float(t_m15['RSI2']):.0f}</code>｜15m J值 <code>{safe_float(t_m15['J']):.0f}</code>\n"
        f"• 🟢 <b>低接買點</b>：回測支撐 <b><code>{qff_pullback_buy:.0f} 元</code></b>｜🚀 <b>順勢買點</b>：站穩籌碼 <b><code>{qff_chip:.0f} 元</code></b>\n"
        f"  └ 對應停損：<code>{qff_price-qff_sl_dist:.0f}</code>｜第一停利：<code>{qff_price+qff_tp1_dist:.0f}</code>｜目標：<code>{qff_price+qff_tp2_dist:.0f}</code>\n"
        f"• 🛡️ <b>做空避險/對鎖價</b>：跌破支撐 <b><code>{safe_float(t_m15['Support_Low']):.0f} 元</code></b> 啟動\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🧭 <b>小那(NQ)</b>：<code>{nq_price:.0f} ({nq_chg_pct:+.2f}%)</code>｜<b>TSM溢價</b>：<code>{adr_premium_pct:+.1f}%</code>\n"
        f"📊 <b>台指({tx_price:.0f})位階</b>：支撐 <code>{S1}</code> / 樞紐 <code>{P}</code> / 籌碼重心 <code>{tx_chip_poc}</code> / 壓力 <code>{R1}</code>"
    )
    send_tg_message(summary_card)
    save_bot_state(state, git_sync=False)
  else:
    print(
        f"[{now_str}] 第 {iter_idx+1}/{MAX_ITERATIONS} 輪"
        f" ({sess['label']}) | SRF={srf_price:.2f}({srf_data['source']}) |"
        f" QFF={qff_price:.0f}({qff_data['source']}) | TX={tx_price:.0f}"
    )

  state_after = json.dumps(state, sort_keys=True)
  return state_Before != state_after


def main():
  state = load_bot_state()
  state_changed = False
  # 時間預算: 單輪掃描含 ~20 次外部請求, 實測遠超過 POLL_INTERVAL_SEC;
  # 115 輪絕對超過 workflow 的 60 分鐘 timeout 而被硬殺 (finally 未必跑完)。
  # 以牆鐘預算主動收尾, 確保狀態正常存檔後再結束, 讓接力棒乾淨交棒。
  MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "3000"))
  start_ts = time.time()
  try:
    for i in range(MAX_ITERATIONS):
      if i > 0 and (time.time() - start_ts) >= MAX_RUNTIME_SEC:
        print(f"⏱ 已達 {MAX_RUNTIME_SEC} 秒時間預算, 主動結束常駐迴圈。")
        break
      now_tw = datetime.now(TW_TZ)
      sess = get_session_status(now_tw)
      if sess["can_run"] or (i == 0 and FORCE_REPORT):
        try:
          is_force = FORCE_REPORT and (i == 0)
          if run_single_scan(state, force_notify=is_force, iter_idx=i):
            state_changed = True
            save_bot_state(state, git_sync=False)
        except Exception as e:
          print(f"單輪掃描例外: {e}")
          traceback.print_exc()
      else:
        print(f"[{now_tw.strftime('%H:%M:%S')}] {sess['label']}，結束常駐迴圈。")
        break

      if not sess["can_run"]:
        break
      if i < MAX_ITERATIONS - 1:
        time.sleep(POLL_INTERVAL_SEC)
  finally:
    if state_changed:
      save_bot_state(state, git_sync=True)


if __name__ == "__main__":
  main()
