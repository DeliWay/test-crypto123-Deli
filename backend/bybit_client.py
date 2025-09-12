import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)
BASE_URL = "https://api.bybit.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "crypto-trading-platform/merged"})

def _get(path: str, params: Dict[str, Any], timeout: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    url = f"{BASE_URL}{path}"
    try:
        resp = SESSION.get(url, params=params, timeout=timeout)
        code = resp.status_code
        if code != 200:
            logger.warning("Bybit non-200 %s %s", code, resp.text[:300])
            return None, code
        data = resp.json()
        return data, code
    except Exception as e:
        logger.exception("Bybit request error: %s", e)
        return None, None

def ticker(symbol: str) -> Optional[Dict[str, Any]]:
    data, code = _get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if not data or data.get("retCode") != 0:
        return None
    rows = data.get("result", {}).get("list") or []
    if not rows:
        return None
    row = rows[0]
    # Strict fields
    return {
        "symbol": row.get("symbol"),
        "lastPrice": row.get("lastPrice"),
        "price24hPcnt": row.get("price24hPcnt"),
        "highPrice24h": row.get("highPrice24h"),
        "lowPrice24h": row.get("lowPrice24h"),
        "ts": int(time.time()*1000)
    }

def klines(symbol: str, interval: str, limit: int = 200) -> Optional[List[Dict[str, Any]]]:
    data, code = _get("/v5/market/kline", {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit})
    if not data or data.get("retCode") != 0:
        return None
    arr = data.get("result", {}).get("list") or []
    # Normalize: [start, open, high, low, close, volume, turnover] (strings)
    # Convert to dict per bar with numeric values where safe
    out: List[Dict[str, Any]] = []
    for row in arr:
        # bybit returns most-recent last? We'll sort ascending by open time to be safe
        otime = int(row[0])
        o, h, l, c, v = map(float, (row[1], row[2], row[3], row[4], row[5]))
        out.append({"open_time": otime, "open": o, "high": h, "low": l, "close": c, "volume": v})
    out.sort(key=lambda x: x["open_time"])
    return out
