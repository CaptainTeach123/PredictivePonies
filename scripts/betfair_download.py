"""Download purchased Betfair historical files (historicdata.betfair.com).

    python scripts/betfair_download.py --from 2025-01 --to 2025-12 --countries AU --dry-run
    python scripts/betfair_download.py --from 2025-01 --to 2025-12 --countries AU

Credentials come from environment variables only (never from the command line
or files):
  BETFAIR_SSOID                 a session token, or else
  BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY   to log in
  BETFAIR_SSO_URL               (optional) login host; Australian accounts may
                                need https://identitysso.betfair.com.au/api/login

Files must first be "purchased" (the Basic plan is free) on the website; the
API only downloads what is in My Data. Saved to data/raw/betfair/ (git-ignored:
licensed data). Prints counts only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "raw" / "betfair"
API = "https://historicdata.betfair.com/api/"


def login() -> str:
    if os.environ.get("BETFAIR_SSOID"):
        return os.environ["BETFAIR_SSOID"]
    user, pw, key = (os.environ.get(k) for k in ("BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"))
    if not (user and pw and key):
        sys.exit("set BETFAIR_SSOID, or BETFAIR_USERNAME + BETFAIR_PASSWORD + BETFAIR_APP_KEY")
    url = os.environ.get("BETFAIR_SSO_URL") or "https://identitysso.betfair.com/api/login"
    req = urllib.request.Request(url, data=urllib.parse.urlencode({"username": user, "password": pw}).encode(),
                                 headers={"X-Application": key, "Accept": "application/json",
                                          "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    if res.get("status") != "SUCCESS":
        sys.exit(f"login failed: {res.get('error', res.get('status'))}")
    return res["token"]


def post(ssoid: str, op: str, body: dict):
    req = urllib.request.Request(API + op, data=json.dumps(body).encode(),
                                 headers={"ssoid": ssoid, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"null")


def month_filter(a, y1, m1, y2, m2) -> dict:
    import calendar
    return {"sport": "Horse Racing", "plan": a.plan,
            "fromDay": 1, "fromMonth": m1, "fromYear": y1,
            "toDay": calendar.monthrange(y2, m2)[1], "toMonth": m2, "toYear": y2,
            "eventId": None, "eventName": None,
            "marketTypesCollection": a.market_types.split(","),
            "countriesCollection": a.countries.split(","),
            "fileTypeCollection": ["M"]}


def download(ssoid: str, file_path: str) -> bool:
    dest = STORE / file_path.lstrip("/").replace("data/xds/historic/", "")
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = API + "DownloadFile?filePath=" + urllib.parse.quote(file_path, safe="")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"ssoid": ssoid})
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt * 2)
    raise RuntimeError("download failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", required=True, help="YYYY-MM")
    ap.add_argument("--to", dest="end", required=True, help="YYYY-MM")
    ap.add_argument("--countries", default="AU", help="comma-separated, e.g. AU or GB,IE")
    ap.add_argument("--market-types", default="WIN,PLACE")
    ap.add_argument("--plan", default="Basic Plan")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="show purchased data and file count/size only")
    a = ap.parse_args()

    ssoid = login()
    req = urllib.request.Request(API + "GetMyData", headers={"ssoid": ssoid})  # a GET per Betfair's docs
    with urllib.request.urlopen(req, timeout=60) as r:
        mine = json.loads(r.read() or b"[]")
    racing = [m for m in mine if m.get("sport") == "Horse Racing"]
    print(json.dumps({"purchased_horse_racing_months": len(racing),
                      "plans": sorted({m.get("plan") for m in racing})}))
    (y1, m1), (y2, m2) = (map(int, s.split("-")) for s in (a.start, a.end))
    flt = month_filter(a, y1, m1, y2, m2)
    size = post(ssoid, "GetAdvBasketDataSize", flt)
    print(json.dumps({"basket": size}))
    if a.dry_run:
        return 0
    files = post(ssoid, "DownloadListOfFiles", flt) or []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        new = sum(pool.map(lambda f: download(ssoid, f), files))
    print(json.dumps({"files_listed": len(files), "downloaded_new": int(new)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
