"""Fetch Betfair's public daily Starting Price files (Australian WIN markets).

    python scripts/bsp_fetch.py --from 2026-09-25 --to 2026-10-25 [--probe]

Files: https://promo.betfair.com/betfairsp/prices/dwbfpricesauswin<DDMMYYYY>.csv
Saved under data/raw/formfav/bsp/ (git-ignored with the rest of the store;
load_store ignores that folder). Prints status codes and counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "formfav" / "bsp"
BASE = "https://promo.betfair.com/betfairsp/prices/"


def fetch(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"url": url.rsplit("/", 1)[-1], "error": type(e).__name__}))
        return 0, b""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--kind", default="auswin", help="auswin | ausplace | ukwin | ukplace | irewin ...")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    status, body = fetch(BASE)
    print(json.dumps({"index_status": status, "index_bytes": len(body)}))
    if a.probe:
        d = d1
        url = f"{BASE}dwbfprices{a.kind}{d:%d%m%Y}.csv"
        status, body = fetch(url)
        head = body.decode("utf-8", errors="replace").splitlines()[:1]
        print(json.dumps({"sample_day": d.isoformat(), "status": status, "bytes": len(body),
                          "header": head[0] if head else None}))
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    got = skipped = failed = 0
    d = d0
    while d <= d1:
        dest = OUT / f"{a.kind}_{d.isoformat()}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
        else:
            status, body = fetch(f"{BASE}dwbfprices{a.kind}{d:%d%m%Y}.csv")
            if status == 200 and body.strip():
                dest.write_bytes(body)
                got += 1
            else:
                failed += 1
            time.sleep(1)
        d += timedelta(days=1)
    print(json.dumps({"kind": a.kind, "days": (d1 - d0).days + 1, "fetched": got, "already_had": skipped, "failed": failed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
