"""Collect FormFav race cards, live odds and results into data/raw/formfav/.

Runs in GitHub Actions (.github/workflows/formfav.yml) with the API key in the
FORMFAV_API_KEY secret. FormFav data is licensed and this repo is public, so:
  * nothing collected is ever committed (data/raw/formfav/ is git-ignored);
    the workflow keeps the store in the Actions cache;
  * this script prints only counts and HTTP status codes, never data or the key.

Transport: FormFav's own MCP server (JSON-RPC over HTTP), calling the same tools
the Claude connector uses: get_meetings, get_race_card, get_prices,
get_race_results. FormFav publishes no public REST docs, so ``--probe`` tries
the likely URL / auth combinations once and reports which one answers.

    python scripts/formfav_collect.py --probe
    python scripts/formfav_collect.py --days-back 29 --days-ahead 1 --max-calls 2500

Store layout (one file per meeting, merged on every run):
    data/raw/formfav/<YYYY-MM-DD>/<track-slug>.json
    {"meeting": {...}, "cards": {"1": {...}}, "prices": {"1": {...}},
     "results": {...}, "fetched": {"card:1": "<utc iso>", ...}}

Point-in-time: FormFav cards for past dates are pre-race snapshots (checked on
2026-10-18 Moe R1), and finished races keep their price history ("movements"),
so a backfill is as good as collecting live, within the 30-day window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "raw" / "formfav"
SYDNEY = ZoneInfo("Australia/Sydney")

# Candidates for --probe. The first that answers is printed; set the
# FORMFAV_MCP_URL / FORMFAV_AUTH repository variables to pin it.
CANDIDATE_URLS = ["https://mcp.formfav.com/mcp", "https://mcp.formfav.com/", "https://api.formfav.com/mcp"]
AUTH_STYLES = ["bearer", "x-api-key", "query"]


class FormFavError(RuntimeError):
    pass


class McpClient:
    """Minimal MCP streamable-HTTP client (initialize + tools/call)."""

    def __init__(self, url: str, key: str, auth: str = "bearer", timeout: int = 60):
        self.url, self.key, self.auth, self.timeout = url, key, auth, timeout
        self.session: str | None = None
        self.calls = 0
        self._id = 0

    def _request(self, payload: dict) -> dict | None:
        url, headers = self.url, {"Content-Type": "application/json",
                                  "Accept": "application/json, text/event-stream"}
        if self.auth == "bearer":
            headers["Authorization"] = f"Bearer {self.key}"
        elif self.auth == "x-api-key":
            headers["x-api-key"] = self.key
        elif self.auth == "query":
            url += ("&" if "?" in url else "?") + "api_key=" + self.key
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.session = resp.headers.get("Mcp-Session-Id") or self.session
            body = resp.read().decode("utf-8", errors="replace")
            if "event-stream" in (resp.headers.get("Content-Type") or ""):
                msgs = [json.loads(l[5:]) for l in body.splitlines() if l.startswith("data:") and l[5:].strip()]
                return next((m for m in msgs if "id" in m), msgs[-1] if msgs else None)
            return json.loads(body) if body.strip() else None

    def _rpc(self, method: str, params: dict) -> dict:
        self._id += 1
        msg = self._request({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        if not msg or "error" in msg:
            raise FormFavError(f"{method}: {(msg or {}).get('error', 'empty response')}")
        return msg["result"]

    def initialize(self) -> None:
        self._rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "predictiveponies-collector", "version": "1"}})
        try:
            self._request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except urllib.error.HTTPError as e:
            if e.code not in (202, 204):
                raise

    def tool(self, name: str, **args) -> dict:
        """Call a FormFav tool; returns its JSON payload's "result" object."""
        for attempt in range(5):
            try:
                self.calls += 1
                res = self._rpc("tools/call", {"name": name, "arguments": args})
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise
        if res.get("isError"):
            raise FormFavError(f"{name}: tool error")
        payload = res.get("structuredContent")
        if payload is None:
            text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
            payload = json.loads(text)
        payload = payload.get("result", payload)
        if isinstance(payload, dict) and "error" in payload and len(payload) == 1:
            raise FormFavError(f"{name}: {payload['error'][:160]}")
        return payload


def probe(key: str) -> int:
    """Try each URL x auth style once; report status only."""
    today = datetime.now(SYDNEY).date().isoformat()
    found = None
    urls = [u for u in [os.environ.get("FORMFAV_MCP_URL")] if u] + CANDIDATE_URLS
    for url in dict.fromkeys(urls):
        for auth in AUTH_STYLES:
            c = McpClient(url, key, auth, timeout=30)
            try:
                c.initialize()
                meetings = c.tool("get_meetings", date=today)["meetings"]
                print(f"OK    {url} auth={auth}: {len(meetings)} meetings today")
                found = found or (url, auth)
            except urllib.error.HTTPError as e:
                print(f"HTTP  {url} auth={auth}: {e.code}")
            except Exception as e:  # noqa: BLE001 - report and keep probing
                print(f"FAIL  {url} auth={auth}: {type(e).__name__}: {str(e)[:120].replace(key, '***')}")
    if found:
        print(f"\nUSE: FORMFAV_MCP_URL={found[0]}  FORMFAV_AUTH={found[1]}")
        return 0
    print("\nNo combination worked. Check the key, or the FormFav account's API docs for the endpoint.")
    return 1


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"cards": {}, "prices": {}, "fetched": {}}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def collect(client: McpClient, days: list[date], countries: set[str], max_calls: int, pause: float) -> dict:
    """Fill the store for ``days``. Re-runs only fetch what's missing or stale:
    results until 'final'; cards/prices of upcoming races are refreshed."""
    stats = {"meetings": 0, "cards": 0, "prices": 0, "results_final": 0, "errors": 0}
    now = datetime.now(timezone.utc)
    today = datetime.now(SYDNEY).date()
    for d in days:
        if client.calls >= max_calls:
            break
        try:
            meetings = client.tool("get_meetings", date=d.isoformat())["meetings"]
        except Exception as e:  # noqa: BLE001
            print(f"{d}: meetings failed ({type(e).__name__})")
            stats["errors"] += 1
            continue
        for m in meetings:
            if m.get("country") not in countries or m.get("abandoned"):
                continue
            path = STORE / d.isoformat() / f"{m['slug']}.json"
            data = _load(path)
            data["meeting"] = {k: m.get(k) for k in ("track", "slug", "country")}
            stats["meetings"] += 1
            past = d < today
            for race in m.get("races", []):
                if client.calls >= max_calls:
                    break
                n = str(race["raceNumber"])
                if race.get("abandoned"):
                    continue
                for kind, tool in (("cards", "get_race_card"), ("prices", "get_prices")):
                    # past races: fetch once; upcoming races: refresh each run
                    if past and n in data[kind]:
                        continue
                    try:
                        data[kind][n] = client.tool(tool, date=d.isoformat(), track=m["slug"], race=int(n))
                        data["fetched"][f"{kind[:-1]}:{n}"] = now.isoformat(timespec="seconds")
                        stats[kind] += 1
                    except Exception as e:  # noqa: BLE001
                        stats["errors"] += 1
                        print(f"{d} {m['slug']} R{n} {tool}: {type(e).__name__}")
                    time.sleep(pause)
            res = data.get("results") or {}
            finished = res.get("races") and all(r.get("resultStatus") in ("final", "abandoned") for r in res["races"])
            if past and not finished and client.calls < max_calls:
                try:
                    data["results"] = client.tool("get_race_results", date=d.isoformat(), track=m["slug"])
                    data["fetched"]["results"] = now.isoformat(timespec="seconds")
                    stats["results_final"] += sum(r.get("resultStatus") == "final"
                                                  for r in data["results"].get("races", []))
                except Exception as e:  # noqa: BLE001
                    stats["errors"] += 1
                    print(f"{d} {m['slug']} results: {type(e).__name__}")
                time.sleep(pause)
            _save(path, data)
    return stats


def store_summary() -> dict:
    files = list(STORE.rglob("*.json"))
    races = final = 0
    for f in files:
        d = json.loads(f.read_text())
        races += len(d.get("cards", {}))
        final += sum(r.get("resultStatus") == "final" for r in (d.get("results") or {}).get("races", []))
    days = sorted({f.parent.name for f in files})
    return {"meetings": len(files), "race_cards": races, "races_with_final_result": final,
            "days": f"{days[0]} .. {days[-1]}" if days else "-"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="find which URL/auth works, then exit")
    ap.add_argument("--days-back", type=int, default=3, help="past days to fill (max 30)")
    ap.add_argument("--days-ahead", type=int, default=1, help="future days (cards + live odds)")
    ap.add_argument("--countries", default="au", help="comma-separated FormFav country codes")
    ap.add_argument("--max-calls", type=int, default=2500, help="stop after this many API calls")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between calls")
    a = ap.parse_args()

    key = os.environ.get("FORMFAV_API_KEY", "").strip()
    if not key:
        print("FORMFAV_API_KEY is not set (add it as a repository secret).")
        return 2
    if a.probe:
        return probe(key)

    client = McpClient(os.environ.get("FORMFAV_MCP_URL") or CANDIDATE_URLS[0], key,
                       os.environ.get("FORMFAV_AUTH") or "bearer")
    client.initialize()
    today = datetime.now(SYDNEY).date()
    # oldest first, so the days about to leave the 30-day window are saved first
    days = [today - timedelta(days=i) for i in range(min(a.days_back, 29), 0, -1)] + \
           [today + timedelta(days=i) for i in range(0, a.days_ahead + 1)]
    stats = collect(client, days, set(a.countries.split(",")), a.max_calls, a.pause)
    print("this run:", json.dumps({**stats, "api_calls": client.calls}))
    print("store:   ", json.dumps(store_summary()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
