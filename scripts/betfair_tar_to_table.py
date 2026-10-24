"""Convert a Betfair historical-data TAR (as downloaded from My Data) into one
compact table, one row per runner per market, without unpacking it.

    python scripts/betfair_tar_to_table.py data.tar --out data/raw/betfair/runners.csv.gz

Reads the archive as a stream, decompresses each market/event file in memory
and keeps the market types asked for (default WIN and PLACE). Prints counts
only. The output is licensed data: keep it in data/raw/betfair/ (git-ignored).
"""

from __future__ import annotations

import argparse
import bz2
import io
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies.betfair import parse_market  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--market-types", default="WIN,PLACE")
    ap.add_argument("--countries", default=None, help="e.g. AU or GB,IE (default: all)")
    a = ap.parse_args()
    keep_types = set(a.market_types.split(","))
    keep_countries = set(a.countries.split(",")) if a.countries else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        a.out.unlink()
    files = bad = 0
    buf, written, header = [], 0, True
    countries, types = {}, {}

    def flush():
        nonlocal buf, written, header
        if buf:
            pd.DataFrame(buf).to_csv(a.out, mode="a", header=header, index=False, compression="gzip")
            written += len(buf)
            header, buf = False, []

    truncated, last = False, None
    tar = tarfile.open(a.tar, mode="r|*")
    members = iter(tar)
    while True:
        try:
            member = next(members)
        except StopIteration:
            break
        except tarfile.ReadError:  # archive cut short: keep everything read so far
            truncated = True
            break
        if not member.isfile() or not member.name.endswith(".bz2"):
            continue
        files += 1
        last = member.name
        try:
            raw = bz2.decompress(tar.extractfile(member).read())
            rows = parse_market(io.StringIO(raw.decode("utf-8", errors="replace")))
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        for r in rows:
            types[r["market_type"]] = types.get(r["market_type"], 0) + 1
            countries[r["country"]] = countries.get(r["country"], 0) + 1
        buf += [r for r in rows if r["market_type"] in keep_types
                and (keep_countries is None or r["country"] in keep_countries)]
        if len(buf) > 200_000:
            flush()
        if files % 20_000 == 0:
            print(json.dumps({"files": files, "rows_kept": written + len(buf)}), flush=True)
    flush()
    top = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1])[:12])
    print(json.dumps({"files": files, "unreadable": bad, "rows_kept": written, "archive_truncated": truncated,
                      "last_file": last,
                      "runner_rows_by_market_type": top(types), "runner_rows_by_country": top(countries)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
