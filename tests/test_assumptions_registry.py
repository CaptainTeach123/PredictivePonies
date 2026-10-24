"""Keeps ASSUMPTIONS.md and the code in sync: every [A-xx] tag must be in both."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = re.compile(r"\[A-(\d{2})\]")


def test_registry_matches_code():
    code_ids = set()
    for path in list((ROOT / "ponies").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        code_ids |= set(TAG.findall(path.read_text()))
    registry_ids = set(re.findall(r"^\|\s*A-(\d{2})\s*\|", (ROOT / "ASSUMPTIONS.md").read_text(), re.M))
    assert code_ids - registry_ids == set(), "IDs used in code but missing from ASSUMPTIONS.md"
    assert registry_ids - code_ids == set(), "IDs in ASSUMPTIONS.md not referenced in code"
