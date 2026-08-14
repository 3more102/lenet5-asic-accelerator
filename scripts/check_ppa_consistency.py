#!/usr/bin/env python3
"""Fail if docs/PPA.md's headline table disagrees with the raw STA results.

Every number in docs/PPA.md is transcribed by hand from
asic/sta/results/ppa_summary.csv. That is exactly the kind of coupling that
rots silently: re-run the flow, forget one table cell, and the document now
claims something the tool never produced.

This is the same guard the vectors already have -- CI regenerates them and
fails on any diff -- applied to the PPA numbers. Run it with `make check-ppa`,
or directly:

    python3 scripts/check_ppa_consistency.py

Exit status is 0 when the document matches the CSV, 1 otherwise.
"""

import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "asic" / "sta" / "results" / "ppa_summary.csv"
DOC_PATH = ROOT / "docs" / "PPA.md"

# Cell text -> (csv column, converter from the csv value to the doc's units).
# Each entry says how to turn one raw STA number into the figure the table
# prints, so a mismatch points at a specific column rather than "a number moved".
FIELDS = [
    ("cells", "cells", lambda v: float(v)),
    ("seq_cells", "seq_cells", lambda v: float(v)),
    ("area_um2", "area_um2", lambda v: float(v)),
    ("crit_path_ns", "crit_path_ns", lambda v: float(v)),
    ("fmax_mhz", "crit_path_ns", lambda v: 1000.0 / float(v)),
    ("whs_ns", "whs_ns", lambda v: float(v)),
    ("power_uw", "power_total_w", lambda v: float(v) * 1e6),
]


def parse_number(text):
    """Pull a signed number out of a table cell, tolerating ',' and unicode minus."""
    cleaned = text.replace(",", "").replace("−", "-").replace("µ", "u")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def load_csv():
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["block"]: row for row in csv.DictReader(handle)}


def doc_rows():
    """Yield (block, [cell, ...]) for each headline-table row in the document."""
    for line in DOC_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name = re.match(r"`([A-Za-z0-9_]+)`", cells[0])
        if name and len(cells) >= 9:
            yield name.group(1), cells


def main():
    for path in (CSV_PATH, DOC_PATH):
        if not path.exists():
            print(f"FAIL check_ppa: missing {path.relative_to(ROOT)}")
            return 1

    rows = load_csv()
    problems = []
    checked = 0

    for block, cells in doc_rows():
        if block not in rows:
            problems.append(f"{block}: in docs/PPA.md but not in ppa_summary.csv")
            continue

        # Columns: name | cells | seq | area | min period | fmax | setup | whs | power
        printed = [
            ("cells", cells[1]),
            ("seq_cells", cells[2]),
            ("area_um2", cells[3]),
            ("crit_path_ns", cells[4]),
            ("fmax_mhz", cells[5]),
            ("whs_ns", cells[7]),
            ("power_uw", cells[8]),
        ]

        for (label, column, convert), (_, text) in zip(FIELDS, printed):
            want = convert(rows[block][column])
            got = parse_number(text)
            if got is None:
                problems.append(f"{block}.{label}: no number in cell {text!r}")
                continue
            # The document rounds for readability, so allow anything within half
            # a unit of the last digit it prints. Deliberately a tolerance rather
            # than round()-then-compare: Python rounds halves to even, so a
            # correctly-written 2.1965 -> "2.197" would otherwise be flagged.
            decimals = len(text.split(".")[1].split()[0]) if "." in text else 0
            tolerance = 10 ** -decimals / 2 + 1e-9
            if abs(want - got) > tolerance:
                problems.append(
                    f"{block}.{label}: docs say {got}, ppa_summary.csv gives "
                    f"{want:.{decimals}f}"
                )
            checked += 1

    missing = set(rows) - {block for block, _ in doc_rows()}
    for block in sorted(missing):
        problems.append(f"{block}: in ppa_summary.csv but not in docs/PPA.md")

    if problems:
        print("FAIL check_ppa: docs/PPA.md disagrees with asic/sta/results/")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"PASS check_ppa: {checked} figures across {len(rows)} blocks match the STA results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
