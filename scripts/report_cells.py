#!/usr/bin/env python3
"""Summarize Yosys generic-netlist JSON as a Markdown table.

Used by the CI synthesis job to turn `write_json` output into a job summary.
Cell counts here are *generic* Yosys cells, not standard cells: they track
relative datapath size across commits, and say nothing about the area a real
library would give. Run locally with:

    python3 scripts/report_cells.py results/*_generic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def module_rows(path: Path) -> list[tuple[str, int, int]]:
    with path.open(encoding="utf-8") as handle:
        design = json.load(handle)

    rows = []
    for name, module in design.get("modules", {}).items():
        rows.append(
            (name, len(module.get("cells", {})), len(module.get("netnames", {})))
        )
    return sorted(rows)


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv[1:]]
    if not paths:
        print(f"usage: {argv[0]} <netlist.json> [...]", file=sys.stderr)
        return 2

    print("## Yosys generic synthesis\n")
    print("| Netlist | Module | Generic cells | Nets |")
    print("|---|---|---:|---:|")

    missing = 0
    for path in paths:
        if not path.is_file():
            missing += 1
            print(f"| `{path.name}` | _not generated_ | - | - |")
            continue
        for name, cells, nets in module_rows(path):
            print(f"| `{path.name}` | `{name}` | {cells} | {nets} |")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
