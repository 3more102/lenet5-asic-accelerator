#!/usr/bin/env bash
# Run the OpenROAD Flow Scripts RTL-to-GDS flow on conv5x5_pe and copy the
# sign-off reports back into the repository.
#
# Usage:  bash asic/openroad/run_orfs.sh
#
# ORFS_ROOT may be overridden if your install lives elsewhere. The flow itself
# is not vendored here -- only the configuration and the harvested reports are.

set -euo pipefail

ORFS_ROOT="${ORFS_ROOT:-/root/OpenROAD-flow-scripts}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_HOME="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/results"

if [ ! -d "$ORFS_ROOT/flow" ]; then
    echo "ERROR: no ORFS install at $ORFS_ROOT" >&2
    echo "Set ORFS_ROOT to your OpenROAD-flow-scripts checkout." >&2
    exit 1
fi

for tool in yosys openroad; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool not on PATH" >&2; exit 1; }
done

mkdir -p "$OUT"

echo "== ORFS   : $ORFS_ROOT"
echo "== design : conv5x5_pe on sky130hd"
echo "== config : $HERE/config.mk"
echo

# ORFS defaults its tool paths to a self-built tree under tools/install/, which
# does not exist when OpenROAD/yosys/klayout come from system packages. Passing
# them as make command-line variables overrides the makefile definitions
# unconditionally, which `export VAR=...` in the environment would not.
YOSYS_BIN="$(command -v yosys)"
OPENROAD_BIN="$(command -v openroad)"
KLAYOUT_BIN="$(command -v klayout || true)"

cd "$ORFS_ROOT/flow"
make DESIGN_CONFIG="$HERE/config.mk" \
     YOSYS_EXE="$YOSYS_BIN" \
     OPENROAD_EXE="$OPENROAD_BIN" \
     KLAYOUT_CMD="$KLAYOUT_BIN" \
     2>&1 | tee "$OUT/orfs_run.log"

# ORFS writes to logs/<platform>/<design>/<variant>/ and
# reports/<platform>/<design>/<variant>/. Harvest the sign-off artefacts.
PLAT=sky130hd
DES=conv5x5_pe
for variant in base; do
    RDIR="$ORFS_ROOT/flow/reports/$PLAT/$DES/$variant"
    LDIR="$ORFS_ROOT/flow/logs/$PLAT/$DES/$variant"
    ODIR="$ORFS_ROOT/flow/results/$PLAT/$DES/$variant"
    [ -d "$RDIR" ] && cp -r "$RDIR"/. "$OUT/reports_$variant/" 2>/dev/null || true
    [ -d "$LDIR" ] && mkdir -p "$OUT/logs_$variant" && cp "$LDIR"/*.log "$OUT/logs_$variant/" 2>/dev/null || true
    # The GDS and final netlist are large; keep the netlist, skip the GDS.
    if [ -d "$ODIR" ]; then
        mkdir -p "$OUT/netlist_$variant"
        cp "$ODIR"/*.sdc "$OUT/netlist_$variant/" 2>/dev/null || true
        cp "$ODIR"/6_final.v "$OUT/netlist_$variant/" 2>/dev/null || true
    fi
done

echo
echo "== harvested into $OUT"
