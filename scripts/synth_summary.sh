#!/usr/bin/env bash
# Run generic Yosys synthesis for every currently-synthesizable module and
# print just the headline cell counts, so the numbers quoted in
# docs/ARCHITECTURE.md can be reproduced in one screen.
#
#   wsl -e bash scripts/synth_summary.sh
#
# conv5x5_pe's figure is taken from Yosys's "design hierarchy" block because
# that is the one that includes its conv5x5_row_mac and requantize submodules;
# the bare "=== conv5x5_pe ===" block counts only the top level.
set -u

LOG=${LOG:-results/synth_summary.log}
mkdir -p "$(dirname "$LOG")"

if ! make synth >"$LOG" 2>&1; then
    echo "FAILED -- see $LOG"
    tail -20 "$LOG"
    exit 1
fi

# Everything is printed after the run so no tool output interleaves with it.
echo "make synth -- generic synthesis with Yosys $(yosys -V | awk '{print $2}')"

report() {
    local label=$1 header=$2 after=$3
    echo
    echo "$label"
    grep -A "$after" "^=== ${header} ===" "$LOG" \
        | grep -E 'Number of (wires|cells)' \
        | head -2
}

report "conv5x5_pe  (+ conv5x5_row_mac, requantize)" 'design hierarchy' 20
report "avg_pool2x2_int8  (combinational)"           'avg_pool2x2_int8' 12
report "dense_row_mac  (combinational)"              'dense_row_mac'    12

echo
echo "PASS: generic synthesis completed for all 3 modules -- full log in $LOG"
