#!/usr/bin/env bash
# Run the full open-source regression and print only the verdict lines, so the
# result fits on one screen for the presentation.
#
#   wsl -e bash scripts/regression_summary.sh
#
# The full transcript (every iverilog/vvp invocation) stays in the log file.
set -u

LOG=${LOG:-results/regression_summary.log}
mkdir -p "$(dirname "$LOG")"

# awk (not `| head -1`) reads iverilog's whole banner: closing the pipe early
# makes iverilog print "Unable to get version from ..." on stderr.
IVL=$(iverilog -V 2>/dev/null | awk 'NR==1 {print $4}')
PY=$(python3 -V 2>&1 | awk '{print $2}')

echo "+ make regression   (Icarus $IVL, Python $PY)"
echo "  ... running, a few minutes: sim-top alone simulates ~203,000 cycles"
echo

if ! make regression >"$LOG" 2>&1; then
    echo "FAILED -- see $LOG"
    tail -25 "$LOG"
    exit 1
fi

grep -E '^(Ran [0-9]+ tests|OK)$' "$LOG"
grep -E '^PASS ' "$LOG"

echo
echo "$(grep -c '^PASS ' "$LOG")/8 RTL testbenches passed -- full log in $LOG"
