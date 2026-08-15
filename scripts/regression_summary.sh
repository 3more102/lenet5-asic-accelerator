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

# Number of distinct testbenches expected to report a PASS line. Asserted
# rather than printed, because a testbench that silently stops running still
# leaves a screen full of green PASS lines -- the failure mode this summary
# screen would otherwise hide.
EXPECTED_TBS=14

echo "+ make regression   (Icarus $IVL, Python $PY)"
echo "  ... running, a few minutes: sim-top alone simulates ~356,000 cycles"
echo "      (two back-to-back inferences at 146,544 cycles each, plus ROM load)"
echo

if ! make regression >"$LOG" 2>&1; then
    echo "FAILED -- see $LOG"
    tail -25 "$LOG"
    exit 1
fi

grep -E '^(Ran [0-9]+ tests|OK)$' "$LOG"
grep -E '^PASS ' "$LOG"

echo
# Count distinct testbenches, not PASS lines: tb_lenet5_top, tb_config_guard,
# tb_robustness and tb_extremes each report several, so counting lines would
# print a nonsense fraction like 24/12.
TBS=$(grep -Eo '^PASS [a-zA-Z0-9_]+' "$LOG" | sort -u | wc -l)
echo "$TBS/$EXPECTED_TBS RTL testbenches passed -- full log in $LOG"

if [ "$TBS" -ne "$EXPECTED_TBS" ]; then
    echo "FAILED -- expected $EXPECTED_TBS testbenches to report PASS, got $TBS."
    echo "         A testbench stopped running rather than started failing."
    exit 1
fi
