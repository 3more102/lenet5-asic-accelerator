#!/usr/bin/env bash
# Settle one question: is the surviving conv5x5_pe output-register mutation a
# hole in the stimulus, or a change no consumer can observe?
#
#   bash scripts/prove_pe_output_hold.sh [depth]     # default 20
#
# Analysis artifact, deliberately outside `make regression` and CI: it proves a
# property of a *mutant*, which is only meaningful next to the mutation it came
# from. The result is written up in docs/VERIFICATION_PLAN.md; this script is
# here so the claim can be re-derived rather than taken on trust.
#
# Three runs, and all three matter:
#
#   PROOF     the miter's bad_o is unreachable, so the mutation is invisible at
#             the valid/ready interface;
#   COVER     an output beat is actually reachable inside the same window --
#             without this the proof could hold vacuously over a window in which
#             the PE never produces anything;
#   NEGATIVE  the same proof over an observably wrong design must FAIL, or
#             "proved" is also what a miter wired to a constant would report.
#
# Why the datapath is cut away. Proving this over the real multiplier array
# unrolls two 5-tap MACs -- 591k variables at depth 24, which is the same SAT
# wall unbounded equivalence hits on a mapped MAC (see docs/VERIFICATION_PLAN.md).
# The property does not depend on the arithmetic at all, so `cutpoint` turns
# row_sum and quantized into free variables and `sat -set` constrains them to be
# the same in both copies. That is 12k variables, and it proves the property for
# *any* row MAC and *any* requantizer rather than only for these two.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_HOME="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_HOME"

YOSYS="${YOSYS:-yosys}"
command -v "$YOSYS" >/dev/null 2>&1 || { echo "ERROR: $YOSYS not on PATH" >&2; exit 1; }

DEPTH="${1:-20}"
W="${W:-$PROJECT_HOME/results/pe_hold}"
rm -rf "$W"; mkdir -p "$W"

MITER="$PROJECT_HOME/synth/pe_output_hold_miter.sv"
MUT="$W/conv5x5_pe_mut.sv"

# Build a mutant from the real RTL rather than a hand-copied one, so this can
# never drift away from the design it claims to be about.
mutate() {
    sed "$1" rtl/conv5x5_pe.sv > "$MUT"
    if cmp -s rtl/conv5x5_pe.sv "$MUT"; then
        echo "ERROR: mutation pattern did not match rtl/conv5x5_pe.sv" >&2
        exit 1
    fi
    sed -i 's|^module conv5x5_pe |module conv5x5_pe_mut |' "$MUT"
}

script_for() {
    cat <<EOF
read_verilog -sv rtl/conv5x5_row_mac.sv
read_verilog -sv rtl/requantize.sv
read_verilog -sv rtl/conv5x5_pe.sv
read_verilog -sv $MUT
read_verilog -sv $MITER
prep -top pe_output_hold_miter -flatten
async2sync
memory_map
cutpoint w:u_gold.row_sum w:u_gate.row_sum w:u_gold.quantized w:u_gate.quantized
opt -full
sat -seq $DEPTH -verify $1 -set-at 1 rst_ni 0 \\
    -set u_gate.row_sum u_gold.row_sum -set u_gate.quantized u_gold.quantized
EOF
}

run() {
    local name="$1" args="$2"
    script_for "$args" > "$W/$name.ys"
    # stderr goes to the same log rather than the terminal: the negative control
    # is *meant* to fail, and yosys printing "proof did fail" between the
    # verdict lines reads like something went wrong when it is the check working.
    "$YOSYS" -q -l "$W/$name.log" -s "$W/$name.ys" 2>>"$W/$name.log"
    echo "$?"
}

fails=0

# ---- 1. the proof ---------------------------------------------------------
mutate 's|                if (rq_valid_q) begin|                if (1'"'"'b1) begin|'
rc=$(run proof "-prove bad_o 1'b0")
if [ "$rc" -eq 0 ]; then
    echo "PROOF     : bad_o unreachable to depth $DEPTH -- the mutation is invisible at the interface"
    grep -E "Solving problem with" "$W/proof.log" | tail -1 | sed 's/^/            /'
else
    echo "PROOF     : FAILED (rc=$rc) -- the mutation IS observable; see $W/proof.log"
    fails=$((fails + 1))
fi

# ---- 2. cover -------------------------------------------------------------
# `sat -verify` errors when a plain satisfiability query finds no model, which
# is what makes it usable as a cover check: rc 0 here means a run reaching an
# output beat exists inside the window.
rc=$(run cover "-set-at $DEPTH u_gold.out_valid_o 1")
if [ "$rc" -eq 0 ]; then
    echo "COVER     : an output beat is reachable at depth $DEPTH -- the proof window is not vacuous"
else
    echo "COVER     : FAILED (rc=$rc) -- no output beat inside the window, so the proof above proves nothing"
    fails=$((fails + 1))
fi

# ---- 3. negative control --------------------------------------------------
# Inverting the payload keeps every net alive. The obvious alternative -- never
# writing out_data_o -- lets opt delete the nets the -set arguments name, and
# yosys then exits non-zero on a *parse* error, which would be read as a
# rejected proof when nothing was ever solved.
mutate 's|                    out_data_o <= quantized;|                    out_data_o <= ~quantized;|'
rc=$(run negative "-prove bad_o 1'b0")
if grep -q "Failed to parse\|ERROR: Can't find" "$W/negative.log"; then
    echo "NEGATIVE  : INVALID -- yosys errored before solving, so this control checked nothing"
    fails=$((fails + 1))
elif [ "$rc" -ne 0 ]; then
    echo "NEGATIVE  : an observably wrong design is rejected, as it must be"
else
    echo "NEGATIVE  : FAILED -- the miter proved a design with an inverted output. Do not trust the proof above."
    fails=$((fails + 1))
fi

echo
if [ "$fails" -ne 0 ]; then
    echo "== $fails of 3 checks did not hold; logs in $W"
    exit 1
fi
echo "== all three hold: the surviving output-register mutation is unobservable at the"
echo "   valid/ready interface, for any row MAC and any requantizer, to depth $DEPTH."
