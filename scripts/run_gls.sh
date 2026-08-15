#!/usr/bin/env bash
# Gate-level simulation: re-run the existing testbenches against the
# sky130hd-mapped netlists instead of the RTL.
#
# Usage:
#   bash scripts/run_gls.sh                       # every block below
#   bash scripts/run_gls.sh conv5x5_pe            # one block
#   ORFS_ROOT=/path/to/OpenROAD-flow-scripts bash scripts/run_gls.sh
#
# Why this exists. Every simulation in `make regression` drives RTL, and
# `make equiv` proves RTL == *generic* netlist. Neither says anything about
# the sky130hd-mapped netlist that docs/PPA.md reports area, timing and power
# for -- and that netlist is the one produced by real technology mapping, where
# abc restructures the logic against a real cell library. `make equiv-mapped`
# proves two of these blocks equivalent outright, but unbounded equivalence
# over a mapped multiply-accumulate is beyond a plain SAT solver (see
# docs/VERIFICATION_PLAN.md). Simulation has no such wall, so this is how the
# MAC blocks' netlists get checked: same testbenches, same golden vectors,
# gates instead of RTL.
#
# The mapping flow below is deliberately identical to asic/sta/run_ppa.sh's --
# same liberty, same `synth -flatten`, same `dfflibmap`, same `abc -liberty`
# at the same 10 ns target -- so this simulates the netlist those numbers came
# from, not a differently-optimized one. Generating the yosys script from a
# heredoc rather than checking one in per block follows run_ppa.sh for the same
# reason it does: the flow is one thing, and it should be edited in one place.
#
# Some blocks run pure-gate (the netlist is the whole DUT) and some run mixed
# RTL/gate (a mapped leaf inside an RTL wrapper), because avg_pool2x2_int8's
# wrapper is where its stream interface lives -- see docs/SEMICUSTOM_FLOW.md
# for why those wrappers are not synthesizable yet.
#
# A block may list more than one testbench. Mapping is what costs the time here
# (abc on dense_row_mac is ~35 minutes; the simulations are seconds), so once a
# netlist exists it is worth running everything that reaches it: the two MAC
# blocks get both their own direct testbench, which sweeps every lane, and the
# wrapper testbench that exercises them in place.

set -euo pipefail

ORFS_ROOT="${ORFS_ROOT:-/root/OpenROAD-flow-scripts}"
LIB="$ORFS_ROOT/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_HOME="$(cd "$HERE/.." && pwd)"
OUT="$PROJECT_HOME/results/gls"

IVERILOG="${IVERILOG:-iverilog}"
VVP="${VVP:-vvp}"
YOSYS="${YOSYS:-yosys}"

# abc takes its delay target in ps; 10 ns is run_ppa.sh's default period.
PERIOD_PS="${PERIOD_PS:-10000}"

[ -f "$LIB" ] || { echo "ERROR: missing PDK liberty $LIB" >&2; exit 1; }
for tool in "$YOSYS" "$IVERILOG" "$VVP"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: $tool not on PATH" >&2; exit 1; }
done

# Sources synthesized into each netlist. conv5x5_pe's list matches run_ppa.sh's:
# the netlist is flattened, so it absorbs conv5x5_row_mac and requantize and one
# netlist covers all three blocks.
sources_for() {
    case "$1" in
        requantize)       echo "rtl/requantize.sv" ;;
        conv5x5_row_mac)  echo "rtl/conv5x5_row_mac.sv" ;;
        conv5x5_pe)       echo "rtl/conv5x5_row_mac.sv rtl/requantize.sv rtl/conv5x5_pe.sv" ;;
        avg_pool2x2_int8) echo "rtl/avg_pool2x2_int8.sv" ;;
        dense_row_mac)    echo "rtl/dense_row_mac.sv" ;;
        *) echo "ERROR: no source list for block '$1'" >&2 ; return 1 ;;
    esac
}

# Which testbenches exercise the netlist, as <root module>:<file> pairs. More
# than one is allowed and costs only the simulation: the mapped netlist is
# built once per block and reused for each.
testbenches_for() {
    case "$1" in
        requantize)       echo "tb_requantize:tb/tb_requantize.sv" ;;
        conv5x5_row_mac)  echo "tb_conv5x5_row_mac:tb/tb_conv5x5_row_mac.sv tb_conv5x5_pe:tb/tb_conv5x5_pe.sv" ;;
        conv5x5_pe)       echo "tb_conv5x5_pe:tb/tb_conv5x5_pe.sv" ;;
        avg_pool2x2_int8) echo "tb_avg_pool2x2_stream:tb/tb_avg_pool2x2_stream.sv" ;;
        dense_row_mac)    echo "tb_dense_row_mac:tb/tb_dense_row_mac.sv tb_dense_engine:tb/tb_dense_engine.sv" ;;
        *) echo "ERROR: no testbench list for block '$1'" >&2 ; return 1 ;;
    esac
}

BLOCKS="${*:-requantize conv5x5_row_mac conv5x5_pe avg_pool2x2_int8 dense_row_mac}"

mkdir -p "$OUT"
cd "$PROJECT_HOME"

fails=0
for block in $BLOCKS; do
    srcs="$(sources_for "$block")"
    tbs="$(testbenches_for "$block")"
    netlist="$OUT/${block}_gls.v"
    shim="tb/gls_shim_${block}.sv"
    ylog="$OUT/${block}.yosys.log"

    [ -f "$shim" ] || { echo "ERROR: missing $shim" >&2; exit 1; }

    # ---- map to sky130hd cells, then back to gates a simulator can run ------
    # read_liberty -lib gives abc/dfflibmap the cell interfaces as blackboxes.
    # Reading the same liberty again with -overwrite replaces those blackboxes
    # with each cell's `function` as real logic, so `flatten` turns the mapped
    # netlist into plain Verilog that Icarus can execute without a vendor cell
    # model. What that does *not* cover is anything living only in a vendor
    # behavioural model -- X-propagation detail, timing checks, UDP internals.
    cat > "$OUT/${block}.ys" <<EOF
# Generated by scripts/run_gls.sh -- do not edit.
read_liberty -lib $LIB
$(for s in $srcs; do echo "read_verilog -sv $PROJECT_HOME/$s"; done)
hierarchy -check -top $block
synth -top $block -flatten
dfflibmap -liberty $LIB
abc -liberty $LIB -D $PERIOD_PS
setundef -zero
opt_clean -purge
check
splitnets
opt_clean -purge
stat -liberty $LIB
read_liberty -overwrite -ignore_miss_func -ignore_miss_dir -ignore_miss_data_latch -ignore_buses $LIB
hierarchy -check -top $block
flatten
opt_clean -purge
rename $block ${block}_gls
write_verilog -noattr $netlist
EOF

    echo "== map : $block"
    "$YOSYS" -q -l "$ylog" -s "$OUT/${block}.ys"
    # Summed from `stat -liberty`'s per-cell table rather than its "Number of
    # cells" line, which yosys -q stopped emitting somewhere between 0.52 and
    # 0.68. Expect these to land near -- not exactly on -- the matching row of
    # asic/sta/results/ppa_summary.csv: the flow is identical, but that CSV was
    # produced by yosys 0.52 and abc maps a little differently now.
    cells=$(awk '$3 ~ /^sky130_fd_sc_hd__/ {n += $1} END {print n+0}' "$ylog")
    area=$(awk '/Chip area for module/{a=$NF} END{printf "%.1f", a+0}' "$ylog")

    # yosys does not emit a `timescale into the netlist. Listing the testbench
    # first would let the netlist inherit one, but Icarus warns about exactly
    # that under -Wall, so prepend it instead and keep the build warning-free.
    printf '`timescale 1ns/1ps\n' | cat - "$netlist" > "$netlist.tmp"
    mv "$netlist.tmp" "$netlist"

    # ---- simulate ----------------------------------------------------------
    # The block's own RTL file is dropped and the shim takes its module name,
    # so anything that instantiates it -- the testbench directly, or an RTL
    # wrapper above it -- binds to the netlist. Sibling RTL files stay in the
    # list; with -s <tbtop> only what the root actually reaches is elaborated.
    rtl_list=""
    for f in rtl/*.sv; do
        [ "$f" = "rtl/${block}.sv" ] && continue
        rtl_list="$rtl_list $f"
    done

    echo "== map : $block  cells=$cells area=${area}um2"

    for entry in $tbs; do
        tbtop="${entry%%:*}"
        tb="${entry#*:}"
        slog="$OUT/${block}.${tbtop}.sim.log"

        echo "== sim : $block  ($tbtop against the mapped netlist)"
        "$IVERILOG" -g2012 -Wall -I. -s "$tbtop" -o "$OUT/${block}.${tbtop}.vvp" \
            $rtl_list "$shim" "$netlist" $tb
        "$VVP" "$OUT/${block}.${tbtop}.vvp" > "$slog" 2>&1 || true

        # The testbenches self-check and $fatal on mismatch, but a netlist can
        # also fail by producing nothing at all -- X-propagation that trips an
        # early $finish leaves a clean exit and an empty verdict. Require the
        # PASS line.
        if grep -q "^PASS $tbtop" "$slog"; then
            echo "   PASS  cells=$cells area=${area}um2  $(grep -m1 "^PASS $tbtop" "$slog")"
        else
            echo "   FAIL  no 'PASS $tbtop' line in $slog"
            tail -5 "$slog" | sed 's/^/        /'
            fails=$((fails + 1))
        fi
    done
done

echo
if [ "$fails" -ne 0 ]; then
    echo "== gate-level simulation FAILED for $fails block(s)"
    exit 1
fi
echo "== gate-level simulation passed for every block: $BLOCKS"
