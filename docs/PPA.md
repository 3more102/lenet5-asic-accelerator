# Real PPA: sky130hd, pre-layout

## Scope, stated plainly

This is **real gate-level synthesis and static timing/power analysis against
the open sky130hd PDK** for every currently-synthesizable block in this
project — not generic techmap counts, and not a projection.

It is also, just as plainly, **not a tapeout database**:

| Covered here | Not covered here |
|---|---|
| Real sky130hd standard cells (`yosys`/`abc`) | Place-and-route (floorplan, CTS, routing) |
| Real Liberty timing arcs (OpenSTA) | Routed parasitics / post-layout timing |
| Real per-cell leakage + switching power | Measured switching activity (see below) |
| One typical corner (tt, 25 °C, 1.8 V) | ss/ff/sign-off multi-corner |
| The five storage-free arithmetic leaf/PE blocks | The five memory-backed blocks (still behavioral) |
| — | DRC, LVS, IR drop, EM, scan, SRAM sign-off |

Two things about the numbers below specifically:

- **Power uses a stated, not measured, switching activity** — 10% toggle
  rate, 50% duty (`set_power_activity -global -activity 0.10 -duty 0.50` in
  `asic/sta/sta.tcl`). It is not derived from real simulation vectors. Treat
  it as a reasonable order-of-magnitude assumption, not a measurement.
- **Timing includes a stated I/O delay budget, not zero-load ideal ports** —
  1 ns input delay, 1 ns output delay, 0.2 ns clock uncertainty, each input
  driven by a real `sky130_fd_sc_hd__inv_2` cell and each output loaded with
  10 fF. That budget is folded into every "minimum period" figure below, so
  they represent an achievable block-level integration target, not an
  inflated best-case gate-delay-only number.

## Methodology: two independent paths, same PDK

**Path 1 — `asic/sta/run_ppa.sh`** (produced every number in the table
below). Synthesizes each block straight to sky130hd cells with `yosys`/`abc`,
then drives OpenSTA — via the `openroad` binary, which embeds it — directly
on the mapped netlist with a generated clock/IO constraint (real SDC files
don't apply cleanly to two of these blocks; see "Combinational blocks"
below). Needs only `yosys` and `openroad` on `PATH`. Run it with:

```bash
make ppa
```

**Path 2 — `asic/openroad/run_orfs.sh`**, the full ORFS RTL-to-GDS entry
point, config in `asic/openroad/config.mk`. This is the "closest to a real
flow" path and is what produced the `conv5x5_pe` **generic-mapped-onto-sky130**
snapshot below (its synthesis stage, harvested into
`asic/openroad/results/synth_stat.txt`) — but place-and-route does not
currently run in this environment (embedded-Python version mismatch, not a
design issue; see `asic/openroad/patches/README.md`), so Path 1 is what
supplies real timing/power here, not Path 2.

Both paths read the same `sky130_fd_sc_hd__tt_025C_1v80.lib` and the same
`sky130_fd_sc_hd.tlef` / `sky130_fd_sc_hd_merged.lef`.

### Combinational blocks

`avg_pool2x2_int8` and `dense_row_mac` have no clock port — they're pure
combinational primitives (see `results/VERIFICATION_SUMMARY.md`). A
conventional SDC with `create_clock` on a real port doesn't apply. `asic/sta/sta.tcl`
handles this the standard STA way: a virtual clock plus input/output delays,
so the block still gets a real, reportable setup check without needing a
clocked wrapper around it.

## Headline table (period = 10 ns / 100 MHz point)

| Block | Cells | Seq. cells | Area (µm²) | Min. period* | Fmax* | Setup met @ 10 ns | Worst hold slack | Power @ 10 ns† |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|
| `conv5x5_row_mac` | 1,776 | 0 | 15,846.448 | 9.804 ns | 102.0 MHz | yes (WNS +0.196 ns) | +2.197 ns | 406.1 µW |
| `requantize` | 687 | 0 | 4,310.384 | 18.544 ns | 53.9 MHz | **no** (WNS −8.544 ns) | +1.875 ns | 120.5 µW |
| `avg_pool2x2_int8` | 128 | 0 | 1,057.264 | 5.841 ns | 171.2 MHz | yes (WNS +4.159 ns) | +2.229 ns | 28.3 µW |
| `dense_row_mac` | 2,836 | 0 | 25,719.667 | 10.664 ns | 93.8 MHz | **no** (WNS −0.664 ns) | +2.063 ns | 655.1 µW |
| `conv5x5_pe` (full PE: row-mac + requantize + accumulator) | 3,094 | 74 | 24,993.971 | 16.007 ns | 62.5 MHz | **no** (WNS −6.007 ns) | +0.238 ns | 943.4 µW |

\* "Min. period" is the clock period at which setup WNS = 0 under the stated
I/O delay budget above — i.e. the real achievable period, not a bare gate
delay. Confirmed **period-invariant** for the first three blocks (see next
section), so it is reported once rather than as a 9-point sweep.
† Power scales with clock frequency in this model (see next section) —
energy/operation, given in the per-block notes below, is the more portable
number.

Full per-block logs and the raw CSV: `asic/sta/results/ppa_summary.csv`,
`asic/sta/results/*.sta.log`, `asic/sta/results/*.yosys.log`.

### `conv5x5_pe`, generic-mapped onto sky130hd (ORFS synthesis stage, Path 2)

For reference, ORFS's own synthesis stage (which flattens and maps somewhat
differently than the standalone Path 1 script) mapped the full PE to:

- **5,490 cells**, **5,494 wires** / 5,615 wire bits, 14 ports / 135 port bits
- **41 flip-flops** (`sky130_fd_sc_hd__dfrtp_1`)
- **39,175.072 µm²** total, of which 1,025.984 µm² (2.62%) is sequential
- `synth_check.txt`: *"Found and reported 0 problems"*
- Largest cell populations: `nand2_1` ×742, `xnor2_1` ×472, `nor2_1` ×444,
  `o21ai_0` ×292, `maj3_1` ×238, `xor2_1` ×215, `clkbuf_1` ×215

Full cell breakdown: `asic/openroad/results/synth_stat.txt`. This ORFS run
did not reach STA (place-and-route is blocked, and this ORFS revision's
synthesis stage does not itself close timing) — the Path-1 `conv5x5_pe` row
above is the real timing/power source. The two cell counts differ (this one
vs. the Path-1 row) because the two flows flatten/map slightly differently;
neither is wrong, they're just not the same recipe. Treat ORFS's count as the
"closest to a real vendor-style flow" reference point and Path 1's as the
timing/power source.

## The period-invariance finding

The first three blocks were each swept across nine target clock periods (20,
15, 12, 10, 8, 6, 5, 4, 3 ns, via `abc -D <period>`). Across all 27 runs, each
block's mapped **cell count, area, and critical path were bit-for-bit
identical regardless of the target period** — only which side of the pass/
fail line a given period fell on changed:

| Block | Critical path across all 9 periods |
|---|---|
| `conv5x5_row_mac` | 9.8039 ns, every time |
| `requantize` | 18.5439 ns, every time |
| `avg_pool2x2_int8` | 5.8413 ns, every time |

In other words, this `yosys`/`abc` invocation is not doing delay-driven
restructuring at these design sizes — `-D` changes which mapping *would* be
selected only if a tighter target forced a different structure, and here it
never did. **A real 9-point sweep carries no more information than one data
point plus arithmetic** (WNS = period − critical_path), which is why
`dense_row_mac` and `conv5x5_pe` above were each run at a single period (10 ns,
the SDC default in `synth/conv5x5_pe.sdc`) rather than repeating the full
sweep — and why power, which is driven by the assumed *toggle rate relative
to the clock*, scales linearly with frequency in this data (confirmed:
`conv5x5_row_mac`'s power at every one of the 9 periods divides out to the
same ~4.06 pJ per cycle). Report energy per operation, not power at an
arbitrary frequency, if quoting one number:

| Block | Energy per operation @ 10 ns (10%/50% activity assumption) |
|---|---:|
| `conv5x5_row_mac` (one kernel-row MAC) | 4.061 pJ |
| `requantize` (one output requantization) | 1.205 pJ |
| `avg_pool2x2_int8` (one 2×2 pooling window) | 0.283 pJ |
| `dense_row_mac` (one dense-layer row MAC) | 6.551 pJ |
| `conv5x5_pe` (one full PE cycle: MAC + requantize) | 9.434 pJ |

Full 9-point sweep data for the three fast blocks:
`asic/sta/results/ppa_period_sweep_3blocks.csv`.

## Where the time actually goes: `requantize`'s adder, not its shifter

`requantize` is the only one of the three fast blocks that fails to close at
the stated 100 MHz / 10 ns reference clock (needs 53.9 MHz instead). Its
setup path (`asic/sta/results/requantize_10ns.sta.log`) starts at `shift_i`,
passes through a handful of decode gates, then a **~20-cell-deep chain of
`sky130_fd_sc_hd__maj3_1`** — the carry chain of the 33-bit round-away-from-
zero adder — before reaching a few more gates and `data_o`. The barrel
shifter that `shift_i` drives is a large fraction of this block's *area*
(see `docs/QA_PREP.md`'s cell-count breakdown), but the adder's carry chain,
not the shifter, is what limits its *frequency*. The two are different
problems with different fixes: constant-folding `shift_i` (every layer uses
shift = 7, [`golden/deploy.py:28`](../golden/deploy.py#L28)) would shrink
area; a pipelined or carry-select adder would fix timing. Neither has been
implemented — this is a measured diagnosis, not yet an optimization.

## `conv5x5_pe`: a measured bottleneck, fixed and re-measured

This is the one place in the project where STA found a real problem, the RTL
changed because of it, and the fix was verified by re-running the same flow.

**The diagnosis.** The original `conv5x5_pe` failed the 100 MHz reference
clock by far more than either of its sub-blocks alone — 37.4 MHz, against
102.0 MHz for `conv5x5_row_mac` and 53.9 MHz for `requantize` individually.
Its setup path started at a weight input (`wgt_row_i[16]`), ran through a
`maj3_1`/`xor3_1` stretch matching `conv5x5_row_mac`'s adder tree, crossed a
2.35 ns high-fanout gate, then ran through a second, longer `maj3_1` stretch
matching `requantize`'s carry chain, before reaching the accumulator
flip-flop. The arithmetic was blunt: **26.76 ns ≈ 9.80 ns + 18.54 ns**. Both
adders sat in one combinational cycle, so the block paid for both critical
paths in series on every accepted row.

**The fix.** One pipeline register. `conv5x5_pe` now latches the accumulator
result into `result_q` before it enters `requantize`
([`rtl/conv5x5_pe.sv`](../rtl/conv5x5_pe.sv)), making requantization a second
pipeline stage with its own ready/valid handshake. Throughput is unchanged —
still one kernel row per accepted cycle — at the cost of one extra cycle of
output latency per pixel.

**The re-measurement**, same flow, same PDK, same constraints:

| | Before (single-cycle) | After (registered RQ) | Change |
|---|---:|---:|---:|
| Critical path | 26.759 ns | 16.007 ns | **−40.2%** |
| Fmax | 37.4 MHz | **62.5 MHz** | **1.67×** |
| Cells | 3,015 | 3,094 | +79 (+2.6%) |
| Sequential cells | 41 | 74 | +33 |
| Area | 24,074.339 µm² | 24,993.971 µm² | +3.8% |
| Power @ 10 ns | 794.9 µW | 943.4 µW | +18.7% |
| Worst hold slack | +0.232 ns | +0.238 ns | — |

The +33 flip-flops are exactly the new register: 32 bits of `result_q` plus
one `rq_valid_q`. **1.67× the frequency for 3.8% more area** is the kind of
trade this project previously had no way to even state, let alone measure.

**What the path became.** The new setup path
(`asic/sta/results/conv5x5_pe_10ns.sta.log`) starts at `shift_i[0]` and runs
through a 33-deep `maj3_1` chain into the output flop — i.e. it is now purely
the `requantize` carry chain described in the section above, with the row-MAC
no longer in series with it. The remaining bottleneck is therefore the *same*
one the standalone `requantize` analysis identified, and the same two
independent fixes apply: constant-folding `shift_i` (now literally the
startpoint of the critical path) for area, and a pipelined or carry-select
adder for the rest of the frequency. Neither is implemented.

**Not applied to `conv2d_engine`.** The production datapath
(`lenet5_top` → `conv2d_engine`) inlines `conv5x5_row_mac` and `requantize`
directly rather than instantiating `conv5x5_pe`, and it has the identical
back-to-back structure — plus an `act_mem`/`wgt_mem` read in the same
combinational cycle, so it is very likely worse. The same one-register fix
would apply. It was deliberately **not** made here: it shifts the end-to-end
cycle count, which is the verified 202,866-cycle result quoted across the
docs and captured in the committed screenshots. Changing a verified baseline
is a separate, deliberate piece of work, not a side effect of a timing fix.

## SRAM sizing for the still-unsynthesized blocks

Not part of this pass (see the scope table above), but the real bit counts,
for context on the memory-macro work `docs/INTERFACES.md`'s worked example
describes:

| Array | Depth × width | Total bits |
|---|---|---:|
| `conv2d_engine.act_mem` | 16,384 × 8 | 131,072 b (16 KiB) |
| `conv2d_engine.wgt_mem` | 48,000 × 8 | 384,000 b (46.9 KiB) |
| `avg_pool2x2_stream.act_mem` | 12,544 × 8 | 100,352 b (12.25 KiB) |
| `dense_engine.wgt_mem` | 10,080 × 8 | 80,640 b (9.84 KiB) |

Against the largest single sky130 OpenRAM macro available in this PDK build
(`sky130_sram_1rw1r_128x256_8`, 32,768 b), `wgt_mem` alone would need on the
order of 12 macros banked together, before even addressing the four/five-
simultaneous-reads-per-cycle issue `docs/INTERFACES.md` covers.

## Toolchain notes

Two version mismatches were hit getting this far, and one blocks going
further (place-and-route). Both are recorded precisely, with the exact
error text and the fix (where one exists), in
[`asic/openroad/patches/README.md`](../asic/openroad/patches/README.md) —
not repeated here to avoid the two copies drifting apart.

## Reproducing this

```bash
# Path 1: real sky130hd synthesis + STA for every leaf/PE block (this doc's source)
make ppa

# Path 2: full ORFS synthesis stage for conv5x5_pe (needs an ORFS install;
# see asic/openroad/patches/ if it aborts on `stat -hierarchy`)
make orfs
```

Raw evidence for everything above: `asic/sta/results/` (Path 1: CSV + per-run
`yosys`/`sta` logs) and `asic/openroad/results/` (Path 2: `synth_stat.txt`,
`synth_check.txt`, yosys/run logs). Numbers in this file are transcribed from
those, not hand-entered independently — if the two ever disagree, the
`results/` files are the source of truth.
