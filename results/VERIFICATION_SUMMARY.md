# Verification Summary

Date: 2026-08-07

## Status

PASS - Python golden-model tests, the full ModelSim/Questa RTL regression,
the full Icarus Verilog RTL regression, and real Yosys synthesis of every
currently-synthesizable module all completed successfully. The Windows-side
environment has no local Yosys/Icarus install; this pass added a WSL2 Ubuntu
toolchain (Yosys 0.52, Icarus Verilog 12.0, GTKWave 3.3.126) specifically to
close that gap, so the results below are real tool output, not projected or
skipped. `avg_pool2x2_stream`/`dense_engine`/`classifier_argmax`/`lenet5_top`
remain unsynthesized by design (behavioral ROM/scratch arrays, same rationale
as `conv2d_engine` -- see docs/SEMICUSTOM_FLOW.md); `avg_pool2x2_int8` and
the new `dense_row_mac` (both pure combinational, no clock) are now
synthesized for the first time.

**Real synthesis caught a real bug**: `dense_row_mac.sv`'s original
reduction-tree implementation used a procedural `while` loop with a
variable-stride `for` bound. It simulated correctly in both ModelSim and
Icarus, but Yosys's synthesizer rejected it (`2nd expression of procedural
for-loop is not constant`) because the loop bound wasn't a compile-time
constant. Fixed by rebuilding the same pairwise tree with `generate`/`genvar`
(elaboration-time, not procedural), re-verified functionally identical in
both simulators, and now confirmed synthesizable.

## Executed results

- 15/15 Python golden-model unit tests passed (`python -m unittest
  golden.test_golden -v`), including the new `dense_int8`, `argmax_classifier`
  (tie-break and saturation-misclassification regressions), and
  `deploy_forward_int8` shape-chain checks.
- Canonical LeNet-5 layer shapes passed from C1 through F6 (paper/canonical
  float model, `golden/lenet5.py`).
- Canonical trainable-parameter total checked at exactly 60,000.
- RTL C3 connectivity checked at exactly 60 enabled channel pairs.
- `conv5x5_pe` accumulation, bias injection, and backpressure test passed.
- `conv2d_engine` produced 48/48 outputs matching the bit-exact Python model.
- `avg_pool2x2_stream` (S2/S4 pooling controller) produced 12/12 outputs
  matching `avg_pool2x2_int8`.
- `dense_engine` (F6 configuration) produced 5/5 outputs matching `dense_int8`.
- `classifier_argmax` predicted the correct class against `argmax_classifier`,
  including a dedicated hand-computed tie-break check (lowest index wins).
- `lenet5_top` produced the correct predicted class end to end on the full
  canonical 32x32x1 input, bit-exact against `golden/deploy.py:
  deploy_forward_int8` (~203,000 cycles simulated, well inside a 600,000-cycle
  watchdog).
- SystemVerilog elaboration/compilation passed for all 14 RTL modules and 8
  testbenches via `scripts/modelsim.do` (ModelSim/Questa Intel FPGA Edition
  10.5b), 0 errors / 0 warnings.
- The same 8 testbenches also pass under `make regression` (Icarus Verilog
  12.0, via WSL2 Ubuntu) -- `python -m unittest`, `lint` elaboration of
  `conv2d_engine`/`conv5x5_pe`/`lenet5_top`, all 8 `sim-*` targets, and the
  canonical-model demo. Full log: `results/icarus_regression_20260807.log`.
- Generic Yosys synthesis (0.52) passed for `conv5x5_pe` (+ `conv5x5_row_mac`,
  `requantize`), `avg_pool2x2_int8`, and `dense_row_mac` via `make synth`.
  Full log: `results/yosys_synth_20260807.log`.

## Generic synthesis snapshot

| Module | Cells | Flip-flops | Notes |
|---|---:|---:|---|
| `conv5x5_pe` (+ `conv5x5_row_mac`, `requantize`) | 4,762 | 74 | Interface and arithmetic unchanged; requantization became a second pipeline stage after STA found the row-MAC and requantize adders sharing one combinational cycle (`docs/PPA.md`). That added 33 flip-flops and 35 cells versus the 4,727/41 reported on 2026-08-07 (`synth/yosys.ys`, `results/conv5x5_pe_generic.json`). |
| `avg_pool2x2_int8` | 303 | 0 (combinational) | First synthesis run for this module -- it predates this session but was never wired into `synth/yosys.ys` before (`synth/avg_pool2x2_int8.ys`, `results/avg_pool2x2_int8_generic.json`). |
| `dense_row_mac` | 4,672 | 0 (combinational) | New this session; required the while-loop-to-generate/genvar fix above (`synth/dense_row_mac.ys`, `results/dense_row_mac_generic.json`). |

These are technology-independent cell counts from generic synthesis
(`proc; opt; fsm; opt; memory; opt; techmap; opt`), not foundry area
estimates -- superseded by the real sky130hd numbers in `docs/PPA.md`.
`avg_pool2x2_int8` and `dense_row_mac` have no clock port (pure combinational
primitives), so a conventional per-block SDC with a real `create_clock` does
not apply to them standalone. `docs/PPA.md` sidesteps that with a virtual
clock plus input/output delays -- the standard STA technique for timing a
combinational block in isolation -- rather than waiting for them to be
embedded in a clocked wrapper (`avg_pool2x2_stream`, `dense_engine`), which
are still not synthesized.

## Expected warnings

Icarus reports that the behavioral activation, weight, and connection arrays
appear in combinational sensitivity lists. These arrays belong to the
memory-backed reference engine. The production architecture replaces them with
banked foundry SRAM macros and a latency-aware feeder, as described in the
architecture and interface documents.

## Not claimed

This pass used generic Yosys techmap only (`proc; opt; fsm; opt; memory; opt;
techmap; opt`) — no standard-cell library, so no silicon area, power,
frequency, DRC, LVS, IR drop, electromigration, scan coverage, or SRAM
sign-off from these numbers specifically.

That is no longer the whole story for this project, though: a later pass ran
real gate-level synthesis onto the **sky130hd** PDK plus static timing/power
analysis for these same arithmetic blocks — see
[`docs/PPA.md`](../docs/PPA.md). Still pre-layout (no place-and-route), still
not a tapeout, but real cells and real Liberty timing arcs, not generic
techmap counts. Don't read the table below as if it were the final PPA
picture.

