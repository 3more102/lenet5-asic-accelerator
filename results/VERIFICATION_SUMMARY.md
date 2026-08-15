# Verification Summary

Date: 2026-08-07, re-verified three times on 2026-08-15 — after the requantize
pipelining change, after adding controller-FSM coverage and the
config-validation reject tier, and after adding the stall-invariance and
reset-interruption tier. See the three Addenda at the end. The body below
reflects the first re-verification; **where a later addendum moves a number,
the later one is authoritative** — that means the cycle figures come from the
second (measured per inference rather than read off `$finish`) and the
testbench count from the third (**11**, not 9 or 10).

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
  deploy_forward_int8` (~209,000 cycles simulated, well inside a 600,000-cycle
  watchdog).
- SystemVerilog elaboration/compilation passed for all 14 RTL modules and 9
  testbenches via `scripts/modelsim.do` (ModelSim/Questa Intel FPGA Edition
  10.5b), 0 errors / 0 warnings.
- The same 9 testbenches also pass under `make regression` (Icarus Verilog
  12.0, via WSL2 Ubuntu) -- `python -m unittest`, `lint` elaboration of
  `conv2d_engine`/`conv5x5_pe`/`lenet5_top`, all 9 `sim-*` targets, and the
  canonical-model demo. Full log: `results/icarus_regression_20260807.log`.
- Generic Yosys synthesis (0.52) passed for `conv5x5_pe` (+ `conv5x5_row_mac`,
  `requantize`), `avg_pool2x2_int8`, and `dense_row_mac` via `make synth`.
  Full log: `results/yosys_synth_20260807.log`.

## Generic synthesis snapshot

| Module | Cells | Flip-flops | Notes |
|---|---:|---:|---|
| `conv5x5_pe` (+ `conv5x5_row_mac`, `requantize`) | 4,125 | 74 | Interface and arithmetic unchanged. Two changes since the 4,727/41 reported on 2026-08-07: requantization became a second pipeline stage (+33 flip-flops, +35 cells), then `requantize`'s rounding was rewritten as shift-then-increment, which removed 637 cells (1,554 → 917 in that submodule). Both are measured in `docs/PPA.md` (`synth/yosys.ys`, `results/conv5x5_pe_generic.json`). |
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

## Addendum — 2026-08-15: requantize pipelining, and full re-verification

Static timing analysis on the sky130hd-mapped netlist showed `conv5x5_pe`
closing at only 37.4 MHz because the row-MAC adder tree and the requantizer's
carry chain shared one combinational cycle. Registering the accumulator
result before `requantize` fixed that (**62.5 MHz, 1.67x, for +3.8% area**), and
rewriting requantize's rounding as shift-then-increment took it further to
**68.6 MHz with 18.8% less area in that block** (**1.84x overall** —
`docs/PPA.md`). The same structural fix was then applied to `conv2d_engine`,
which had the identical pattern plus a memory read in the same cycle.

Everything above was re-run against the changed RTL, in both simulators:

- 15/15 Python golden-model unit tests still pass.
- All 9 self-checking testbenches still pass under Icarus Verilog 12.0 **and**
  ModelSim/Questa 10.5b, 0 errors / 0 warnings.
- `conv2d_engine` still produces 48/48 outputs matching the bit-exact Python
  model, including under the hostile every-7th-cycle backpressure pattern.
- `lenet5_top` still produces the correct predicted class end to end,
  bit-exact against `golden/deploy.py: deploy_forward_int8`.

The one figure that moved is the cycle count: **202,866 -> 209,290 cycles**
(2,092,900 ns at 100 MHz). The `+6,424` is exactly the number of convolution
output pixels (C1 6x28x28 = 4,704, C3 16x10x10 = 1,600, C5 120), i.e. the
predicted one-cycle-per-output-pixel cost and nothing more. Icarus and
ModelSim report the identical finish time independently.

Two limits worth stating plainly. The 1.84x is measured **for `conv5x5_pe`
only**; `conv2d_engine` holds behavioural memory arrays and is not
synthesizable, so its improvement is inferred by structural analogy, not
measured. And `results/screenshots/07_end_to_end_cycles.png` still shows the
old 2,028,660 ns timestamp — it predates this change and needs re-capturing.


## Addendum — 2026-08-15: controller-FSM coverage and the config reject path

Two verification tiers were added and both simulators re-run from scratch.
Everything below is real tool output; logs are
`results/icarus_regression_20260815.log` and
`results/modelsim_regression_20260815.log`.

**New: `tb/fsm_cov.sv` + `tb_lenet5_top` FSM coverage.** A portable
state/transition collector watches `lenet5_top`'s 20-state controller and
fails the run on any unvisited state, any declared-legal transition never
exercised, or any transition outside the declared set. Reported: **20/20
states, 33/33 legal transitions**, in both simulators. It is plain Verilog
rather than a covergroup because Icarus — which runs CI — does not implement
covergroups, and a coverage tier that silently stops collecting still reads as
a pass.

The collector found a real stimulus hole on its first run: `S_RUN_CLS ->
S_IDLE` was never taken, because the testbench had never let the accelerator
return to idle. Closing it meant running a **second inference with no reset and
no weight reload**, which turned out to be the more valuable check of the two.

**New: `tb/tb_config_guard.sv`.** All **22** config-validation reject
conditions (conv 8, pool 8, dense 4, classifier 2) are now driven. Before this,
every testbench only ever asserted `config_error_o` stayed *low* — the
rejection logic had been written, elaborated and synthesized without a single
stimulus reaching it. Each reject is also checked to be inert (`busy_o` low,
`done_o` never pulses), and each engine is then given a legal config to prove
the error clears rather than latching.

**Cycle counts, now measured directly rather than derived.** `tb_lenet5_top`
counts `start_i -> done_o` and asserts the result:

- **146,544 cycles** per inference with the ROMs resident (1.47 ms @ 100 MHz);
- **identical** on the second back-to-back run — asserted, so a one-cycle drift
  fails the regression;
- **209,290 cycles** remains correct as the *cold* path (host writing all
  62,730 ROM words, then one inference): 146,544 + 62,746;
- whole simulation now finishes at 3,558,350 ns = 355,835 cycles, leaving
  **244,165 cycles** of the 600,000-cycle watchdog budget unused, with two
  complete inferences inside it.

Icarus 12.0 and ModelSim/Questa 10.5b produce these numbers independently and
identically.

**Both tiers were proven able to fail before being trusted.** Seven mutations
were injected; all seven were caught:

| Mutation | Caught by |
| --- | --- |
| `dense_engine` start does not clear `out_idx_q` | back-to-back inference — **run 1 still passed** |
| `S_KICK_C1` held two cycles | start-pulse-width check + illegal self-edge |
| `allow(3,3)` removed from the declared legal set | illegal-transition report |
| `NSTAGES=21` (a state the design cannot reach) | unvisited-state check |
| conv `in_ch == 0` reject condition deleted | `REJECT MISSED [conv in_ch == 0]` |
| conv never clears `config_error_o` | recovery check — **all 22 rejects still passed** |
| conv asserts `busy_o` despite the config error | `REJECT UNSAFE` inertness check |

The first and sixth are the ones worth noting. `dense_engine` and
`classifier_argmax` each run once per image, so their start-time state clears
were only ever exercised by reset until the testbench ran two images back to
back — the entire pre-existing suite was blind to that mutation. And the sticky
`config_error_o` mutation left all 22 reject checks passing, which is precisely
why the recovery half of `tb_config_guard` exists: without it that tier would
have been vacuous.

**Counts that changed:** 9 testbenches -> **10**; `scripts/regression_summary.sh`
now counts distinct testbenches (it previously divided a PASS-*line* count by a
hardcoded 8, which would have printed `15/8`) and fails if any of the ten stops
reporting.

**Still needing re-capture:** `results/screenshots/03_icarus_regression.png`
(shows nine testbenches / eight PASS lines) and
`results/screenshots/07_end_to_end_cycles.png` (shows the old 2,028,660 ns
timestamp; the `$finish` timestamp is no longer the right thing to photograph —
screenshot the printed `inference 1: 146544 cycles` line instead).


## Addendum — 2026-08-15: stall invariance and reset interruption

A third tier was added and both simulators re-run from scratch. Logs are
`results/icarus_regression_20260815.log` and
`results/modelsim_regression_20260815.log`.

**New: `tb/tb_robustness.sv` + `tb/stream_hold_check.sv`.** This closes three
items from `docs/VERIFICATION_PLAN.md`'s "Required before tapeout" list: random
output stalls, reset interruption policy, and assertions for stable output
under stall.

Each of `conv2d_engine`, `avg_pool2x2_stream` and `dense_engine` is run three
times over identical operands:

| Run | `out_ready_i` | Checked against |
| --- | --- | --- |
| REF | tied high | the golden vectors (`vectors/expected.hex` and friends) |
| STALL | seeded 16-bit LFSR, ~1-in-4 duty | REF, beat for beat, side-band included |
| ABORT | `rst_ni` asserted mid-stream, then restarted | REF, beat for beat |

Using REF as the oracle for the other two is the design of the tier, not a
shortcut: a functional error that moved REF as well would be caught by the
per-module testbenches, which compare against Python. What is asserted here is
*invariance* — that backpressure and reset cannot change the answer.

Measured under Icarus 12.0:

- `conv2d_engine` — 48 beats, **127 stalled cycles**, abort at beat 20 of 48;
- `avg_pool2x2_stream` — 12 beats, **17 stalled cycles**, abort at beat 5 of 12;
- `dense_engine` — 5 beats, **15 stalled cycles**, abort at beat 2 of 5.

The beat counts and abort points are identical under ModelSim/Questa 10.5b, but
`conv2d_engine` stalls for **115** cycles there rather than 127, and the whole
testbench ends at 33,700 ns rather than 33,830 ns. This is expected, not a
discrepancy to reconcile: the LFSR free-runs from the power-on reset and is
never re-seeded per run, so its phase when a stalled run begins depends on every
cycle that came before it, and the two simulators resolve a `wait` on a
non-blocking-assigned counter a delta apart. The stall *pattern* therefore
differs between the two while the property under test does not — the stalled run
still reproduces the reference beat for beat in both. Only `> 0` is ever
asserted about these counts, as an anti-vacuity guard; the numbers themselves are
reported, not checked. Two independent backpressure patterns agreeing is
stronger evidence than one fixed pattern repeated, so this is left as it is
rather than pinned to a per-run seed.

`stream_hold_check` runs continuously alongside all three and fails the cycle a
producer withdraws `valid` before its beat is accepted, or moves the payload
while stalled. Neither rule was checked anywhere in this project before, and
`dense_engine` had never been stalled at all — `tb_dense_engine` holds
`out_ready_i` high for its whole run.

**Both tiers were proven able to fail.** Seven further mutations were injected
(fourteen across the three campaigns to date); all seven were caught, and all
seven by the specific assertion they target rather than by the watchdog:

| Mutation | Caught by |
| --- | --- |
| conv drops `out_valid_o` while stalled | protocol checker: valid withdrawn |
| conv advances `out_data_o` while stalled | protocol checker: payload changed |
| reset does not clear `busy_o` | abort "went quiet" check |
| reset does not clear `out_valid_o` | abort "went quiet" check — **initially missed** |
| reset corrupts a loaded operand | restarted run diverges from REF |
| backpressure that never actually stalls | stall-cycle vacuity guard |
| abort point moved past the end of the stream | not-busy-at-abort vacuity guard |

The fourth is the one worth recording. It was **missed on the first campaign**,
and the fault was in the test, not the design: with `out_ready_i` tied high,
`out_valid_o` is high for only one cycle per beat, so an abort taken at an
arbitrary moment finds it already low and the `out_valid_o` half of the quiet
check passes without testing anything. The aborts are now pinned to a cycle
where a beat is *announced but not yet accepted*, and the mutation dies
immediately. Two of the seven mutations target the tier's own anti-vacuity
guards for the same reason.

One property this tier pins down that is easy to lose by accident: every engine
re-initialises its full sequencing state from `start_i` rather than relying on
reset, which is what makes a mid-stream abort behaviourally identical to an
idle period. The restart comparison is what stops a future change from quietly
moving that initialisation into the reset branch alone.

**Counts that changed:** 10 testbenches -> **11**, 15 `PASS` lines -> **19**.
`scripts/regression_summary.sh`'s expected total was updated to match and still
fails if any of the eleven stops reporting. `tb_robustness` carries its own
50,000-cycle watchdog; `tb_lenet5_top`'s 600,000-cycle budget is unchanged.
