# Verification Summary

Date: 2026-08-07, re-verified six times on 2026-08-15 — after the requantize
pipelining change, after adding controller-FSM coverage and the
config-validation reject tier, after adding the stall-invariance and
reset-interruption tier, after adding the operand/dimension extremes tier, after
adding post-synthesis formal equivalence, and after extending both formal and
simulation to the sky130hd-mapped netlist. See the six Addenda at the end.
The body below reflects the first re-verification; **where a later addendum
moves a number, the later one is authoritative** — that means the cycle figures
come from the second (measured per inference rather than read off `$finish`) and
the testbench count from the fourth (**12**, not 9, 10 or 11). The fifth and
sixth add checks that are not testbenches at all and are not counted among the
twelve; the sixth re-runs three of the twelve against gates instead of RTL,
which is a second run of an existing testbench and not a thirteenth.

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

---

## Fourth addendum -- 2026-08-15: operand and dimension extremes

`tb/tb_extremes.sv` closes the "min/max and near-overflow accumulator vectors"
and "extreme legal dimension configurations" items from
`docs/VERIFICATION_PLAN.md`.

**The gap it closes.** Every other engine-level vector set in this repo draws
activations from `[-32, 32)` and weights from `[-16, 16)`. The largest
accumulator any of them produced was about 39,000 -- roughly 0.002% of the
32-bit range it is stored in -- and the int8 extremes -128 and +127 were never
driven at any operand position. -128 is exactly where a signed multiplier is
most likely to be wrong, because its negation is not representable in int8.

| Run | Shape | Shift | What it establishes |
|---|---|---|---|
| conv magnitude | 16x5x5 -> 2x1x1, 400 MACs/output | 16 | peak accumulator 7,502,600 carried without wrapping |
| conv cancelling | same operands, weights summing to zero | 0 | a single wrong product is visible, not just a wrong magnitude |
| conv minimum | 1x5x5 -> 1x1x1, one beat | 12 | smallest legal shape; output position asserted as 0/0/0 |
| dense | 120 -> 4 at `MAX_IN_LEN` | 16 | `out_acc_o` bit-exact against the model, peak 2,950,780 |

**Two passes, because one is not enough.** The magnitude run proves the
accumulator holds 7.5 million without wrapping, but at shift 16 one output LSB
is worth 65,536, so it cannot resolve a small per-product error -- measured, not
assumed: a multiplier mutation that reads -128 as -127 moves that accumulator by
5,120 and the output not at all. The cancelling run fixes the resolution by
making the weights sum to zero so the bias *is* the accumulator, read out at
shift 0 where any error of 1 or more fails the comparison.

**The cancelling weights had to be made asymmetric.** The first arrangement used
three equal contiguous blocks, which put an equal share of every weight group in
each of the five row columns -- so a mutation confined to one column cancelled
exactly as neatly as the products did, and it survived the whole regression a
second time. The weights are now assigned per row column
(`-128 + 127 + 1 - 1 + 1 = 0`), with no column summing to zero, so a bug in any
one of the five products shifts the result. Leaving a column at zero would
reopen the same hole from the other side: a wrong product times a zero weight is
still zero.

**A signal that had never been checked.** `golden/generate_vectors.py` has always
written `vectors/accumulator.hex` and `vectors/f6/accumulator.hex`, and no
testbench ever read either. `tb_dense_engine.sv` connects `out_acc_o` and then
ignores it. That is not an incidental output: `classifier_argmax` decides the
predicted class from `out_acc_o` rather than the requantized score, precisely so
two classes that both saturate cannot tie -- the "non-obvious correctness call"
recorded in `docs/ARCHITECTURE.md`. A wrong-but-self-consistent accumulator
would have passed every check in this project. It is now compared against the
golden model beat for beat, and a mutation that corrupts only `out_acc_o` while
leaving `out_data_o` correct is caught by nothing else in the regression.

The conv-side file is a different case, and worth recording rather than
quietly leaving generated: `conv2d_engine` has **no** accumulator port at all
(`out_data_o`, `out_channel_o`, `out_y_o`, `out_x_o` and the status flags are
the whole output side), so `vectors/accumulator.hex` cannot be compared against
anything by any testbench. It is dead data by construction, not an oversight
this tier fixes. Only `dense_engine` exposes the raw accumulator, because
`classifier_argmax` is the one consumer that needs it.

**Accumulator headroom.** The margin quoted is **286x** -- 7,502,600 through
`conv2d_engine` against the 2,147,483,647 an int32 holds -- and it is taken
from `conv2d_engine` because it is the larger of the two peaks: headroom is set
by the worst case, and quoting `dense_engine`'s 2,950,780 instead would report
727x for the same measurement.

The two peaks are known to different standards, which the phrase "measured
peak" would blur. `dense_engine`'s 2,950,780 is *measured*: the testbench
watches `out_acc_o`, compares every beat against the model, and asserts the
observed peak equals the expected one, so the RTL demonstrably carried that
value. `conv2d_engine`'s 7,502,600 is the model's figure, and with no
accumulator port the RTL can only be constrained through the requantized
output -- to within one output LSB, 65,536 at shift 16. That is loose, but it
is far tighter than the failure being excluded: a 32-bit wrap displaces the
accumulator by 2^32 and the output by 65,536 LSBs, which cannot hide inside a
bit-exact comparison. So *no wrap* is established for both engines; *the exact
peak* is established only for dense. That the width is sufficient for every
legal configuration remains an argument rather than a proof, but it is no
longer an untested one.

**Proven able to fail.** Nine mutations were injected (twenty-three across the
four campaigns to date); all nine were caught. Eight fired the assertion they
target; the ninth is a deliberate control -- the same -128 multiplier bug scored
against the magnitude pass alone, which does *not* catch it, which is the
measurement that justifies the cancelling pass existing at all. Two of the nine
attack the tier's own anti-vacuity guards: one makes the generator stop emitting
int8 extremes, the other reverts it to the deployment shift so every output
saturates. Both are caught, so the testbench cannot quietly stop testing what it
claims to test.

**Counts that changed:** 11 testbenches -> **12**, 19 `PASS` lines -> **24**.
`scripts/regression_summary.sh`'s expected total was updated to match and still
fails if any of the twelve stops reporting. `tb_extremes` carries its own
50,000-cycle watchdog and finishes in about 3,100 cycles.

---

## Fifth addendum -- 2026-08-15: post-synthesis equivalence (`make equiv`)

Closes the "post-synthesis equivalence checking" item from
`docs/VERIFICATION_PLAN.md`, partially -- see the limits below. Run under Yosys
0.68; the 0.52 quoted earlier in this document is what the WSL2 toolchain
carried on 2026-08-07 and is left as the historical record of that pass.

**The gap it closes.** All twelve testbenches are RTL simulation. Each compares
the RTL against the Python golden model on whatever stimulus it drives, and none
of them looks at what synthesis produced. Nothing in this project had ever
checked the netlist. `make equiv` proves, for each synthesizable block, that the
generic netlist computes the same function as the RTL it was elaborated from --
by SAT on the combinational equivalence points and induction on the sequential
ones, over all inputs and all reachable states rather than over a vector set.
That is a different *kind* of evidence from everything above it: the twelve
testbenches say "these outputs matched on these inputs", this says "no input
exists that separates them".

| Block | Equivalence points | Unproven | Runtime |
|---|---|---|---|
| `conv5x5_pe` (with `conv5x5_row_mac`, `requantize`) | 694 | 0 | ~196 s |
| `dense_row_mac` | 768 | 0 | ~300 s |
| `avg_pool2x2_int8` | 130 | 0 | seconds |
| **total** | **1,592** | **0** | ~8 min |

**Proven able to fail.** The flow ends in `equiv_status -assert`, which is the
only thing separating a check from a report -- without it Yosys prints the
unproven count and exits 0. Five mutations were injected into the *gate* side
only, which is exactly the failure mode being guarded against (a netlist that
does not implement its RTL): rounding constant 2 -> 1 on the positive branch and
again on the negative branch, one of the four samples dropped from the sum,
`>>> 2` weakened to `>>> 1`, and the sign-extension replaced by zero-extension.
All five were caught, each leaving a different number of unproven points
(14, 16, 10, 16, 10). The unmutated control returns 0 with 130/130 proven --
without that control the 5/5 would be equally consistent with a harness that
always fails.

**Two flow traps, both of which fail loudly rather than quietly.**
`conv5x5_pe` resets asynchronously, and `equiv_simple` has no SAT model for an
async flip-flop: without `async2sync` on *both* sides the run dies with "No SAT
model available for async FF cell". And `equiv_simple -short` before the full
`equiv_simple` discharges most points against shortened input cones, which is
the difference between 196 s and several minutes on `conv5x5_pe`.

**What this does not prove**, since the phrase "formally verified" invites more
than was done. Only the three synthesizable leaf blocks are covered:
`conv2d_engine`, `avg_pool2x2_stream`, `dense_engine`, `classifier_argmax` and
`lenet5_top` hold behavioural ROM/scratch arrays, are not synthesized at all,
and therefore have no netlist to compare against. Only the *generic* netlist is
covered -- the sky130hd/ABC-mapped netlist behind `docs/PPA.md` comes from a
different flow and is not checked. And equivalence is a statement about
function, not timing: it says nothing about setup/hold, SDF back-annotation, or
scan insertion.

`make equiv` is deliberately not part of `make regression`. At roughly eight
minutes it would triple regression wall-clock for a check that only needs
re-running when the RTL or the synthesis flow changes; CI runs it as its own
job so a failure there reads differently from a simulation failure.

## Sixth addendum -- 2026-08-15: the sky130hd-mapped netlist (`make equiv-mapped`, `make gls`)

Narrows the same tapeout item further. The fifth addendum checked the *generic*
netlist; this one checks the netlist `docs/PPA.md` actually measures -- real
sky130hd standard cells, `dfflibmap` plus `abc -liberty`. Run under Yosys 0.68
and Icarus Verilog 12.0.

**Why the generic netlist was not enough.** Generic `techmap` preserves most of
the RTL's structure and all of its internal net names. Technology mapping does
neither: `abc` rebuilds the logic against a real cell library, and the result is
`maj3`/`xnor3`/`a211oi` cells with no counterpart in the source. If a synthesis
bug were going to appear anywhere, it would appear there -- and that netlist,
the one carrying the area and timing numbers this project quotes, had never been
checked by anything.

**Two methods, because neither reaches every block.**

Formal (`make equiv-mapped`) -- the same `equiv_*` flow, mapped netlist as the
gate side:

| Block | Equivalence points | Unproven | Runtime |
|---|---|---|---|
| `requantize` | 8 | 0 | ~2 s |
| `avg_pool2x2_int8` | 8 | 0 | ~2 s |

Eight points where the generic proof had 694. `equiv_make` pairs signals by
name and `abc` leaves no internal names to pair, so every point here is a full
input-cone SAT problem instead of a cheap match against a neighbouring node.
Fewer points, harder proof. For `avg_pool2x2_int8` those 8 points settle all
2^32 input patterns -- a stronger statement than any simulation in this repo
makes about any block.

**Why not the other three.** All three contain a wide multiply-accumulate, and
unbounded equivalence over a *mapped* one is a classic hard SAT instance:
mapping destroys the structural correspondence between the two adder trees and
leaves the solver proving a 32-bit product-sum from scratch. What was actually
measured: `conv5x5_row_mac` — the 25-tap MAC itself — sat in `equiv_simple` for
714 s without discharging a single equivalence point, `equiv_struct` having
reported "Nothing to merge" beforehand. `conv5x5_pe` embeds that MAC; an earlier
run of it, without the `equiv_struct` passes, had proven its 8 output bits and
was still grinding through the 32-bit `accumulator_q` when it was stopped at
600 s. `dense_row_mac` was never run to completion — `abc` alone takes 2,099 s
on it under Yosys 0.68, before equivalence begins — so its exclusion rests on it
being the same structure rather than on a measurement, and this document should
not pretend otherwise. Arithmetic-aware datapath matching is precisely what commercial LEC
has and `equiv_simple` does not: a limitation of the tool, not a doubt about
those blocks, so they are covered the other way.

Simulation (`make gls`) -- `scripts/run_gls.sh` maps each block with the same
flow `asic/sta/run_ppa.sh` uses (same liberty, same `synth -flatten`, same
`abc -liberty -D 10000`) and re-runs the *existing* testbenches against the
netlist instead of the RTL. Same golden vectors, same PASS lines:

| Block | Testbench | Mode | Cells | Result | Runtime |
|---|---|---|---|---|---|
| `requantize` | `tb_requantize` (5,504 cases) | pure gate | 550 | PASS | seconds |
| `conv5x5_pe` (+ `conv5x5_row_mac`, `requantize`) | `tb_conv5x5_pe` | pure gate | 3,102 | PASS | ~10 s |
| `avg_pool2x2_int8` | `tb_avg_pool2x2_stream` | mixed RTL/gate | 135 | PASS | seconds |
| `dense_row_mac` | `tb_dense_engine` (F6 layer) | mixed RTL/gate | 2,984 | PASS | 35 min |

`dense_row_mac` is 35 of those 36 minutes, and effectively all of it is `abc`:
2,099 s of a 2,102 s Yosys run, with the simulation itself taking seconds. The
same block maps far faster under Yosys 0.52 (see `asic/sta/results/`), so this
is a tool-version cost, not something about the design -- and it is why
`make gls` stays out of `make regression`.

The mixed runs exist because `avg_pool2x2_int8` and `dense_row_mac` have no
testbench of their own: the mapped leaf sits inside its RTL wrapper and the
wrapper's testbench drives it. `tb/gls_shim_*.sv` makes the substitution
possible -- a gate netlist has no parameters, every wrapper instantiates these
blocks *with* parameter overrides, so each shim re-presents the RTL's
parameterized interface and `$fatal`s if a caller ever asks for a configuration
the netlist was not synthesized at. That check is the point of the shim; without
it a testbench could start overriding `ACC_WIDTH` and keep passing against a
netlist that is still 32 bits wide.

**Proven able to fail.** Mutations go into the netlist *after* mapping and
*before* the check, which is exactly the failure being guarded against -- a
netlist that does not implement its RTL. Five per block: drop an inverter, AND
becomes OR, OR becomes AND, tie an output bit low, invert a gate input.
Every campaign ran a passing unmutated control first; without it, "all caught"
is also the score of a harness that fails on everything.

| Tier | Block | Caught |
|---|---|---|
| `equiv-mapped` | `requantize` | 5/5 |
| `equiv-mapped` | `avg_pool2x2_int8` | 5/5 |
| `gls` | `requantize` | 5/5 |
| `gls` | `avg_pool2x2_int8` | 4/5 |
| `gls` | `conv5x5_pe` | 3/4 (one had no matching line in that netlist) |
| `gls` | `dense_row_mac` | 2/5 |

**The five `gls` survivors are the most useful result here**, and the score is
the least interesting part of them. A mutation survives when the testbench's
stimulus never drives that node into an observable difference, so what this
measures is the *stimulus*, not the netlist -- and it degrades precisely where
the stimulus is thinnest.

`dense_row_mac` is the extreme case at 2/5, and the reason is not subtle:
`tb_dense_engine` checks five F6 output values, and five vectors against a
2,984-cell 8-lane MAC leave most of it untouched. The honest reading is that
`make gls` establishes this netlist computes the F6 layer correctly and not much
more. The `avg_pool2x2_int8` survivor is a dropped inverter that the twelve
golden pooling outputs never distinguish -- and that `make equiv-mapped` catches
instantly, because 8 SAT points cover every input pattern; that is the two tiers
doing exactly what having two tiers is for. The `conv5x5_pe` survivor is an
OR-to-AND swap on a node `tb_conv5x5_pe`'s directed stimulus never drives both
ways, and that block has no formal cover, so it stands as a real gap rather than
something this addendum papers over.

The pattern is worth stating rather than smoothing away: the two blocks formal
can reach are the ones simulation already covers well, and the MAC blocks --
the ones formal cannot reach -- are exactly where simulation is weakest. Nothing
in this tier fixes that. Block-level constrained-random stimulus, instead of
replaying layer vectors through a wrapper, is what would.

**What this does not cover.** Not the vendor's cell models: Yosys reads each
cell's liberty `function` back as logic so the netlist runs without one, which
is the same source of truth `abc` mapped against and therefore the netlist's
real function -- but X-propagation detail, timing checks and UDP internals live
only in a behavioural model and are not exercised. Not the same tool version as
`docs/PPA.md`, whose numbers came from Yosys 0.52 while this runs on 0.68, so
the netlists have the same shape rather than identical cell counts (135 here
against that CSV's 128 for `avg_pool2x2_int8`). Still function and not timing --
zero-delay gates, no SDF, no back-annotation. And neither target runs in CI:
both need the sky130hd liberty from an ORFS install, which the CI runner does
not have, so the committed evidence is `results/gls_20260815.log` rather than a
job badge.
