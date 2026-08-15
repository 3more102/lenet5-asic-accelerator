# Verification Summary

Date: 2026-08-07, re-verified eight times (seven on 2026-08-15, one on
2026-08-16) — after the requantize
pipelining change, after adding controller-FSM coverage and the
config-validation reject tier, after adding the stall-invariance and
reset-interruption tier, after adding the operand/dimension extremes tier, after
adding post-synthesis formal equivalence, after extending both formal and
simulation to the sky130hd-mapped netlist, after giving the two
multiply-accumulate blocks a testbench of their own, and after giving the PE's
control logic one. See the eight Addenda at the end. The body below reflects the
first re-verification; **where a later addendum moves a number, the later one is
authoritative** — that means the cycle figures come from the second (measured per
inference rather than read off `$finish`) and the testbench count from the
**eighth** (**15**, not 9, 10, 11, 12 or 14). The fifth and sixth add checks that
are not testbenches at all and are not counted; the sixth re-runs existing
testbenches against gates instead of RTL, which is a second run of an existing
testbench rather than a new one. The seventh and eighth add genuinely new
testbenches — two and one — which is why the count moves.

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

## Addendum — 2026-08-15: the two MAC blocks get a testbench of their own

The addendum above closed by naming its own weakest point: the MAC blocks are
where simulation is thinnest and where formal cannot reach, and block-level
stimulus rather than layer vectors replayed through a wrapper is what would fix
it. This is that fix, and the testbench count moves from **12 to 14** because
these are two genuinely new testbenches rather than existing ones re-run.

`conv5x5_row_mac` and `dense_row_mac` are where every multiplication in the
design happens. Until now the 5-tap MAC was reached only through
`tb_conv5x5_pe` and the 8-lane MAC only through `tb_dense_engine`'s five F6
output values.

**The oracle is not new code.** One lane group of a dense layer with zero bias
*is* a row MAC, so both testbenches take their expected values from
`dense_int8`'s accumulator output — the same model the rest of the suite is
checked against, not a second implementation of the arithmetic being compared.

**The stimulus is built for a lane MAC's failure modes**, not for size:

- Every lane swept across all 256 int8 values on both operands, with the other
  lanes held at a non-zero background whose products have distinct magnitudes.
  Zeroing them would be the worst available choice: with the rest of the row
  zeroed, dropping a lane, duplicating one, or pairing lane *i*'s activation
  with lane *j*'s weight all produce the same sum as a correct design.
- Weights rotated against fixed activations. Rotating *both* rows leaves the
  multiset of products, and therefore the sum, unchanged — a test that passes
  on a mis-wired lane pairing.
- Products cancelling to exactly zero with a distinct magnitude per lane, one
  set anchored on both extreme products `(-128)*(-128) = +16,384` and
  `127*(-128) = -16,256`. The correct answer has no bits set, so any error is
  the whole output.
- Uniform random, plus a mixed distribution pinning half the lanes to extremes
  and leaving the rest small — the shape a quantized layer produces after ReLU.

6,592 and 9,540 cases, reaching the theoretical peaks exactly: +81,920/-81,280
for five lanes, +131,072/-130,048 for eight.

**The first draft of the cancelling class was wrong, and is worth recording.**
It produced five *identical* products summing to -640 — neither cancelling nor
asymmetric, and precisely the symmetric shape `tb_extremes.sv` had already
learned to avoid. Only one vector in 6,592 landed on zero, and it did so by
accident out of the random class. The generator now asserts each cancelling set
sums to zero and contains no repeated or zero product, and each testbench
requires at least six such rows rather than merely "not zero".

**The vacuity guards were proven able to fire.** Each testbench asserts it drove
every lane individually to both int8 extremes. Narrowing the directed sweeps by
two lanes does *not* fail it — 3,000 random rows genuinely do cover every lane,
and the guard reports coverage rather than intent. Narrowing the directed
classes *and* clamping the random class away from the rails fails it with
"only 4/5 lanes saw +127", which is the check working.

**Proven able to fail.** Five RTL mutations per block, each behind a passing
unmutated control: drop a lane from the tree, re-pair a lane's weight, turn a
tree adder into a subtractor, zero-extend instead of sign-extend, tie the low
bit low. **5/5 and 5/5.** One further attempt was thrown out rather than
counted — it was not legal Verilog, and the harness had been scoring the
compile error as a catch. A build break proves nothing about a testbench.

**The controlled experiment the previous addendum could only approximate.**
`make gls` now maps a block once and simulates every testbench that reaches it,
since mapping is the only expensive step. That allows the same netlist to be
mutated identically and driven two different ways:

| Netlist | Driven by | Caught |
|---|---|---|
| `conv5x5_row_mac` (1,833 cells) | `tb_conv5x5_row_mac` (6,592 cases) | **5/5** |
| `conv5x5_row_mac` (1,833 cells) | `tb_conv5x5_pe` (wrapper) | 4/5 |
| `dense_row_mac` (2,984 cells) | `tb_dense_row_mac` (9,540 cases) | **5/5** |
| `dense_row_mac` (2,984 cells) | `tb_dense_engine` (five F6 values) | 2/5 |

Same netlist, same five mutations, same tool, same run. The only variable is the
stimulus, and it is the whole of the difference — which is the claim the
previous addendum made across *different* blocks and could not isolate.

The 2/5 reproduces the previous addendum's measurement exactly, down to which
three mutations survive, so it is a reproducible property of that stimulus
rather than a noisy result. Counting each block once by its strongest driver,
`make gls` now catches **22 of 24** against the 14 of 19 it caught before.

**All five GLS blocks still pass**, at unchanged cell counts — `requantize` 550,
`conv5x5_row_mac` 1,833, `conv5x5_pe` 3,102, `avg_pool2x2_int8` 135,
`dense_row_mac` 2,984 — with the two MAC netlists now checked both pure-gate
against their own testbench and in place inside their RTL wrapper.

**What this does not close.** Two survivors remain. The `avg_pool2x2_int8` one is
covered by `make equiv-mapped`, which catches it instantly. The `conv5x5_pe` one
is not: that block is the *wrapper* — its MAC now has a testbench of its own,
but the bias/accumulate and requantize path around it does not, and it has no
formal cover. Naming it here rather than letting a 22-of-24 headline absorb it.

**Full regression after all of it:** 26 PASS lines across **14** distinct
testbenches, zero failures, under both Icarus Verilog 12.0 and ModelSim ASE
18.1. `scripts/regression_summary.sh` asserts the count of 14, so a testbench
that stops running fails the gate rather than leaving a screen full of green.

## Addendum — 2026-08-16: the PE's control logic, and a survivor that is not a gap

The addendum above closed by naming what it had not closed: the wrapper logic in
`conv5x5_pe`, where the last gate-level survivor lived. This closes it, and the
testbench count moves from **14 to 15**.

`conv5x5_pe` is the row MAC plus the control around it — bias at `first_i`, an
int32 accumulator carried across rows and restarted between pixels,
requantization at `last_i`, an output held under backpressure. Its only
testbench produced **one** output pixel, at **shift 0 with ReLU off**. So the
requantizer inside the PE was never driven at another shift, never saturated,
never clamped; the bias was never negative or large; `first_i` and `last_i` were
never asserted on the same beat; and no pixel ever followed another, leaving the
accumulator restart untested.

`tb_conv5x5_pe_stream.sv` drives **848 pixels / 3,147 rows**: 1 to 8 rows per
pixel, every shift 0..31 with ReLU off and on, both saturation rails, the exact
rounding ties, bias from zero to ±2²⁸, **48 single-beat pixels**, **428 pixels
started while the previous one was still requantizing**, randomized gaps on the
input stream and randomized backpressure on the output (**241 stall cycles**
measured), with `tb/stream_hold_check.sv` policing the output protocol.

**Corners are reached by choosing the accumulator and solving for the bias**
(`bias = target − Σ row sums`), so the ties and rails are driven through genuine
multi-row accumulations rather than one hand-picked row.

**The oracle is checked rather than trusted.** Expected values compose
`dense_int8`'s accumulator per row with `requantize`, in the order the PE's
contract specifies. A single-channel 5×5 convolution is exactly one PE pixel of
five rows, so `conv2d_valid_int8` must agree; the generator asserts that at
eight (shift, ReLU) combinations and raises if it ever does not.

**Proven able to fail, against both testbenches.** Five RTL mutations of the
wrapper — bias never injected, mux arms swapped so the accumulator carries
across pixels, requantize fed the running accumulator instead of the latched
result, shift ignored, output register updated with no result pending:

| Mutation | old `tb_conv5x5_pe` | new `tb_conv5x5_pe_stream` |
|---|---|---|
| P1 bias never injected | caught | caught |
| P2 accumulator carried across pixels | caught | caught |
| P3 requantize fed `accumulator_q` | caught | caught |
| P4 shift ignored | **survived** | **caught** |
| P5 output register updated with no result pending | survived | survived |

P4 is the point: the old testbench drives shift = 0, so a design that ignores
shift entirely is indistinguishable from a correct one.

Worth recording about P1–P3: the old testbench catches them but reports all
three as *"Output was not held correctly under backpressure"*, because its only
value check sits inside the backpressure loop. The mutation harness had been
matching on message text and scored those three as INVALID; it now treats the
testbench's own `$fatal` as the verdict, whatever the wording.

**On the netlist, same netlist and same mutations, only the driver changed:**

| Netlist | Driven by | Caught |
|---|---|---|
| `conv5x5_pe`, 3,102 cells | `tb_conv5x5_pe_stream`, 848 pixels | **4/4** |
| `conv5x5_pe`, 3,102 cells | `tb_conv5x5_pe`, one pixel | 3/4 |

The mutation the streaming testbench catches is exactly the one that survived
the previous addendum. Counting each block once by its strongest driver,
`make gls` is now **23 of 24**.

**P5 survives both testbenches, and it is not a coverage gap.** It drops the
`rq_valid_q` guard on the output register, so `out_data_o` is rewritten on every
ready cycle. `equiv_make` reports `out_data_o` as unproven — correctly, the two
designs really do drive different values there. But they differ only on cycles
where `out_valid_o` is **low**, which no consumer honouring valid/ready samples.
`scripts/prove_pe_output_hold.sh` proves it rather than arguing it: a miter
whose `bad_o` is "the handshakes differ, or the payload differs while valid is
high" is unreachable to depth 20 from reset — with a **cover check** showing an
output beat is reachable inside that window, and a **negative control** (payload
inverted) confirming the miter rejects an observably wrong design.

The datapath is cut away with `cutpoint`, `row_sum` and `quantized` constrained
equal in both copies, which makes it **12,065 SAT variables instead of 591,417**
over the real multiplier arrays — where the same query does not converge, the
same wall unbounded equivalence hits on a mapped MAC. It also makes the result
hold for *any* row MAC and *any* requantizer.

Two harness faults were found and fixed while establishing that, both the same
kind of error — a tool failure read as a verdict. The first negative control used
`if (1'b0)`, so `opt` deleted the nets the `-set` arguments named and yosys
exited non-zero on a **parse** error, which the script reported as "correctly
rejects". And yosys's stderr from the negative run interleaved into the cover
run's terminal output, so an expected failure appeared under the wrong heading.
Both are guarded now.

**Every surviving gate-level mutation in the project is now either caught by
another tier or proven unobservable**: `avg_pool2x2_int8`'s dropped inverter by
`make equiv-mapped`, and this one by the proof above.

**Full regression after all of it:** 27 PASS lines across **15** distinct
testbenches, zero failures, `MAKE_RC=0`, under both Icarus Verilog 12.0 and
ModelSim ASE 18.1. `scripts/regression_summary.sh` asserts the count of 15.

`results/icarus_regression_20260815.log` and
`results/modelsim_regression_20260815.log` now hold **this** run, dated
2026-08-16. The filenames are kept because earlier addenda cite them; the
content is always the latest full regression rather than a snapshot, and git
history holds the earlier versions.
