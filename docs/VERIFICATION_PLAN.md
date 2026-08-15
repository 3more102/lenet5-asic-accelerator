# Verification Plan

## Implemented checks

| Feature | Golden/reference | RTL test |
|---|---|---|
| Signed 5-tap row MAC | `dense_int8` accumulator output | `tb_conv5x5_row_mac.sv` (6,592 cases) + `tb_conv5x5_pe.sv` |
| Signed 8-lane dense MAC | same | `tb_dense_row_mac.sv` (9,540 cases) + `tb_dense_engine.sv` |
| Every MAC lane swept across the full int8 range, both operands | same | `tb_conv5x5_row_mac.sv`, `tb_dense_row_mac.sv` |
| Largest-magnitude int8 product (-128 x -128) in every lane at once | same | same |
| Lane pairing: activation *i* must meet weight *i* | same (weights rotated against fixed activations) | same |
| Bias and multi-row accumulation | hand calculation | `tb_conv5x5_pe.sv` |
| Output backpressure/hold | protocol expectation | PE and engine tests |
| Round-nearest-away-from-zero | NumPy | Python unit tests + engine comparison |
| Saturation and optional ReLU | NumPy | Python unit tests + generated vectors |
| Every shift 0..31, each with ReLU off and on | NumPy `requantize` | `tb_requantize.sv` (5,504 cases) |
| Rounding ties either side of zero, and +-1 around each | same | `tb_requantize.sv` |
| Both saturation boundaries, approached from both sides | same | `tb_requantize.sv` |
| Sparse input-channel skipping | NumPy mask | engine comparison |
| C3 exact connectivity | 1998 Table I | Python + dedicated RTL test |
| Runtime H/W/C configuration | generated config | engine comparison |
| Output order and coordinates | Python flattening | engine scoreboard |
| Canonical layer shapes | 1998 architecture | full NumPy forward test |
| Canonical parameter total | layer table | exact 60,000 assertion |
| Dense (fully-connected) layer | NumPy `dense_int8` | `tb_dense_engine.sv` (F6 config) |
| S2/S4 average-pool streaming | NumPy `avg_pool2x2_int8` | `tb_avg_pool2x2_stream.sv` |
| Output classifier (dense+argmax) | NumPy `argmax_classifier` | `tb_classifier_argmax.sv` + `tb_classifier_argmax_tie.sv` |
| Full C1->S2->C3->S4->C5->F6->classifier pipeline | `golden/deploy.py:deploy_forward_int8` | `tb_lenet5_top.sv` |
| Config validation reject path (22 conditions) | RTL guard conditions, enumerated | `tb_config_guard.sv` |
| Re-runnability: back-to-back inference, no reset | `deploy_forward_int8` (same image twice) | `tb_lenet5_top.sv` |
| Control-FSM state and transition coverage | declared legal edge set | `tb/fsm_cov.sv` in `tb_lenet5_top.sv` |
| Stall invariance under pseudorandom backpressure | the engine's own unstalled run | `tb_robustness.sv` |
| valid/ready hold and payload stability | AXI-style stream protocol rules | `tb/stream_hold_check.sv`, instantiated in `tb_robustness.sv` |
| Reset interruption mid-stream, and restart | the engine's own uninterrupted run | `tb_robustness.sv` |
| int8 operand extremes (-128/+127) at every position | `conv2d_valid_int8`, `dense_int8` | `tb_extremes.sv` |
| Near-maximum accumulator magnitude, and no wrap | same | `tb_extremes.sv` |
| Raw pre-requantization accumulator (`out_acc_o`) | `dense_int8` accumulator output | `tb_extremes.sv` |
| Largest and smallest legal layer dimensions | same | `tb_extremes.sv` |
| Generic netlist computes the same function as the RTL | the RTL itself, by SAT | `synth/equiv_*.ys` (`make equiv`) |
| sky130hd-mapped netlist computes the same function as the RTL | the RTL itself, by SAT | `synth/equiv_mapped_*.ys` (`make equiv-mapped`) |
| sky130hd-mapped netlist against the golden vectors | Python model, via the existing testbenches | `scripts/run_gls.sh` (`make gls`) |

The main engine test generates activations, weights, biases, a sparse mask, raw
accumulators, and expected int8 outputs from a deterministic seed. RTL output is
checked cycle-by-cycle while periodic output stalls are injected. The
`avg_pool2x2_stream`/`dense_engine` testbenches follow the same
generated-vector, cycle-by-cycle-scoreboard pattern; `classifier_argmax` and
`lenet5_top` check only the final decision (class index), since neither
module exposes intermediate per-stage data externally.

**Classifier decision rule (normative, see `docs/INTERFACES.md`):**
`argmax_classifier` (golden) and `classifier_argmax` (RTL) both compare raw
pre-shift/pre-saturate accumulators, not requantized int8 scores, and both
break ties toward the lowest class index (`np.argmax`'s default in the golden
model; a strict `>` comparator in the RTL). `test_argmax_classifier_
avoids_saturation_misclassification` in `golden/test_golden.py` and
`tb_classifier_argmax_tie.sv` are the regressions that pin this down.

**Robustness (`tb_robustness.sv`):** each of `conv2d_engine`,
`avg_pool2x2_stream` and `dense_engine` is run three times over identical
operands -- unstalled, with pseudorandom backpressure from a seeded LFSR, and
with `rst_ni` asserted mid-stream and the operation restarted. The unstalled
run is checked against the golden vectors; the other two must reproduce it
beat for beat, side-band included. `tb/stream_hold_check.sv` runs continuously
alongside all three and fails the cycle a producer withdraws `valid_o` before
its beat is accepted, or advances the payload while stalled.

Two properties this pins down that are easy to lose by accident. First, every
engine re-initialises its full sequencing state from `start_i` rather than
relying on reset, which is what makes a mid-stream abort behaviourally
identical to an idle period; the restart comparison is what stops a future
change from quietly moving that initialisation into the reset branch alone.
Second, the aborts are deliberately taken while a beat is *announced but not
yet accepted* -- with `out_ready_i` tied high `out_valid_o` is high for only
one cycle per beat, so an abort taken at an arbitrary moment finds it already
low and the `out_valid_o` half of the quiet check never tests anything. That
was not hypothetical: a mutation deleting `out_valid_o` from
`conv2d_engine`'s reset branch survived the entire regression until the abort
was pinned to that state.

**Operand and dimension extremes (`tb_extremes.sv`):** every other engine-level
vector set in this repo draws activations from `[-32, 32)` and weights from
`[-16, 16)`, so no accumulator ever exceeded about 39,000 of a 32-bit range and
the int8 extremes -128 and +127 were never driven at any operand position.
`tb_extremes.sv` runs `conv2d_engine` at `MAX_IN_CH` (16 channels x 25 taps =
400 MACs per output, the C5 cost and the most the engine accepts) and
`dense_engine` at `MAX_IN_LEN`, with -128 and +127 at every position.

Three things make this test say something rather than merely run:

1. **Shift is 16, not the deployment 6/7.** At shift 7 an accumulator of
   several million requantizes far past the int8 rails and every output pins to
   +-127, insensitive to the accumulator beneath it -- the arithmetic could be
   wrong by millions and the comparison would still hold. The testbench asserts
   that no expected output sits at a rail, so this cannot silently regress.
2. **A second pass with weights that cancel to exactly zero, at shift 0.** The
   magnitude pass proves the accumulator carries 7.5 million without wrapping,
   but one output LSB is worth 65,536 there, so it cannot resolve a small
   per-product error. In the cancelling pass the output *is* the accumulator,
   so a single wrong product moves it. The cancelling weights are arranged
   across the five row columns (`-128 + 127 + 1 - 1 + 1 = 0`) rather than in
   equal blocks: an earlier blocked arrangement cancelled the *bug* as neatly
   as it cancelled the products, and a multiplier mutation survived the whole
   regression until the columns were made asymmetric.
3. **`out_acc_o` is compared against the golden model.** It never was before.
   `golden/generate_vectors.py` had always written `vectors/f6/accumulator.hex`
   and no testbench read it; `tb_dense_engine.sv` connects `out_acc_o` and
   ignores it. That signal is what `classifier_argmax` votes on -- deliberately,
   so two saturating classes cannot tie -- so a wrong-but-self-consistent
   accumulator would have passed every check in the project.

The measured peak is 2,950,780 on `dense_engine`'s `out_acc_o` and 7,502,600
through `conv2d_engine`, against the 2,147,483,647 a 32-bit accumulator holds.
The margin quoted is the one from the *larger* of the two -- **286x**, from
`conv2d_engine` -- because headroom is set by the worst case, not the average
of the cases measured. The width is now exercised rather than assumed, though
the argument that no legal configuration can exceed it remains an argument,
not a proof.

**The two MAC blocks, driven directly (`tb_conv5x5_row_mac.sv`,
`tb_dense_row_mac.sv`):** `conv5x5_row_mac` and `dense_row_mac` are where every
multiplication in the design happens, and until these existed neither had a
testbench of its own. They were reached only through a wrapper — the 5-tap MAC
through `tb_conv5x5_pe`, the 8-lane MAC through `tb_dense_engine`'s five F6
output values. The gate-level tier put a number on how thin that is: mutation
scores of 3/4 and 2/5 where blocks with their own testbench scored 5/5. They are
also the two blocks unbounded equivalence cannot reach, so simulation is the
only evidence they have.

Both testbenches take their expected values from `dense_int8`'s accumulator
output rather than a second implementation of the arithmetic: one lane group of
a dense layer with zero bias *is* a row MAC, so the oracle is the same model the
rest of the suite is checked against. The stimulus is built to make a lane MAC's
specific failure modes observable rather than to be large:

1. **Every lane swept across all 256 int8 values, on both operands**, with the
   other lanes held at a non-zero background. Zeroing them would be the worst
   available choice: with the rest of the row zeroed, a design that dropped a
   lane, duplicated one, or paired lane *i*'s activation with lane *j*'s weight
   would produce the same sum as a correct one. The background products
   (87, -155, -259, 451, 559, 799, 1007, -1357) have distinct magnitudes, so
   removing any single lane lands on a different value than removing any other.
2. **Weights rotated against fixed activations.** Rotating *both* rows leaves
   the multiset of products unchanged and therefore the sum unchanged — a test
   that passes on a mis-wired lane pairing. Rotating the weights alone re-pairs
   every lane, so lane *i*'s activation must meet lane *i*'s weight.
3. **Products that cancel to exactly zero, with a distinct magnitude per lane.**
   The correct answer has no bits set, so any error is the whole output. One
   set anchors on both extreme products, `(-128)*(-128) = +16,384` and
   `127*(-128) = -16,256`. The generator asserts each set sums to zero and has
   no repeated or zero product, for the reason `tb_extremes.sv` learned the hard
   way: a symmetric cancellation cancels the bug as neatly as the products.
4. **Uniform random, plus a mixed distribution** that pins half the lanes to
   extremes and leaves the rest small — the shape a real quantized layer
   produces after ReLU, which uniform sampling almost never generates.

The peaks reached are the theoretical ones: +81,920 and -81,280 for five lanes,
+131,072 and -130,048 for eight. Each testbench asserts it reached both, reached
at least six cancelling rows, and drove **every lane individually** to both int8
extremes — a run that quietly stopped covering a lane fails rather than passes.

**Post-synthesis equivalence (`synth/equiv_*.ys`, `make equiv`):** everything
above is RTL simulation. It compares the RTL against the Python model on
whatever stimulus a testbench drives, and says nothing about what synthesis
emitted. `make equiv` closes that by proving, for each synthesizable block, that
the generic netlist computes the same function as the RTL it was elaborated
from -- by SAT on the combinational points and induction on the sequential ones,
over all inputs and all reachable states rather than over a vector set.

`equiv_make` pairs the two designs' signals, `equiv_simple` discharges points by
SAT, `equiv_induct` proves the rest inductively, and `equiv_status -assert`
exits non-zero if a single point is left unproven. That assert is what makes it
a check: without it the flow prints a report and returns success regardless.

| Block | Equivalence points | Runtime |
|---|---|---|
| `conv5x5_pe` (with `conv5x5_row_mac`, `requantize`) | 694 | ~196 s |
| `dense_row_mac` | 768 | ~300 s |
| `avg_pool2x2_int8` | 130 | seconds |

What it does **not** prove, stated plainly:

- **Only the three synthesizable leaf blocks.** `conv2d_engine`,
  `avg_pool2x2_stream`, `dense_engine`, `classifier_argmax` and `lenet5_top`
  hold behavioural ROM/scratch arrays and are not synthesized at all yet, so
  there is no netlist of them to compare against.
- **Only the *generic* netlist.** This is the `synth/*.ys` flow — Yosys
  `techmap` with no real cell library. The sky130hd/ABC-mapped netlist behind
  `docs/PPA.md` is a *different* netlist produced by a different flow; it is
  covered separately, by `make equiv-mapped` and `make gls` below.
- **Function, not timing.** Equivalence says the two compute the same values;
  it says nothing about setup/hold, and nothing about anything a real library,
  SDF back-annotation or scan insertion would introduce.

Two flow details that are easy to get wrong, both recorded in the scripts:
`async2sync` must run on **both** sides or `equiv_simple` dies with "No SAT
model available for async FF cell" on `conv5x5_pe`'s asynchronous reset; and
`equiv_simple -short` before the full `equiv_simple` discharges most points
cheaply, which is the difference between 196 s and minutes on `conv5x5_pe`.

**The sky130hd-mapped netlist (`make equiv-mapped`, `make gls`):** the check
above stops at the generic netlist. The netlist `docs/PPA.md` reports area,
timing and power for is a different one — `dfflibmap` plus `abc -liberty`
mapping onto real sky130hd standard cells — and until this tier existed, nothing
checked it. That gap mattered more than the generic one: technology mapping is
by far the more aggressive transformation, and the netlist it produces is built
from `maj3`/`xnor3`/`a211oi` cells that have no counterpart in the RTL at all.

Two methods cover it, because neither alone reaches every block.

**Formal, where it converges (`make equiv-mapped`).** Same `equiv_*` flow as
above, with the mapped netlist as the gate side.

| Block | Equivalence points | Runtime | Result |
|---|---|---|---|
| `requantize` | 8 | ~2 s | proven |
| `avg_pool2x2_int8` | 8 | ~2 s | proven |

Eight points, not 694. `equiv_make` pairs signals by name; generic mapping
leaves internal names intact, while `abc` leaves nothing but the ports. So each
of these points is a full input-cone SAT problem rather than a cheap match
against a neighbouring node — fewer points here means a harder proof, not a
weaker one. For `avg_pool2x2_int8`, 8 points is a complete statement about all
2^32 input patterns, which no simulation in this repo comes close to.

**Why only two blocks.** The other three all contain a wide multiply-accumulate,
and unbounded equivalence over a *mapped* one is a classic hard SAT instance:
technology mapping destroys the structural correspondence between the two adder
trees, so the solver is left proving a 32-bit product-sum from scratch.
Measured, not assumed: `conv5x5_row_mac` — the 25-tap MAC itself — spent 714 s
inside `equiv_simple` without discharging a single equivalence point, with
`equiv_struct` reporting "Nothing to merge" beforehand. `conv5x5_pe` embeds that
same MAC; an earlier run of it (without the `equiv_struct` passes) had proven
the 8 output bits and was grinding through the 32-bit `accumulator_q` when it
was stopped at 600 s. `dense_row_mac` was not run to completion at all —
`abc` alone takes upwards of 18 minutes on it under Yosys 0.68 — so its
exclusion here rests on it being the same structure, not on a measurement.

Arithmetic-aware datapath matching is exactly what commercial LEC has and
`equiv_simple` does not. This is a limit of the tool, not a suspicion about
those blocks — so they are covered the other way.

**Simulation, everywhere else (`make gls`).** `scripts/run_gls.sh` maps each
block with the same flow `asic/sta/run_ppa.sh` uses — same liberty, same
`synth -flatten`, same `abc -liberty` at the same 10 ns target — then re-runs
the *existing* testbenches against the netlist instead of the RTL. Same golden
vectors, same PASS lines, gates underneath.

A block may be driven by more than one testbench. Mapping is what costs the time
— `abc` on `dense_row_mac` is ~35 minutes, the simulations are seconds — so once
a netlist exists, everything that reaches it is worth running.

| Block | Testbench | Mode | Cells | Result | Runtime |
|---|---|---|---|---|---|
| `requantize` | `tb_requantize` (5,504 cases) | pure gate | 550 | PASS | seconds |
| `conv5x5_row_mac` | `tb_conv5x5_row_mac` (6,592 cases) | pure gate | 1,833 | PASS | seconds |
| `conv5x5_row_mac` | `tb_conv5x5_pe` | mixed RTL/gate | 1,833 | PASS | seconds |
| `conv5x5_pe` (+`conv5x5_row_mac`, `requantize`) | `tb_conv5x5_pe` | pure gate | 3,102 | PASS | ~10 s |
| `avg_pool2x2_int8` | `tb_avg_pool2x2_stream` | mixed RTL/gate | 135 | PASS | seconds |
| `dense_row_mac` | `tb_dense_row_mac` (9,540 cases) | pure gate | 2,984 | PASS | seconds |
| `dense_row_mac` | `tb_dense_engine` (F6 layer) | mixed RTL/gate | 2,984 | PASS | seconds |

`dense_row_mac` dominates the wall-clock, and all of it is `abc`: 2,099 s of a
2,102 s Yosys run. The same block maps in a fraction of that under Yosys 0.52
(`asic/sta/results/`), so this is a tool-version cost rather than anything about
the design. It is why `make gls` is not part of `make regression`.

The *mixed* runs put the mapped leaf inside its RTL wrapper and let the
wrapper's testbench drive it. For `avg_pool2x2_int8` that is the only option —
its stream interface lives in the wrapper. For the two MAC blocks it is now a
second check rather than the only one: each is run pure-gate against its own
testbench *and* in place inside its wrapper, so the netlist is checked both for
its arithmetic and for still fitting where it has to fit.

`tb/gls_shim_*.sv` is what makes the substitution work — a gate netlist has no
parameters, but every wrapper instantiates these blocks with parameter
overrides, so each shim presents the RTL's parameterized interface and
`$fatal`s if a caller ever asks for a configuration the netlist was not
synthesized at.

What this tier does **not** cover:

- **Not the vendor's cell models.** Yosys reads each sky130hd cell's liberty
  `function` back as logic so the netlist simulates without a vendor model.
  That is the same source of truth `abc` mapped against, so the *function* is
  the netlist's — but anything living only in a behavioural model (X-propagation
  detail, timing checks, UDP internals) is not exercised.
- **Not the same tool version as `docs/PPA.md`.** Those numbers came from Yosys
  0.52; this runs on 0.68, and `abc` maps a little differently now — 135 cells
  against the CSV's 128 for `avg_pool2x2_int8`, and so on. Same flow, same
  liberty, same target; a netlist of the same shape rather than bit-identical.
- **Still function, not timing.** No SDF, no back-annotation, zero-delay gates.
- **No CI.** Both targets need the sky130hd liberty from an ORFS install, which
  the CI runner does not have, so neither runs there. The committed evidence is
  `results/gls_20260815.log`.

Both tiers were proven able to fail, by mutating the netlist *after* mapping and
*before* the check — which is the failure being guarded against, a netlist that
does not implement its RTL. Five mutations per block: drop an inverter, AND
becomes OR, OR becomes AND, tie an output bit low, invert a gate input. Each
campaign ran a passing unmutated control first — without it, "all caught" would
also be the score of a harness that fails on everything.

| Tier | Netlist | Driven by | Caught |
|---|---|---|---|
| `equiv-mapped` | `requantize` | — (SAT) | 5/5 |
| `equiv-mapped` | `avg_pool2x2_int8` | — (SAT) | 5/5 |
| `gls` | `requantize` | `tb_requantize` | 5/5 |
| `gls` | `conv5x5_row_mac` | `tb_conv5x5_row_mac` | **5/5** |
| `gls` | `conv5x5_row_mac` | `tb_conv5x5_pe` | 4/5 |
| `gls` | `conv5x5_pe` | `tb_conv5x5_pe` | 3/4 (one had no matching line in that netlist) |
| `gls` | `avg_pool2x2_int8` | `tb_avg_pool2x2_stream` | 4/5 |
| `gls` | `dense_row_mac` | `tb_dense_row_mac` | **5/5** |
| `gls` | `dense_row_mac` | `tb_dense_engine` | 2/5 |

Counting each block once, by its strongest driver, `gls` now catches **22 of 24**
against the 14 of 19 it caught when every MAC was reached through a wrapper.

**The two rows that share a netlist are the point of the table.** A mutation
survives when the testbench's stimulus never drives that node into an observable
difference, so the score measures the *stimulus*, not the netlist. Until now
that claim rested on comparing different blocks, which cannot separate "thin
stimulus" from "harder block". These two pairs hold the netlist, the mutations,
the tool and the day fixed and change only the driver:

| Netlist | Driven by | Caught |
|---|---|---|
| `conv5x5_row_mac`, 1,833 cells | `tb_conv5x5_row_mac`, 6,592 cases | **5/5** |
| `conv5x5_row_mac`, 1,833 cells | `tb_conv5x5_pe`, wrapper | 4/5 |
| `dense_row_mac`, 2,984 cells | `tb_dense_row_mac`, 9,540 cases | **5/5** |
| `dense_row_mac`, 2,984 cells | `tb_dense_engine`, five F6 values | 2/5 |

The 2/5 reproduces the earlier measurement exactly, down to which three
mutations survive. Nothing about the netlist changed; five vectors against a
2,984-cell 8-lane MAC simply leave most of it undriven.

Two survivors remain, and neither is closed by this tier:

- The `avg_pool2x2_int8` survivor is a dropped inverter that
  `tb_avg_pool2x2_stream`'s twelve golden outputs never distinguish — and that
  `make equiv-mapped` catches immediately, since 8 SAT points cover every input
  pattern. That is exactly what having two tiers is for.
- The `conv5x5_pe` survivor is an OR-to-AND swap deep inside the PE, on a node
  `tb_conv5x5_pe`'s directed stimulus never drives both ways. `conv5x5_pe` is
  the wrapper — its MAC now has its own testbench, but the bias/accumulate and
  requantize path around it does not — and it has no formal cover. It stands as
  a real gap rather than one another tier closes.

The earlier version of this section ended by naming block-level stimulus as what
would close the MAC blocks. That is now done and measured. What it did not close
is the wrapper logic in `conv5x5_pe`, which is where the remaining survivor
lives.

## Required before tapeout

- trained-network C1, C3, and C5 layer tests;
- extreme *legal* dimension configurations beyond the largest and smallest
  (`tb_config_guard.sv` covers the invalid ones -- all 22 reject conditions
  across the four engines that implement config validation, each proven inert
  and clearing on the next legal config -- and `tb_extremes.sv` now covers both
  ends of the legal range, but not the interior);
- legal counter ranges (random output stalls, stable-output-under-stall
  assertions and reset interruption policy are now covered -- see the
  robustness note above);
- functional and code coverage goals beyond the top-level controller
  (`tb/fsm_cov.sv` enforces 20/20 states and 33/33 transitions on
  `lenet5_top`'s FSM every run; the five engines sequence with nested counters
  rather than an enumerated state register, so they have no equivalent
  structural target and remain covered behaviourally by their oracles);
- post-synthesis equivalence checking **of the remaining blocks, and of the
  mapped MAC blocks** (`make equiv` proves RTL == generic netlist for the three
  synthesizable leaf blocks, and `make equiv-mapped` proves RTL == sky130hd
  netlist for `requantize` and `avg_pool2x2_int8` -- see the equivalence notes
  above. The five ROM/scratch-array modules still have no netlist at all, and
  the mapped MAC blocks are beyond `equiv_simple`. Those two now hold
  *thorough* simulation evidence rather than incidental -- their own
  testbenches sweep every lane and catch 5/5 gate-level mutations each -- but
  simulation on chosen inputs is still not a proof over all of them);
- scan/MBIST verification;
- **SDF** gate-level simulations, and gate-level reset simulation (`make gls`
  now runs every mapped leaf block against the golden vectors, but with
  zero-delay gates and no back-annotation, so nothing here checks timing);
- SRAM model replacement tests;
- calibrated per-layer quantization scales (the deployment model currently
  uses a fixed shift=7 for every layer -- see `docs/ARCHITECTURE.md` Section 7);
- OpenROAD/place-and-route extension to the newly-synthesizable
  `avg_pool2x2_int8`/`dense_row_mac` (only `conv5x5_pe` has an OpenROAD
  config today); Yosys synthesis of `avg_pool2x2_stream`/`dense_engine`/
  `classifier_argmax`/`lenet5_top` remains blocked on real SRAM macros
  replacing their behavioral ROM/scratch arrays (see `docs/SEMICUSTOM_FLOW.md`).

## Reproducibility

Run `make regression` (Icarus Verilog -- confirmed working via a WSL2 Ubuntu
toolchain: Yosys 0.52, Icarus Verilog 12.0, GTKWave 3.3.126, `apt install
yosys iverilog gtkwave make python3-numpy`) or `scripts/run_modelsim.ps1`
(ModelSim/Questa; Windows-native, no WSL needed). Both are exercised and kept
green -- see `results/icarus_regression_20260807.log` and
`results/modelsim_run.log`. `make synth` needs the same WSL/Yosys path.
A passing run must show:

- fifteen Python unit tests passing (`python -m unittest golden.test_golden -v`);
- PE test passing;
- C3 table test passing with 60 connections;
- `conv2d_engine` outputs matching the Python golden model;
- `avg_pool2x2_stream` outputs matching `avg_pool2x2_int8`;
- `dense_engine` (F6 configuration) outputs matching `dense_int8`;
- `classifier_argmax` predicted class matching `argmax_classifier`, and the
  tie-break case resolving to the lowest index;
- all 22 config-validation reject conditions flagged and inert, and all four
  engines accepting a legal config afterwards;
- all three streaming engines reproducing their unstalled output beat for beat
  under pseudorandom backpressure and after a mid-stream reset, with the
  protocol checker reporting no withdrawn `valid` and no payload movement
  while stalled;
- `conv2d_engine` and `dense_engine` bit-exact with -128 and +127 at every
  operand position, at both the largest and smallest legal layer shapes, and
  `dense_engine`'s raw `out_acc_o` matching the golden model beat for beat;
- `lenet5_top` predicted class matching `deploy_forward_int8` on the full
  canonical 32x32x1 input, and the same class again on a second back-to-back
  inference with no reset and no weight reload;
- `lenet5_top` control FSM reporting 20/20 states and 33/33 transitions;
- canonical NumPy shapes through F6;
- no elaboration failure.

