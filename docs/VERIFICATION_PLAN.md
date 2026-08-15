# Verification Plan

## Implemented checks

| Feature | Golden/reference | RTL test |
|---|---|---|
| Signed 5-tap row MAC | hand calculation | `tb_conv5x5_pe.sv` |
| Bias and multi-row accumulation | hand calculation | `tb_conv5x5_pe.sv` |
| Output backpressure/hold | protocol expectation | PE and engine tests |
| Round-nearest-away-from-zero | NumPy | Python unit tests + engine comparison |
| Saturation and optional ReLU | NumPy | Python unit tests + generated vectors |
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

## Required before tapeout

- trained-network C1, C3, and C5 layer tests;
- min/max and near-overflow accumulator vectors;
- every shift from 0 through the supported deployment maximum;
- negative ties, positive ties, saturation boundaries, and ReLU boundaries;
- extreme *legal* dimension configurations (the invalid ones are covered:
  `tb_config_guard.sv` drives all 22 reject conditions across the four engines
  that implement config validation, checks each is inert, and proves the error
  clears on the next legal config);
- legal counter ranges (random output stalls, stable-output-under-stall
  assertions and reset interruption policy are now covered -- see the
  robustness note above);
- functional and code coverage goals beyond the top-level controller
  (`tb/fsm_cov.sv` enforces 20/20 states and 33/33 transitions on
  `lenet5_top`'s FSM every run; the five engines sequence with nested counters
  rather than an enumerated state register, so they have no equivalent
  structural target and remain covered behaviourally by their oracles);
- post-synthesis equivalence checking;
- scan/MBIST verification;
- gate-level reset and SDF simulations;
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
- `lenet5_top` predicted class matching `deploy_forward_int8` on the full
  canonical 32x32x1 input, and the same class again on a second back-to-back
  inference with no reset and no weight reload;
- `lenet5_top` control FSM reporting 20/20 states and 33/33 transitions;
- canonical NumPy shapes through F6;
- no elaboration failure.

