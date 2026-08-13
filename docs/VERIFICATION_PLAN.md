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

## Required before tapeout

- trained-network C1, C3, and C5 layer tests;
- min/max and near-overflow accumulator vectors;
- every shift from 0 through the supported deployment maximum;
- negative ties, positive ties, saturation boundaries, and ReLU boundaries;
- all-valid, all-invalid (error), and extreme legal dimension configurations;
- random output stalls and reset interruption policy;
- assertions for stable output under stall and legal counter ranges;
- functional and code coverage goals;
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
- `lenet5_top` predicted class matching `deploy_forward_int8` on the full
  canonical 32x32x1 input;
- canonical NumPy shapes through F6;
- no elaboration failure.

