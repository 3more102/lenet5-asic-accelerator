# Contributing

Thanks for looking at this project. It is a verified RTL starting point for a
semi-custom CNN convolution accelerator, so the bar for a change is that the
regression still proves the arithmetic — not just that it compiles.

## Toolchain

Everything in CI is open source:

| Tool | Version used | Purpose |
|---|---|---|
| Python | 3.10+ with NumPy | golden models and vector generation |
| Icarus Verilog | 12.0 | elaboration, lint, and all self-checking testbenches |
| Yosys | 0.33+ | generic synthesis of the arithmetic leaf blocks |

On Debian/Ubuntu:

```bash
sudo apt-get install -y iverilog yosys python3-numpy
```

Siemens ModelSim/Questa is optional. It is the reference for waveform debug and
for the end-to-end cycle count quoted in `docs/ARCHITECTURE.md`, but it is
proprietary, so CI cannot run it. If you change RTL timing, rerun it locally:

```bash
python golden/generate_vectors.py
vsim -do scripts/modelsim.do
```

## Before opening a pull request

```bash
make regression
make synth
```

`make regression` runs, in order: the golden-model unit tests, SystemVerilog
elaboration/lint of the full RTL list, all eight self-checking testbenches, and
the end-to-end demo. Every testbench prints an explicit `PASS <name>` line; a
silent finish is a failure, not a pass.

## The rule that matters: golden model first

No testbench in this repo asserts hand-written expected values. Each one checks
RTL against a Python oracle:

- `golden/lenet5.py` is the **canonical 1998 paper model** — scaled tanh,
  trainable subsampling, RBF output. It is a reference and is deliberately not
  required to agree with the RTL.
- `golden/deploy.py` and `golden/quantized_conv.py` are the **bit-exact int8
  deployment model** the RTL actually implements — ReLU, fixed 2x2 average
  pooling, dense+argmax classifier.

So a datapath change is a two-sided change:

1. update the Python oracle in `golden/`,
2. regenerate vectors with `make vectors`,
3. make the RTL match,
4. commit the regenerated `vectors/` alongside the RTL.

CI regenerates the vectors and fails on any diff against the committed ones, so
a stale `vectors/` directory is caught rather than silently trusted.

## Adding RTL

- Add the file to **both** the `RTL` list in the `Makefile` and the `vlog` list
  in `scripts/modelsim.do`. They are kept deliberately identical; the lint and
  sim targets compile the whole set so an unreferenced module goes unused
  instead of drifting out of the file list.
- Match the surrounding style: `always_comb` / `always_ff`, `logic` only,
  explicit signed widths, no unpacked arrays in datapath code that synthesis
  might infer as memory (see the comment in `rtl/conv5x5_row_mac.sv`).
- Keep RTL technology-independent. No PDK is assumed anywhere in this repo, and
  no vendor primitive should be instantiated.
- If a block is synthesizable at the arithmetic tier, give it a script in
  `synth/` and a target in the `synth` rule. Blocks with behavioural ROM/SRAM
  arrays are intentionally excluded — they need a memory compiler, as described
  in `docs/SEMICUSTOM_FLOW.md`.

## Scope boundary

Please do not add claims the repo cannot support. The shipped vectors are
deterministic random values that verify arithmetic; they are **not trained
MNIST weights**, and no accuracy number should be stated until training,
quantization-aware calibration, and exported per-layer scales exist. Likewise,
area/timing/power figures require real standard-cell libraries and SRAM macros,
which this repository does not assume.

Open work is tracked in `docs/VERIFICATION_PLAN.md` and Section 7 of
`docs/ARCHITECTURE.md`.
