# Semi-Custom ASIC Flow

## Deliverable maturity

The RTL and golden-model arithmetic are regression-tested. This package is not
a tapeout database: no SRAM compiler, IO library, DFT insertion, clock-tree
implementation, parasitic extraction, multi-corner sign-off, DRC, or LVS was
run. Those remain correctly out of scope until real place-and-route sign-off.

That said, **real pre-layout numbers now exist** for the storage-free
arithmetic tier, against the open **sky130hd** PDK: gate-level synthesis onto
real standard cells (`yosys`/`abc`) plus static timing and power analysis
(OpenSTA, via the `openroad` binary) for `conv5x5_pe` and its four sibling
leaf blocks. See [`docs/PPA.md`](PPA.md) for the full table, the switching-
activity assumption behind the power numbers, and exactly what pre-layout vs.
post-layout means for each figure quoted there. Final (post-P&R) area,
frequency, power, DRC, and LVS results are still not available — see
"Toolchain notes" in `docs/PPA.md` for why (an embedded-Python version
mismatch blocks this environment's place-and-route stage specifically, not
synthesis or STA).

## Recommended RTL-to-GDS sequence

1. **Freeze the deployment model**
   - train and validate the chosen LeNet-5 variant;
   - calibrate int8 scales on representative data;
   - export weights, int32 biases, shifts, masks, and known-answer vectors.
2. **Close front-end verification**
   - run the supplied bit-exact regression;
   - add full C1/C3/C5 layer vectors from exported weights;
   - run assertions, code/functional coverage, and constrained random stalls;
   - review CDC/RDC and lint results.
3. **Choose the technology**
   - standard-cell Liberty/LEF at all required PVT corners;
   - SRAM macros with timing/power models, LEF/GDS, and simulation models;
   - IO/pad ring, power intent, metal stack, antenna and reliability rules.
4. **Synthesis and DFT**
   - synthesize `conv5x5_pe` first for arithmetic PPA exploration;
   - **status**: `make synth` (`synth-pe` + `synth-pool` + `synth-mac`) now
     also covers `avg_pool2x2_int8` and `dense_row_mac` -- the two new
     arithmetic-tier leaf modules added for the full C1-F6-classifier
     pipeline that have no behavioral storage of their own, following the
     same "synthesize the storage-free PE tier, not the memory-backed
     wrapper" split `conv5x5_pe`/`conv2d_engine` already established. Real
     cell counts are in `results/VERIFICATION_SUMMARY.md`. `avg_pool2x2_stream`,
     `dense_engine`, `classifier_argmax`, and `lenet5_top` remain
     unsynthesized on purpose: each owns a behavioral ROM/scratch array
     (`act_mem`/`wgt_mem`/`bias_mem`/per-layer ROMs), exactly the same
     "verification array, not a real memory" situation `conv2d_engine` is
     already in -- see the SRAM replacement notes in `docs/INTERFACES.md`;
   - integrate the macro-backed feeder and top-level controller;
   - apply the project SDC, resolve unconstrained paths, and inspect inferred
     multiplier architecture;
   - add scan, memory BIST, test clocks, and test-mode timing constraints.
5. **Physical implementation**
   - place SRAMs around the PE/feeder to minimize five-lane bus length;
   - plan power grid and IR-drop margin before placement;
   - place/optimize, build CTS, route, and repair timing/DRV/antenna issues;
   - keep quantization and accumulator control paths out of critical datapaths.
6. **Sign-off**
   - MCMM setup/hold with extracted parasitics;
   - DRC, LVS, ERC, antenna, density, and metal fill;
   - IR drop and EM;
   - gate-level simulation with SDF for reset, scan, and critical interfaces;
   - power analysis from representative post-layout switching activity.

## Constraint starting point

`synth/conv5x5_pe.sdc` defines a 100 MHz clock, 200 ps uncertainty, 1 ns
input/output delay, and an asynchronous-reset false path. These are example
integration assumptions, not an SoC timing budget — replace them with the
real one before integration. `docs/PPA.md` uses the same input/output delay
and uncertainty style (generated per block by `asic/sta/sta.tcl` rather than
hand-written per block) to sweep clock period and report each block's actual
achievable frequency against sky130hd; that sweep is a design-space result,
not a substitute for the SoC-level constraint work below.

At minimum, audit:

- all clocks and generated clocks;
- asynchronous reset deassertion strategy;
- input/output delays at the real chip boundary;
- clock-gating checks;
- false/multicycle paths with written functional justification;
- max transition, capacitance, fanout, and pulse-width limits.

## PPA exploration knobs

| Knob | Area/energy effect | Throughput effect |
|---|---|---|
| 1, 5, or 25 multipliers | roughly increasing arithmetic area | 25, 5, or 1 row-cycle factors |
| int8 versus lower precision | smaller multipliers/SRAM at lower precision | often higher frequency or more lanes |
| weight SRAM word width | wider macro and routing | fewer read cycles |
| clock gating | lowers inactive dynamic power | no ideal functional change |
| sparse channel skipping | reduces C3 activity | reduces active cycles |
| pipelined adder tree | more registers/clock power | improves timing |

The supplied design is the five-multiplier point. It is deliberately modest,
easy to verify, and appropriate for early PPA sweeps.

## Open-source implementation

`asic/openroad/config.mk` targets the reusable PE and defaults to the
`sky130hd` platform used by OpenROAD Flow Scripts (ORFS). Install the matching
ORFS release, run `bash asic/openroad/run_orfs.sh`, and tune die area/density
for that release; the official flow tutorial documents the current commands
and reports. This is the "closest to a real flow" path and is what produced
the `conv5x5_pe` numbers in `docs/PPA.md` — but on a yosys/OpenROAD pairing
where place-and-route currently cannot run (see
`asic/openroad/patches/README.md` for the two version mismatches hit and how
one of them is patched).

Because P&R is blocked here, `asic/sta/run_ppa.sh` covers the rest: it
synthesizes each storage-free leaf block straight to sky130hd cells with
`yosys`/`abc` and drives OpenSTA (through the `openroad` binary, which embeds
it) directly on the mapped netlist, sweeping clock period to find each block's
real achievable frequency. It needs only `yosys` and `openroad` on `PATH` —
no ORFS `make` flow, so no dependency on the broken P&R stage. Both paths read
the same PDK and land in `docs/PPA.md`.

## Commercial implementation

For Cadence Genus/Innovus or Synopsys Design Compiler/ICC2, use the same RTL and
SDC but replace the generic synthesis step with the qualified corporate flow.
Library setup, operating conditions, RC corners, MMMC views, tie/filler/decap
cells, power intent, and sign-off decks are PDK-specific and should not be
guessed in a reusable repository.

