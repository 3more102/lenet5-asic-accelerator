PYTHON ?= python3
IVERILOG ?= iverilog
VVP ?= vvp
YOSYS ?= yosys

RTL := \
	rtl/conv5x5_row_mac.sv \
	rtl/requantize.sv \
	rtl/conv5x5_pe.sv \
	rtl/avg_pool2x2_int8.sv \
	rtl/lenet5_c3_connectivity.sv \
	rtl/lenet5_layer_config.sv \
	rtl/conv2d_engine.sv \
	rtl/dense_row_mac.sv \
	rtl/avg_pool2x2_stream.sv \
	rtl/dense_engine.sv \
	rtl/classifier_argmax.sv \
	rtl/lenet5_pool_config.sv \
	rtl/lenet5_dense_config.sv \
	rtl/lenet5_top.sv

.PHONY: all vectors golden-test demo lint sim-pe sim-pe-stream sim-requantize sim-mac-conv \
	sim-mac-dense sim-c3 sim-engine sim-pool sim-f6 \
	sim-classifier sim-classifier-tie sim-config-guard sim-robustness sim-extremes \
	sim-layer-shapes sim-trained sim-top regression synth synth-pe synth-pool synth-mac \
	equiv equiv-pe equiv-pool equiv-mac equiv-mapped equiv-mapped-requantize \
	equiv-mapped-pool gls ppa check-ppa orfs clean

all: regression

vectors:
	PYTHONPATH=. $(PYTHON) golden/generate_vectors.py

golden-test:
	PYTHONPATH=. $(PYTHON) -m unittest -v golden.test_golden

demo:
	PYTHONPATH=. $(PYTHON) golden/run_lenet5_demo.py

# Every sim-<name> target and lint compile the full $(RTL) set (kept in
# sync with scripts/modelsim.do's vlog list) rather than a hand-picked
# subset, so unreferenced modules simply go unused instead of silently
# drifting out of the file list.
lint:
	$(IVERILOG) -g2012 -Wall -tnull -s conv2d_engine $(RTL)
	$(IVERILOG) -g2012 -Wall -tnull -s conv5x5_pe $(RTL)
	$(IVERILOG) -g2012 -Wall -tnull -s lenet5_top $(RTL)

sim-pe:
	$(IVERILOG) -g2012 -Wall -s tb_conv5x5_pe -o results/tb_pe.vvp \
		$(RTL) tb/tb_conv5x5_pe.sv
	$(VVP) results/tb_pe.vvp

# Differential sweep of requantize against the Python oracle: half-way values,
# both saturation boundaries, the shift = 0 bypass and int32 extremes, plus
# randomized coverage. Every other testbench reaches this module only
# indirectly and only at shift = 7.
sim-requantize: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_requantize -o results/tb_requantize.vvp \
		$(RTL) tb/tb_requantize.sv
	$(VVP) results/tb_requantize.vvp

# The two multiply-accumulate blocks, driven directly instead of through a
# wrapper. Both were previously reached only incidentally -- conv5x5_row_mac
# through tb_conv5x5_pe, dense_row_mac through tb_dense_engine's five F6
# output values -- and both are beyond what `make equiv-mapped` can prove, so
# simulation is the only evidence they have. Each vector set sweeps every lane
# across the full int8 range on both operands and asserts it reached the sum's
# extremes, a cancelling case, and both int8 extremes on every lane.
sim-mac-conv: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_conv5x5_row_mac \
		-o results/tb_mac_conv.vvp $(RTL) tb/tb_conv5x5_row_mac.sv
	$(VVP) results/tb_mac_conv.vvp

sim-mac-dense: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_dense_row_mac \
		-o results/tb_mac_dense.vvp $(RTL) tb/tb_dense_row_mac.sv
	$(VVP) results/tb_mac_dense.vvp

# The PE's wrapper logic -- bias at first_i, the accumulator carried across
# rows and restarted between pixels, requantization at last_i, output held
# under backpressure. sim-pe drives one pixel at shift 0 with ReLU off, which
# leaves most of that unexecuted; this drives 848 pixels of 1-8 rows across
# every shift with both ReLU settings, both saturation rails, randomized input
# gaps and randomized output backpressure.
sim-pe-stream: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_conv5x5_pe_stream \
		-o results/tb_pe_stream.vvp $(RTL) tb/stream_hold_check.sv tb/tb_conv5x5_pe_stream.sv
	$(VVP) results/tb_pe_stream.vvp

sim-c3:
	$(IVERILOG) -g2012 -Wall -s tb_lenet5_c3_connectivity \
		-o results/tb_c3.vvp $(RTL) tb/tb_lenet5_c3_connectivity.sv
	$(VVP) results/tb_c3.vvp

sim-engine: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_conv2d_engine \
		-o results/tb_engine.vvp $(RTL) tb/tb_conv2d_engine.sv
	$(VVP) results/tb_engine.vvp

sim-pool: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_avg_pool2x2_stream \
		-o results/tb_pool.vvp $(RTL) tb/tb_avg_pool2x2_stream.sv
	$(VVP) results/tb_pool.vvp

sim-f6: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_dense_engine \
		-o results/tb_f6.vvp $(RTL) tb/tb_dense_engine.sv
	$(VVP) results/tb_f6.vvp

sim-classifier: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_classifier_argmax \
		-o results/tb_classifier.vvp $(RTL) tb/tb_classifier_argmax.sv
	$(VVP) results/tb_classifier.vvp

sim-classifier-tie:
	$(IVERILOG) -g2012 -Wall -s tb_classifier_argmax_tie \
		-o results/tb_classifier_tie.vvp $(RTL) tb/tb_classifier_argmax_tie.sv
	$(VVP) results/tb_classifier_tie.vvp

# Drives all 22 config-validation reject conditions across the four engines
# that implement one, then proves each engine still accepts a legal config
# afterwards. Every other testbench only ever asserts config_error_o stays
# low, so without this target the whole reject path ships unexecuted.
sim-config-guard:
	$(IVERILOG) -g2012 -Wall -s tb_config_guard \
		-o results/tb_config_guard.vvp $(RTL) tb/tb_config_guard.sv
	$(VVP) results/tb_config_guard.vvp

sim-robustness: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_robustness \
		-o results/tb_robustness.vvp $(RTL) tb/stream_hold_check.sv tb/tb_robustness.sv
	$(VVP) results/tb_robustness.vvp

# The extremes vectors are the only ones that drive -128/+127 at every operand
# position and the largest/smallest layer shapes the engines accept; see the
# header of tb/tb_extremes.sv for why the shifts differ from deployment.
sim-extremes: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_extremes \
		-o results/tb_extremes.vvp $(RTL) tb/tb_extremes.sv
	$(VVP) results/tb_extremes.vvp

# conv2d_engine and dense_engine reconfigured across the real network's own
# C1/C3/C5/F6/classifier shapes, each layer's beat stream checked against
# deploy_forward_int8 directly rather than only the final predicted class.
# See the header of tb/tb_layer_shapes.sv.
sim-layer-shapes: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_layer_shapes \
		-o results/tb_layer_shapes.vvp $(RTL) tb/tb_layer_shapes.sv
	$(VVP) results/tb_layer_shapes.vvp

# The only tier driven by a trained network on real MNIST digits, and the
# only one that instantiates lenet5_top at non-default SHIFT_* parameters --
# a per-layer calibrated network does not decode correctly at a flat shift=7.
sim-trained: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_trained_mnist \
		-o results/tb_trained_mnist.vvp $(RTL) tb/tb_trained_mnist.sv
	$(VVP) results/tb_trained_mnist.vvp

# tb/fsm_cov.sv is a testbench-side coverage collector, not RTL, so it is
# listed here rather than in $(RTL). It enforces full state and transition
# coverage of lenet5_top's 20-state control FSM and fails the run on any
# unvisited state, unexercised legal edge, or illegal edge.
sim-top: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_lenet5_top \
		-o results/tb_top.vvp $(RTL) tb/fsm_cov.sv tb/tb_lenet5_top.sv
	$(VVP) results/tb_top.vvp

regression: golden-test lint sim-pe sim-pe-stream sim-requantize sim-mac-conv sim-mac-dense \
	sim-c3 sim-engine sim-pool sim-f6 sim-classifier sim-classifier-tie sim-config-guard \
	sim-robustness sim-extremes sim-layer-shapes sim-trained sim-top demo

# synth-pe is the original target name kept as an alias so existing callers
# of `make synth-pe`/`synth/yosys.ys` keep working; `synth` now means
# "everything currently synthesizable" (arithmetic-tier leaf modules with
# no behavioral ROM/SRAM arrays -- see docs/SEMICUSTOM_FLOW.md for what's
# still deferred).
synth: synth-pe synth-pool synth-mac

synth-pe:
	$(YOSYS) -s synth/yosys.ys

synth-pool:
	$(YOSYS) -s synth/avg_pool2x2_int8.ys

synth-mac:
	$(YOSYS) -s synth/dense_row_mac.ys

# Post-synthesis formal equivalence: prove by SAT that each generic netlist
# computes the same function as the RTL it came from, for all inputs and all
# reachable states. Every other check in this repo is RTL simulation against the
# Python model and says nothing about what synthesis emitted. Each script ends
# in `equiv_status -assert`, so a single unproven equivalence point fails the
# target rather than printing a report nobody reads.
#
# Deliberately NOT part of `regression`: conv5x5_pe alone takes ~200s and
# dense_row_mac ~300s, which would triple regression wall-clock for a check that
# only needs re-running when the RTL or the synthesis flow changes. CI runs it as
# its own job. See docs/VERIFICATION_PLAN.md for what it does and does not prove.
equiv: equiv-pe equiv-pool equiv-mac

equiv-pe:
	$(YOSYS) -s synth/equiv_conv5x5_pe.ys

equiv-pool:
	$(YOSYS) -s synth/equiv_avg_pool2x2_int8.ys

equiv-mac:
	$(YOSYS) -s synth/equiv_dense_row_mac.ys

# The same proof, but against the sky130hd-mapped netlist docs/PPA.md measures
# rather than the generic one -- real standard cells, real abc restructuring.
# Only two blocks: unbounded equivalence over a mapped multiply-accumulate does
# not converge with a plain SAT solver (conv5x5_row_mac ran 20 minutes without
# finishing), so the MAC blocks' netlists are covered by `make gls` instead.
# Needs the sky130hd liberty from ORFS, so unlike `make equiv` this cannot run
# in CI; the committed evidence is results/gls_equiv_mapped_20260815.log.
equiv-mapped: equiv-mapped-requantize equiv-mapped-pool

equiv-mapped-requantize:
	$(YOSYS) -s synth/equiv_mapped_requantize.ys

equiv-mapped-pool:
	$(YOSYS) -s synth/equiv_mapped_avg_pool2x2_int8.ys

# Gate-level simulation: re-run the existing testbenches against the
# sky130hd-mapped netlists instead of the RTL, some pure-gate and some as mixed
# RTL/gate runs. This is how the blocks `equiv-mapped` cannot prove get their
# netlists checked. Same golden vectors, same PASS lines, gates underneath.
# Needs yosys, iverilog and the sky130hd liberty; see scripts/run_gls.sh for the
# flow and docs/VERIFICATION_PLAN.md for its scope.
#
# A block may list several testbenches; each netlist is built once and reused,
# because mapping is the only expensive step here. Budget ~40 minutes, of which
# dense_row_mac is ~35 and abc is essentially all of it (2,099 s of a 2,102 s
# yosys run under 0.68; far quicker under 0.52). The simulations themselves take
# seconds. Pass block names to run a subset:
#   bash scripts/run_gls.sh requantize conv5x5_row_mac avg_pool2x2_int8
gls: vectors
	bash scripts/run_gls.sh

# Real sky130hd area/timing/power: yosys+abc technology mapping followed by
# OpenSTA (via the openroad binary, which embeds it) on every storage-free
# leaf block, swept across clock periods. Needs `yosys` and `openroad` on
# PATH; does not need a working ORFS place-and-route stage (see
# asic/openroad/patches/README.md for why that stage is currently blocked
# here). Results land in asic/sta/results/ppa_summary.csv and are written up
# in docs/PPA.md.
ppa:
	bash asic/sta/run_ppa.sh

# Fail if docs/PPA.md's headline table drifts from the raw STA results in
# asic/sta/results/ppa_summary.csv. Those figures are transcribed by hand, so
# re-running `make ppa` without updating the document would otherwise leave it
# quietly claiming numbers the tool never produced. Needs only Python, so
# unlike `make ppa` itself this runs anywhere -- including CI.
check-ppa:
	python3 scripts/check_ppa_consistency.py

# Full ORFS RTL-to-GDS synthesis stage for conv5x5_pe (requires an ORFS
# install; set ORFS_ROOT if it is not at /root/OpenROAD-flow-scripts, and see
# asic/openroad/patches/ if synthesis aborts on `stat -hierarchy`).
orfs:
	bash asic/openroad/run_orfs.sh

clean:
	rm -f results/*.vvp results/*.vcd results/*.json results/*.log

