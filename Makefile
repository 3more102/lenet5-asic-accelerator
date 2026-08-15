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

.PHONY: all vectors golden-test demo lint sim-pe sim-requantize sim-c3 sim-engine sim-pool sim-f6 \
	sim-classifier sim-classifier-tie sim-config-guard sim-top regression synth synth-pe synth-pool \
	synth-mac ppa check-ppa orfs clean

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

# tb/fsm_cov.sv is a testbench-side coverage collector, not RTL, so it is
# listed here rather than in $(RTL). It enforces full state and transition
# coverage of lenet5_top's 20-state control FSM and fails the run on any
# unvisited state, unexercised legal edge, or illegal edge.
sim-top: vectors
	$(IVERILOG) -g2012 -Wall -I. -s tb_lenet5_top \
		-o results/tb_top.vvp $(RTL) tb/fsm_cov.sv tb/tb_lenet5_top.sv
	$(VVP) results/tb_top.vvp

regression: golden-test lint sim-pe sim-requantize sim-c3 sim-engine sim-pool sim-f6 \
	sim-classifier sim-classifier-tie sim-config-guard sim-top demo

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

