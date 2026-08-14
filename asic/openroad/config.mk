# OpenROAD Flow Scripts configuration for the arithmetic PE.
#
# This is a real, executed configuration -- not a template. Run it with
# `bash asic/openroad/run_orfs.sh` (see that script for the ORFS location it
# expects). Measured results land in asic/openroad/results/ and are quoted in
# docs/PPA.md; do not edit those numbers by hand.
#
# Scope: conv5x5_pe is the storage-free arithmetic tier -- the same split
# docs/SEMICUSTOM_FLOW.md already draws between the PE and the memory-backed
# conv2d_engine wrapper. Blocks that own behavioral arrays are not placed and
# routed here because their arrays are verification scaffolding, not real
# macros.

PROJECT_HOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)

export DESIGN_NAME      = conv5x5_pe
export DESIGN_NICKNAME  = conv5x5_pe
export PLATFORM         = sky130hd

export VERILOG_FILES = \
	$(PROJECT_HOME)/rtl/conv5x5_row_mac.sv \
	$(PROJECT_HOME)/rtl/requantize.sv \
	$(PROJECT_HOME)/rtl/conv5x5_pe.sv

export SDC_FILE = $(PROJECT_HOME)/synth/conv5x5_pe.sdc

# Let the floorplanner size the die from utilization rather than asserting a
# die area up front. The previous hand-picked 350x350 um box was a guess that
# had never been through placement, so it carried no evidence either way.
export CORE_UTILIZATION = 40
export PLACE_DENSITY    = 0.55
export CORE_MARGIN      = 2

# The PE is a MAC tree; let synthesis pick the adder architecture rather than
# forcing the platform adder map, which ORFS notes can hurt small datapaths.
export ADDER_MAP_FILE :=
