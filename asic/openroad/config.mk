# OpenROAD Flow Scripts starter configuration for the arithmetic PE.
# PLATFORM and floorplan values are examples and must match your installation.

PROJECT_HOME := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)

export DESIGN_NAME = conv5x5_pe
export PLATFORM = sky130hd

export VERILOG_FILES = \
	$(PROJECT_HOME)/rtl/conv5x5_row_mac.sv \
	$(PROJECT_HOME)/rtl/requantize.sv \
	$(PROJECT_HOME)/rtl/conv5x5_pe.sv

export SDC_FILE = $(PROJECT_HOME)/synth/conv5x5_pe.sdc

export DIE_AREA = 0 0 350 350
export CORE_AREA = 10 10 340 340
export CORE_UTILIZATION = 35
export PLACE_DENSITY = 0.45

