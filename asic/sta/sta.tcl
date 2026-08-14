# Generic post-synthesis static timing / power analysis for one block.
#
# Driven entirely from the environment so a single script serves every block
# in the PPA sweep. Run it through the OpenROAD binary, which embeds OpenSTA:
#
#   openroad -no_init -exit asic/sta/sta.tcl
#
# Required environment:
#   STA_LIB       Liberty file for the target corner
#   STA_TLEF      technology LEF
#   STA_LEF       standard-cell LEF
#   STA_NETLIST   gate-level netlist to analyse
#   STA_TOP       top module name
#   STA_PERIOD    clock period in ns
#
# Everything the harvesting script needs is printed between @@TAG@@ markers so
# the reports stay human-readable while still being machine-parsable.

set lib     $::env(STA_LIB)
set tlef    $::env(STA_TLEF)
set lef     $::env(STA_LEF)
set netlist $::env(STA_NETLIST)
set top     $::env(STA_TOP)
set period  $::env(STA_PERIOD)

read_liberty $lib
read_lef $tlef
read_lef $lef
read_verilog $netlist
link_design $top

# ---------------------------------------------------------------------------
# Constraints
#
# Built here rather than read from an SDC because the sweep covers blocks with
# different port lists, and two of them are purely combinational. Ports are
# classified by name, which is safe: the RTL uses one naming convention
# throughout (clk_i / rst_ni / *_i / *_o).
# ---------------------------------------------------------------------------
set clk_ports  {}
set rst_ports  {}
set data_ports {}
foreach port [all_inputs] {
    set name [get_full_name $port]
    if {$name eq "clk_i"}  { lappend clk_ports  $port ; continue }
    if {$name eq "rst_ni"} { lappend rst_ports  $port ; continue }
    lappend data_ports $port
}

if {[llength $clk_ports]} {
    create_clock -name core_clk -period $period $clk_ports
} else {
    # Combinational block: a virtual clock plus I/O delays turns the longest
    # input-to-output path into a reportable setup check.
    create_clock -name core_clk -period $period
}
set_clock_uncertainty 0.200 [get_clocks core_clk]

set_input_delay  1.000 -clock core_clk $data_ports
set_output_delay 1.000 -clock core_clk [all_outputs]
if {[llength $rst_ports]} {
    set_false_path -from $rst_ports
}

# An ideal driver and a zero load would make every number optimistic and would
# not survive contact with a real floorplan. Drive the data inputs from a real
# cell and load each output with roughly a handful of standard-cell input pins.
set_driving_cell -lib_cell sky130_fd_sc_hd__inv_2 -pin Y $data_ports
set_load 0.010 [all_outputs]

# Power needs a switching assumption; state it explicitly rather than relying
# on a tool default, so the reported number is reproducible.
if {[llength [info commands set_power_activity]]} {
    set_power_activity -global -activity 0.10 -duty 0.50
}

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
puts "@@DESIGN@@ $top @@PERIOD@@ $period"

puts "@@SETUP_BEGIN@@"
report_checks -path_delay max -digits 4 -group_count 3
puts "@@SETUP_END@@"

puts "@@HOLD_BEGIN@@"
report_checks -path_delay min -digits 4 -group_count 1
puts "@@HOLD_END@@"

puts "@@WNS_BEGIN@@"
report_worst_slack -max -digits 4
report_worst_slack -min -digits 4
report_tns -digits 4
puts "@@WNS_END@@"

puts "@@POWER_BEGIN@@"
report_power -digits 4
puts "@@POWER_END@@"

puts "@@AREA_BEGIN@@"
report_design_area
puts "@@AREA_END@@"

puts "@@CHECK_BEGIN@@"
report_checks -unconstrained -digits 4 -group_count 1
puts "@@CHECK_END@@"

puts "STA_DONE"
