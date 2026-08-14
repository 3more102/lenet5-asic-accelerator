# Timing constraints for conv5x5_pe.
#
# Written against the OpenSTA command set, which is what both OpenSTA and
# OpenROAD implement. Note that the Synopsys collection commands
# (remove_from_collection and friends) are NOT part of that set -- an earlier
# revision of this file used remove_from_collection to drop clk_i/rst_ni from
# all_inputs, which loads in Design Compiler but aborts OpenROAD with
# `invalid command name "remove_from_collection"`. The explicit loop below is
# the portable spelling and behaves identically.

create_clock -name core_clk -period 10.000 [get_ports clk_i]
set_clock_uncertainty 0.200 [get_clocks core_clk]

# Data ports = every input except the clock and the reset. The reset is
# constrained as a false path below, and the clock is not a data port at all.
set data_inputs {}
foreach port [all_inputs] {
    set name [get_full_name $port]
    if {$name ne "clk_i" && $name ne "rst_ni"} {
        lappend data_inputs $port
    }
}

set_input_delay  1.000 -clock core_clk $data_inputs
set_output_delay 1.000 -clock core_clk [all_outputs]
set_false_path -from [get_ports rst_ni]

unset -nocomplain data_inputs port name
