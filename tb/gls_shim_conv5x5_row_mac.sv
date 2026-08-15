`timescale 1ns/1ps

// Gate-level-simulation shim for `conv5x5_row_mac`. See
// tb/gls_shim_requantize.sv for why these shims exist.
//
// This block is now driven two ways against the same netlist, because mapping
// is what costs the time and simulating is nearly free: tb_conv5x5_row_mac
// runs it pure-gate across every lane and both int8 extremes, and
// tb_conv5x5_pe runs the same netlist inside the RTL `conv5x5_pe` wrapper, so
// the netlist is checked both for its arithmetic and for still fitting where
// it has to fit.
//
// `conv5x5_pe` forwards its own DATA_WIDTH and ACC_WIDTH down, and the netlist
// is fixed at 8 and 32. The lane count is not a parameter here -- five is in
// the module name -- so unlike dense_row_mac there is no lane width to check.
module conv5x5_row_mac #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32
) (
    input  logic signed [(5*DATA_WIDTH)-1:0] act_row_i,
    input  logic signed [(5*DATA_WIDTH)-1:0] wgt_row_i,
    output logic signed [ACC_WIDTH-1:0]       row_sum_o
);
    initial begin
        if (DATA_WIDTH != 8 || ACC_WIDTH != 32) begin
            $display("FAIL gls_shim_conv5x5_row_mac: netlist is DATA_WIDTH=8 ACC_WIDTH=32, caller asked for %0d/%0d",
                     DATA_WIDTH, ACC_WIDTH);
            $fatal(1);
        end
    end

    conv5x5_row_mac_gls u_netlist (
        .act_row_i (act_row_i),
        .wgt_row_i (wgt_row_i),
        .row_sum_o (row_sum_o)
    );
endmodule
