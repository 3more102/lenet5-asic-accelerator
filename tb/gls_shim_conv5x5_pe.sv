`timescale 1ns/1ps

// Gate-level-simulation shim for `conv5x5_pe`. See tb/gls_shim_requantize.sv
// for why these shims exist.
//
// The conv5x5_pe netlist is flattened: `conv5x5_row_mac` and `requantize` are
// synthesized into it, so this one netlist covers all three blocks. That is
// also why `make gls` runs tb_conv5x5_pe against a pure netlist -- no RTL
// submodule survives underneath it.
module conv5x5_pe #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32
) (
    input  logic                             clk_i,
    input  logic                             rst_ni,

    input  logic                             in_valid_i,
    output logic                             in_ready_o,
    input  logic                             first_i,
    input  logic                             last_i,
    input  logic signed [(5*DATA_WIDTH)-1:0] act_row_i,
    input  logic signed [(5*DATA_WIDTH)-1:0] wgt_row_i,
    input  logic signed [ACC_WIDTH-1:0]      bias_i,
    input  logic        [5:0]                shift_i,
    input  logic                             relu_en_i,

    output logic                             out_valid_o,
    input  logic                             out_ready_i,
    output logic signed [DATA_WIDTH-1:0]     out_data_o
);
    initial begin
        if (DATA_WIDTH != 8 || ACC_WIDTH != 32) begin
            $display("FAIL gls_shim_conv5x5_pe: netlist is DATA_WIDTH=8 ACC_WIDTH=32, caller asked for %0d/%0d",
                     DATA_WIDTH, ACC_WIDTH);
            $fatal(1);
        end
    end

    conv5x5_pe_gls u_netlist (
        .clk_i       (clk_i),
        .rst_ni      (rst_ni),
        .in_valid_i  (in_valid_i),
        .in_ready_o  (in_ready_o),
        .first_i     (first_i),
        .last_i      (last_i),
        .act_row_i   (act_row_i),
        .wgt_row_i   (wgt_row_i),
        .bias_i      (bias_i),
        .shift_i     (shift_i),
        .relu_en_i   (relu_en_i),
        .out_valid_o (out_valid_o),
        .out_ready_i (out_ready_i),
        .out_data_o  (out_data_o)
    );
endmodule
