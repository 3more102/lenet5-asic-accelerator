`timescale 1ns/1ps

// Gate-level-simulation shim for `dense_row_mac`. See
// tb/gls_shim_requantize.sv for why these shims exist.
//
// Like `avg_pool2x2_int8`, this block has no testbench of its own: it is the
// combinational lane MAC inside `dense_engine`, whose weight/activation/bias
// stores are behavioral arrays. `make gls` therefore runs tb_dense_engine as
// a mixed RTL/gate simulation, with the mapped netlist supplying the 8-lane
// multiply-accumulate that the F6 layer's every output element goes through.
//
// LANES matters most here: `dense_engine` forwards its own LANES parameter
// down, and the netlist is a fixed 8-lane structure. An 8-lane check that
// silently ran against a 16-lane request would be worse than no check.
module dense_row_mac #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32,
    parameter integer LANES      = 8
) (
    input  logic signed [(LANES*DATA_WIDTH)-1:0] act_lane_i,
    input  logic signed [(LANES*DATA_WIDTH)-1:0] wgt_lane_i,
    output logic signed [ACC_WIDTH-1:0]          lane_sum_o
);
    initial begin
        if (DATA_WIDTH != 8 || ACC_WIDTH != 32 || LANES != 8) begin
            $display("FAIL gls_shim_dense_row_mac: netlist is DATA_WIDTH=8 ACC_WIDTH=32 LANES=8, caller asked for %0d/%0d/%0d",
                     DATA_WIDTH, ACC_WIDTH, LANES);
            $fatal(1);
        end
    end

    dense_row_mac_gls u_netlist (
        .act_lane_i (act_lane_i),
        .wgt_lane_i (wgt_lane_i),
        .lane_sum_o (lane_sum_o)
    );
endmodule
