`timescale 1ns/1ps

// Signed power-of-two requantization.
// Rounding rule: nearest integer, with half-way values rounded away from zero.
// Saturation happens after the optional ReLU.
//
// Rounding is done as shift-then-increment rather than the textbook
// add-then-shift. For a non-negative magnitude m and shift s >= 1, letting
// t = m >>> (s-1):
//
//     (m + 2^(s-1)) >>> s  ==  (t >>> 1) + t[0]
//
// (t even -> t/2; t odd -> (t+1)/2; both match the floor of the left side.)
// Both forms need one barrel shift, but the left one also needs a full
// ACC_WIDTH+1 adder, while the right needs only an increment by a single bit
// -- a half-adder chain instead of a full-adder chain. That matters here:
// static timing put this adder's carry chain on the critical path of every
// block containing it, as a ~33-deep run of sky130_fd_sc_hd__maj3_1 cells
// (docs/PPA.md). The block stays purely combinational, so nothing about its
// interface, latency or the pipeline's cycle count changes.
module requantize #(
    parameter integer ACC_WIDTH = 32,
    parameter integer OUT_WIDTH = 8
) (
    input  logic signed [ACC_WIDTH-1:0] acc_i,
    input  logic        [5:0]           shift_i,
    input  logic                        relu_en_i,
    output logic signed [OUT_WIDTH-1:0] data_o
);
    logic signed [ACC_WIDTH:0] acc_ext;
    logic signed [ACC_WIDTH:0] magnitude;
    logic signed [ACC_WIDTH:0] rounded;
    logic signed [ACC_WIDTH:0] pre_round;

    localparam logic signed [ACC_WIDTH:0] MAX_Q =
        ({{ACC_WIDTH{1'b0}}, 1'b1} <<< (OUT_WIDTH-1)) - 1;
    localparam logic signed [ACC_WIDTH:0] MIN_Q =
        -({{ACC_WIDTH{1'b0}}, 1'b1} <<< (OUT_WIDTH-1));

    always_comb begin
        acc_ext   = {acc_i[ACC_WIDTH-1], acc_i};
        magnitude = '0;
        rounded   = acc_ext;
        pre_round = '0;

        if (shift_i != 0) begin
            // Round the magnitude, then reapply the sign, so half-way values
            // round away from zero rather than toward negative infinity.
            magnitude = (acc_ext >= 0) ? acc_ext : -acc_ext;
            pre_round = magnitude >>> (shift_i - 1'b1);
            rounded   = (pre_round >>> 1) + {{ACC_WIDTH{1'b0}}, pre_round[0]};
            if (acc_ext < 0) begin
                rounded = -rounded;
            end
        end

        if (relu_en_i && (rounded < 0)) begin
            data_o = '0;
        end else if (rounded > MAX_Q) begin
            data_o = {1'b0, {(OUT_WIDTH-1){1'b1}}};
        end else if (rounded < MIN_Q) begin
            data_o = {1'b1, {(OUT_WIDTH-1){1'b0}}};
        end else begin
            data_o = rounded[OUT_WIDTH-1:0];
        end
    end
endmodule

