`timescale 1ns/1ps

// Reusable convolution processing element.
//
// The feeder presents one 5-element kernel row per accepted cycle. Assert
// first_i on the first connected row of an output pixel and last_i on the
// final connected row. The PE adds bias_i at first_i, accumulates in int32,
// requantizes at last_i, and holds output data under backpressure.
//
// Requantization is a second pipeline stage: the accumulator result is
// registered before it enters `requantize`, so the row-MAC adder tree and the
// requantizer's carry chain no longer share one combinational cycle. Without
// that register the two paths serialize and the block runs at the sum of both
// (measured 26.76 ns / 37.4 MHz without it, 16.01 ns / 62.5 MHz with it, for
// +3.8% area -- see docs/PPA.md). Costs one extra cycle of output latency per
// pixel; throughput is unchanged at one kernel row per accepted cycle.
module conv5x5_pe #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,

    input  logic                              in_valid_i,
    output logic                              in_ready_o,
    input  logic                              first_i,
    input  logic                              last_i,
    input  logic signed [(5*DATA_WIDTH)-1:0] act_row_i,
    input  logic signed [(5*DATA_WIDTH)-1:0] wgt_row_i,
    input  logic signed [ACC_WIDTH-1:0]       bias_i,
    input  logic        [5:0]                 shift_i,
    input  logic                              relu_en_i,

    output logic                              out_valid_o,
    input  logic                              out_ready_i,
    output logic signed [DATA_WIDTH-1:0]       out_data_o
);
    logic signed [ACC_WIDTH-1:0] row_sum;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [ACC_WIDTH-1:0] next_acc;
    logic signed [ACC_WIDTH-1:0] result_q;
    logic signed [DATA_WIDTH-1:0] quantized;
    logic rq_valid_q;
    logic rq_ready;
    logic accept;

    conv5x5_row_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_row_mac (
        .act_row_i (act_row_i),
        .wgt_row_i (wgt_row_i),
        .row_sum_o (row_sum)
    );

    always_comb begin
        next_acc = (first_i ? bias_i : accumulator_q) + row_sum;
    end

    requantize #(
        .ACC_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(DATA_WIDTH)
    ) u_requantize (
        .acc_i     (result_q),
        .shift_i   (shift_i),
        .relu_en_i (relu_en_i),
        .data_o    (quantized)
    );

    assign rq_ready   = ~out_valid_o | out_ready_i;
    assign in_ready_o = ~rq_valid_q | rq_ready;
    assign accept     = in_valid_i & in_ready_o;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            accumulator_q <= '0;
            result_q      <= '0;
            rq_valid_q    <= 1'b0;
            out_valid_o   <= 1'b0;
            out_data_o    <= '0;
        end else begin
            if (rq_ready) begin
                out_valid_o <= rq_valid_q;
                if (rq_valid_q) begin
                    out_data_o <= quantized;
                end
                rq_valid_q <= 1'b0;
            end

            if (accept) begin
                if (last_i) begin
                    result_q   <= next_acc;
                    rq_valid_q <= 1'b1;
                end else begin
                    accumulator_q <= next_acc;
                end
            end
        end
    end
endmodule

