`timescale 1ns/1ps

// Reusable convolution processing element.
//
// The feeder presents one 5-element kernel row per accepted cycle. Assert
// first_i on the first connected row of an output pixel and last_i on the
// final connected row. The PE adds bias_i at first_i, accumulates in int32,
// requantizes at last_i, and holds output data under backpressure.
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
    logic signed [DATA_WIDTH-1:0] quantized;
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
        .acc_i     (next_acc),
        .shift_i   (shift_i),
        .relu_en_i (relu_en_i),
        .data_o    (quantized)
    );

    assign in_ready_o = ~out_valid_o | out_ready_i;
    assign accept     = in_valid_i & in_ready_o;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            accumulator_q <= '0;
            out_valid_o   <= 1'b0;
            out_data_o    <= '0;
        end else begin
            if (out_valid_o && out_ready_i) begin
                out_valid_o <= 1'b0;
            end

            if (accept) begin
                if (last_i) begin
                    out_data_o  <= quantized;
                    out_valid_o <= 1'b1;
                end else begin
                    accumulator_q <= next_acc;
                end
            end
        end
    end
endmodule

