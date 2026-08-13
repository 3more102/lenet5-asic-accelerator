`timescale 1ns/1ps

// Signed power-of-two requantization.
// Rounding rule: nearest integer, with half-way values rounded away from zero.
// Saturation happens after the optional ReLU.
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
    logic signed [ACC_WIDTH:0] rounding_offset;

    localparam logic signed [ACC_WIDTH:0] MAX_Q =
        ({{ACC_WIDTH{1'b0}}, 1'b1} <<< (OUT_WIDTH-1)) - 1;
    localparam logic signed [ACC_WIDTH:0] MIN_Q =
        -({{ACC_WIDTH{1'b0}}, 1'b1} <<< (OUT_WIDTH-1));

    always_comb begin
        acc_ext         = {acc_i[ACC_WIDTH-1], acc_i};
        magnitude       = '0;
        rounded         = acc_ext;
        rounding_offset = '0;

        if (shift_i != 0) begin
            rounding_offset = ({{ACC_WIDTH{1'b0}}, 1'b1} <<< (shift_i - 1'b1));
            if (acc_ext >= 0) begin
                rounded = (acc_ext + rounding_offset) >>> shift_i;
            end else begin
                magnitude = -acc_ext;
                rounded   = -((magnitude + rounding_offset) >>> shift_i);
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

