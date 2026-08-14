`timescale 1ns/1ps
`include "vectors/config.svh"

// Differential test for rtl/requantize.sv against golden/quantized_conv.py.
//
// Every other testbench exercises this module only indirectly, and only on
// the accumulator values the fixed convolution vectors happen to produce at
// shift = 7. This one drives it directly across the cases most likely to be
// wrong -- exact half-way values, both saturation boundaries, the shift = 0
// bypass, int32 extremes -- plus randomized coverage of the whole
// (accumulator, shift, relu) space. Vectors come from
// golden/generate_vectors.py: generate_requantize_vectors().
//
// The module is purely combinational, so this needs no clock: apply inputs,
// let them settle, compare.
module tb_requantize;
    localparam integer COUNT = `TV_RQ_COUNT;

    logic signed [31:0] acc;
    logic        [5:0]  shift;
    logic               relu_en;
    logic signed [7:0]  data_o;

    logic [31:0] acc_vector      [0:COUNT-1];
    logic [7:0]  shift_vector    [0:COUNT-1];
    logic [7:0]  relu_vector     [0:COUNT-1];
    logic [7:0]  expected_vector [0:COUNT-1];

    integer index;
    integer mismatches;
    integer relu_clamped;
    integer saturated_high;
    integer saturated_low;
    logic signed [7:0] expected;

    requantize #(
        .ACC_WIDTH(32),
        .OUT_WIDTH(8)
    ) dut (
        .acc_i     (acc),
        .shift_i   (shift),
        .relu_en_i (relu_en),
        .data_o    (data_o)
    );

    initial begin
        $readmemh("vectors/requant/acc.hex", acc_vector);
        $readmemh("vectors/requant/shift.hex", shift_vector);
        $readmemh("vectors/requant/relu.hex", relu_vector);
        $readmemh("vectors/requant/expected.hex", expected_vector);

        mismatches     = 0;
        relu_clamped   = 0;
        saturated_high = 0;
        saturated_low  = 0;

        for (index = 0; index < COUNT; index = index + 1) begin
            acc     = $signed(acc_vector[index]);
            shift   = shift_vector[index][5:0];
            relu_en = relu_vector[index][0];
            #1;

            expected = $signed(expected_vector[index]);
            if (data_o !== expected) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10) begin
                    $display("MISMATCH [%0d]: acc=%0d shift=%0d relu=%0b -> got %0d, expected %0d",
                             index, acc, shift, relu_en, data_o, expected);
                end
            end

            // Track which interesting behaviours the vector set actually
            // reached, so a silently narrowed generator cannot masquerade as
            // full coverage.
            if (expected == 8'sd127)  saturated_high = saturated_high + 1;
            if (expected == -8'sd128) saturated_low  = saturated_low + 1;
            if (relu_en && acc < 0 && expected == 8'sd0) relu_clamped = relu_clamped + 1;
        end

        if (mismatches != 0) begin
            $fatal(1, "tb_requantize: %0d/%0d cases mismatched the golden model",
                   mismatches, COUNT);
        end

        // A pass with nothing exercised would be a false pass, so require the
        // corners to have actually been hit.
        if (saturated_high == 0 || saturated_low == 0 || relu_clamped == 0) begin
            $fatal(1,
                   "tb_requantize: vectors never reached a corner (sat+ %0d, sat- %0d, relu %0d)",
                   saturated_high, saturated_low, relu_clamped);
        end

        $display("PASS tb_requantize: %0d/%0d cases matched the Python golden model", COUNT, COUNT);
        $display("  corners hit: %0d saturate-high, %0d saturate-low, %0d ReLU-clamped",
                 saturated_high, saturated_low, relu_clamped);
        $finish;
    end
endmodule
