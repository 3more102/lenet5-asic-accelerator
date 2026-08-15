`timescale 1ns/1ps
`include "vectors/config.svh"

// Differential test for rtl/conv5x5_row_mac.sv against golden/quantized_conv.py.
//
// Until this existed the block was reached only through tb_conv5x5_pe, which
// drives it with one directed convolution's worth of values. That is thin
// stimulus for five independent multiply lanes, and the gate-level tier
// measured how thin: 3 of 4 mutations caught, against 5 of 5 for requantize
// which has a testbench of its own (results/gls_20260815.log). It is also one
// of the two blocks unbounded equivalence cannot reach -- mapping destroys the
// correspondence between the two adder trees, so `make equiv-mapped` covers
// requantize and avg_pool2x2_int8 and stops there -- which leaves simulation
// as the only evidence this block has.
//
// Vectors come from golden/generate_vectors.py: generate_mac_vectors(). The
// module is purely combinational, so this needs no clock: apply, settle,
// compare.
module tb_conv5x5_row_mac;
    localparam integer COUNT = `TV_MACC_COUNT;
    localparam integer LANES = `TV_MACC_LANES;

    logic signed [(5*8)-1:0] act_row;
    logic signed [(5*8)-1:0] wgt_row;
    logic signed [31:0]      row_sum;

    logic [39:0] act_vector      [0:COUNT-1];
    logic [39:0] wgt_vector      [0:COUNT-1];
    logic [31:0] expected_vector [0:COUNT-1];

    integer index;
    integer lane;
    integer mismatches;
    integer peak_pos_hits;
    integer peak_neg_hits;
    integer zero_sum_hits;
    integer negative_sums;
    integer lanes_at_min;
    integer lanes_at_max;
    logic signed [31:0] expected;
    logic signed [7:0]  act_lane;
    logic signed [7:0]  wgt_lane;

    // Bit per lane, set the first time that lane is driven to an int8 extreme
    // on either operand. A generator that quietly narrowed to a subset of
    // lanes would still produce a passing run without these.
    logic [LANES-1:0] lane_saw_min;
    logic [LANES-1:0] lane_saw_max;

    conv5x5_row_mac #(
        .DATA_WIDTH(8),
        .ACC_WIDTH(32)
    ) dut (
        .act_row_i (act_row),
        .wgt_row_i (wgt_row),
        .row_sum_o (row_sum)
    );

    initial begin
        $readmemh("vectors/mac/conv_act.hex", act_vector);
        $readmemh("vectors/mac/conv_wgt.hex", wgt_vector);
        $readmemh("vectors/mac/conv_expected.hex", expected_vector);

        mismatches    = 0;
        peak_pos_hits = 0;
        peak_neg_hits = 0;
        zero_sum_hits = 0;
        negative_sums = 0;
        lane_saw_min  = '0;
        lane_saw_max  = '0;

        for (index = 0; index < COUNT; index = index + 1) begin
            act_row = $signed(act_vector[index]);
            wgt_row = $signed(wgt_vector[index]);
            #1;

            expected = $signed(expected_vector[index]);
            if (row_sum !== expected) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10) begin
                    $display("MISMATCH [%0d]: act=%010h wgt=%010h -> got %0d, expected %0d",
                             index, act_vector[index], wgt_vector[index], row_sum, expected);
                end
            end

            if (expected == `TV_MACC_PEAK_POS) peak_pos_hits = peak_pos_hits + 1;
            if (expected == `TV_MACC_PEAK_NEG) peak_neg_hits = peak_neg_hits + 1;
            if (expected == 0)                 zero_sum_hits = zero_sum_hits + 1;
            if (expected < 0)                  negative_sums = negative_sums + 1;

            for (lane = 0; lane < LANES; lane = lane + 1) begin
                act_lane = $signed(act_vector[index][(lane*8) +: 8]);
                wgt_lane = $signed(wgt_vector[index][(lane*8) +: 8]);
                if (act_lane == -8'sd128 || wgt_lane == -8'sd128)
                    lane_saw_min[lane] = 1'b1;
                if (act_lane == 8'sd127 || wgt_lane == 8'sd127)
                    lane_saw_max[lane] = 1'b1;
            end
        end

        if (mismatches != 0) begin
            $fatal(1, "tb_conv5x5_row_mac: %0d/%0d cases mismatched the golden model",
                   mismatches, COUNT);
        end

        // A pass over vectors that never reached anything interesting is a
        // false pass. Require both ends of the sum's range, a cancelling case,
        // negative sums at all, and -- the one that matters most for a lane
        // MAC -- every lane individually driven to both int8 extremes.
        lanes_at_min = $countones(lane_saw_min);
        lanes_at_max = $countones(lane_saw_max);

        if (peak_pos_hits == 0 || peak_neg_hits == 0) begin
            $fatal(1, "tb_conv5x5_row_mac: vectors never reached the sum extremes (%0d high, %0d low)",
                   peak_pos_hits, peak_neg_hits);
        end
        // The generator emits six deliberate cancelling rows (two operand sets
        // in three orderings each), so a floor of six here fails if that class
        // is ever dropped -- a plain "not zero" check would be satisfied by
        // the one row that lands on zero by accident out of several thousand
        // random ones.
        if (zero_sum_hits < 6 || negative_sums == 0) begin
            $fatal(1, "tb_conv5x5_row_mac: too few cancelling or negative sums (%0d zero, %0d negative)",
                   zero_sum_hits, negative_sums);
        end
        if (lanes_at_min != LANES || lanes_at_max != LANES) begin
            $fatal(1, "tb_conv5x5_row_mac: only %0d/%0d lanes saw -128 and %0d/%0d saw +127",
                   lanes_at_min, LANES, lanes_at_max, LANES);
        end
        // Cross-check against what the generator recorded, so a stale
        // vectors/ against a fresh config.svh (or the reverse) is caught here
        // rather than showing up as an unexplained mismatch.
        if (lanes_at_min != `TV_MACC_LANES_AT_MIN || lanes_at_max != `TV_MACC_LANES_AT_MAX) begin
            $fatal(1, "tb_conv5x5_row_mac: lane coverage %0d/%0d disagrees with config.svh %0d/%0d",
                   lanes_at_min, lanes_at_max, `TV_MACC_LANES_AT_MIN, `TV_MACC_LANES_AT_MAX);
        end

        $display("PASS tb_conv5x5_row_mac: %0d/%0d cases matched the Python golden model", COUNT, COUNT);
        $display("  peak +%0d hit %0d times, peak %0d hit %0d times, %0d cancelled to zero, %0d negative",
                 `TV_MACC_PEAK_POS, peak_pos_hits, `TV_MACC_PEAK_NEG, peak_neg_hits,
                 zero_sum_hits, negative_sums);
        $display("  all %0d lanes driven to both -128 and +127", LANES);
        $finish;
    end
endmodule
