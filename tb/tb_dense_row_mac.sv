`timescale 1ns/1ps
`include "vectors/config.svh"

// Differential test for rtl/dense_row_mac.sv against golden/quantized_conv.py.
//
// This block had the weakest stimulus in the repo. Until now it was reached
// only through tb_dense_engine, whose F6 configuration checks five output
// values -- five vectors against an eight-lane MAC that maps to 2,984 sky130hd
// cells. The gate-level tier put a number on it: 2 of 5 mutations caught,
// against 5 of 5 for requantize (results/gls_20260815.log). Nothing was wrong
// with the netlist or the mutations; the stimulus simply never drove most of
// the block. It is also, with conv5x5_row_mac, one of the two blocks unbounded
// equivalence cannot reach, so simulation is all the evidence it has.
//
// The pairwise reduction tree is built by generate/genvar and handles an
// unpaired lane at any level by carrying it forward, so the lanes are not
// interchangeable and the stimulus has to reach each one on its own. Vectors
// come from golden/generate_vectors.py: generate_mac_vectors().
module tb_dense_row_mac;
    localparam integer COUNT = `TV_MACD_COUNT;
    localparam integer LANES = `TV_MACD_LANES;

    logic signed [(8*8)-1:0] act_lanes;
    logic signed [(8*8)-1:0] wgt_lanes;
    logic signed [31:0]      lane_sum;

    logic [63:0] act_vector      [0:COUNT-1];
    logic [63:0] wgt_vector      [0:COUNT-1];
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

    logic [LANES-1:0] lane_saw_min;
    logic [LANES-1:0] lane_saw_max;

    // LANES comes from the same macro the coverage loop below uses, so the DUT
    // and the check cannot drift apart if the generator's lane count changes.
    dense_row_mac #(
        .DATA_WIDTH(8),
        .ACC_WIDTH(32),
        .LANES(LANES)
    ) dut (
        .act_lane_i (act_lanes),
        .wgt_lane_i (wgt_lanes),
        .lane_sum_o (lane_sum)
    );

    initial begin
        $readmemh("vectors/mac/dense_act.hex", act_vector);
        $readmemh("vectors/mac/dense_wgt.hex", wgt_vector);
        $readmemh("vectors/mac/dense_expected.hex", expected_vector);

        mismatches    = 0;
        peak_pos_hits = 0;
        peak_neg_hits = 0;
        zero_sum_hits = 0;
        negative_sums = 0;
        lane_saw_min  = '0;
        lane_saw_max  = '0;

        for (index = 0; index < COUNT; index = index + 1) begin
            act_lanes = $signed(act_vector[index]);
            wgt_lanes = $signed(wgt_vector[index]);
            #1;

            expected = $signed(expected_vector[index]);
            if (lane_sum !== expected) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10) begin
                    $display("MISMATCH [%0d]: act=%016h wgt=%016h -> got %0d, expected %0d",
                             index, act_vector[index], wgt_vector[index], lane_sum, expected);
                end
            end

            if (expected == `TV_MACD_PEAK_POS) peak_pos_hits = peak_pos_hits + 1;
            if (expected == `TV_MACD_PEAK_NEG) peak_neg_hits = peak_neg_hits + 1;
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
            $fatal(1, "tb_dense_row_mac: %0d/%0d cases mismatched the golden model",
                   mismatches, COUNT);
        end

        lanes_at_min = $countones(lane_saw_min);
        lanes_at_max = $countones(lane_saw_max);

        if (peak_pos_hits == 0 || peak_neg_hits == 0) begin
            $fatal(1, "tb_dense_row_mac: vectors never reached the sum extremes (%0d high, %0d low)",
                   peak_pos_hits, peak_neg_hits);
        end
        // Six deliberate cancelling rows, as in tb_conv5x5_row_mac -- a floor
        // rather than a "not zero" check, so dropping that stimulus class
        // fails the run instead of being covered by an accidental zero.
        if (zero_sum_hits < 6 || negative_sums == 0) begin
            $fatal(1, "tb_dense_row_mac: too few cancelling or negative sums (%0d zero, %0d negative)",
                   zero_sum_hits, negative_sums);
        end
        if (lanes_at_min != LANES || lanes_at_max != LANES) begin
            $fatal(1, "tb_dense_row_mac: only %0d/%0d lanes saw -128 and %0d/%0d saw +127",
                   lanes_at_min, LANES, lanes_at_max, LANES);
        end
        if (lanes_at_min != `TV_MACD_LANES_AT_MIN || lanes_at_max != `TV_MACD_LANES_AT_MAX) begin
            $fatal(1, "tb_dense_row_mac: lane coverage %0d/%0d disagrees with config.svh %0d/%0d",
                   lanes_at_min, lanes_at_max, `TV_MACD_LANES_AT_MIN, `TV_MACD_LANES_AT_MAX);
        end

        $display("PASS tb_dense_row_mac: %0d/%0d cases matched the Python golden model", COUNT, COUNT);
        $display("  peak +%0d hit %0d times, peak %0d hit %0d times, %0d cancelled to zero, %0d negative",
                 `TV_MACD_PEAK_POS, peak_pos_hits, `TV_MACD_PEAK_NEG, peak_neg_hits,
                 zero_sum_hits, negative_sums);
        $display("  all %0d lanes driven to both -128 and +127", LANES);
        $finish;
    end
endmodule
