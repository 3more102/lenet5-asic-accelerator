// ---------------------------------------------------------------------------
// tb_extremes.sv -- int8-extreme operands at the largest layer the engines
// accept, plus the smallest legal configuration.
//
// What this adds over the existing per-module testbenches
// ------------------------------------------------------
// Every other engine-level vector set in this project draws activations from
// [-32, 32) and weights from [-16, 16). The largest accumulator any of them
// produces is around 39,000 -- about 0.002% of the 32-bit accumulator's range
// -- and the int8 extremes -128 and +127 are never driven at any operand
// position. Two things go untested as a result:
//
//   1. -128. Its negation is not representable in int8, which makes it the
//      classic place a signed multiplier or a sign-extension is wrong. A
//      design that computes (-128)(-128) as -16384 instead of +16384 passes
//      every existing testbench in this repo.
//
//   2. Whether ACC_WIDTH is wide enough. "32 bits is plenty" is an assumption
//      the repo states and never exercises; nothing has ever put more than
//      ~2^16 into an accumulator that is sized for 2^31.
//
// The shapes are the worst case the *design* permits rather than the shapes
// LeNet-5 happens to use: 16 input channels of 5x5 taps is 400 MACs per output
// pixel, which is what C5 costs and the most conv2d_engine can be configured
// for at MAX_IN_CH = 16. The dense case uses MAX_IN_LEN = 120.
//
// Why the shifts are 16 and 12, not the deployment 6/7
// ----------------------------------------------------
// At shift 7 an accumulator of several million requantizes far past +-127 and
// every output pins to the rail. out_data_o would then be completely
// insensitive to the accumulator underneath it: the arithmetic could be wrong
// by millions and the test would still pass. The shifts here are chosen so the
// extreme accumulators land back inside int8 and the output is a faithful
// function of the whole accumulator. Saturation itself is not skipped, it is
// covered exhaustively at both boundaries by tb_requantize.
//
// The same trap is why the conv case has only two output channels. A pattern
// whose total contribution is a few thousand requantizes to zero at shift 16
// and would still requantize to zero with its sign inverted -- interesting to
// look at, vacuous to assert on. Those sign-structure patterns are in the
// dense case instead, where out_acc_o is compared directly and magnitude is
// irrelevant.
//
// out_acc_o had no oracle check anywhere before this file
// ------------------------------------------------------
// golden/generate_vectors.py has always written vectors/f6/accumulator.hex and
// vectors/accumulator.hex, and no testbench ever read either one. dense_engine
// exposes out_acc_o and tb_dense_engine connects it and then ignores it, so the
// raw accumulator was never compared against the model. That signal is not
// incidental: classifier_argmax decides the predicted class from out_acc_o
// rather than the requantized score, precisely so two classes that both
// saturate cannot tie -- see docs/ARCHITECTURE.md and the
// test_argmax_classifier_avoids_saturation_misclassification regression. A
// wrong-but-self-consistent accumulator would have passed every check in the
// project. This testbench compares it against the golden model beat for beat.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

`include "vectors/config.svh"

module tb_extremes;

    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;

    // Generous: the whole run is a few thousand cycles. This exists so a
    // deadlocked engine fails loudly instead of hanging a CI job.
    localparam integer WATCHDOG_CYCLES = 50000;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    integer cycles = 0;
    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (cycles > WATCHDOG_CYCLES) begin
            $fatal(1, "watchdog: no completion after %0d cycles", WATCHDOG_CYCLES);
        end
    end

    // ==================================================================
    // conv2d_engine at MAX_IN_CH: 16 channels x 25 taps = 400 MACs/output
    // ==================================================================
    localparam integer C_ACT_DEPTH  = `TV_EX_ACT_COUNT;
    localparam integer C_WGT_DEPTH  = `TV_EX_WGT_COUNT;
    localparam integer C_BIAS_DEPTH = `TV_EX_BIAS_COUNT;
    localparam integer C_CONN_DEPTH = `TV_EX_CONN_COUNT;
    localparam integer C_OUT_COUNT  = `TV_EX_OUT_COUNT;
    localparam integer C_ACT_AW  = (C_ACT_DEPTH  <= 1) ? 1 : $clog2(C_ACT_DEPTH);
    localparam integer C_WGT_AW  = (C_WGT_DEPTH  <= 1) ? 1 : $clog2(C_WGT_DEPTH);
    localparam integer C_BIAS_AW = (C_BIAS_DEPTH <= 1) ? 1 : $clog2(C_BIAS_DEPTH);
    localparam integer C_CONN_AW = (C_CONN_DEPTH <= 1) ? 1 : $clog2(C_CONN_DEPTH);

    reg signed [DATA_WIDTH-1:0] c_act_vec  [0:C_ACT_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] c_wgt_vec  [0:C_WGT_DEPTH-1];
    reg signed [ACC_WIDTH-1:0]  c_bias_vec [0:C_BIAS_DEPTH-1];
    reg        [7:0]            c_conn_vec [0:C_CONN_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] c_exp_vec  [0:C_OUT_COUNT-1];

    // Second pass over the same DUT and the same activations: weights that
    // cancel to exactly zero, run at shift 0 so the output is the accumulator
    // itself. See the header -- the magnitude pass cannot resolve a small
    // per-product error and this one cannot reach a large accumulator, so
    // neither alone is enough.
    localparam integer CC_WGT_DEPTH  = `TV_EXC_WGT_COUNT;
    localparam integer CC_BIAS_DEPTH = `TV_EXC_BIAS_COUNT;
    reg signed [DATA_WIDTH-1:0] cc_wgt_vec  [0:CC_WGT_DEPTH-1];
    reg signed [ACC_WIDTH-1:0]  cc_bias_vec [0:CC_BIAS_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] cc_exp_vec  [0:`TV_EXC_OUT_COUNT-1];

    reg [5:0] c_shift;
    reg       c_cancel_phase;

    logic c_start, c_busy, c_done, c_config_error;
    logic c_out_valid;
    logic signed [DATA_WIDTH-1:0] c_out_data;
    logic [7:0] c_out_channel, c_out_y, c_out_x;

    reg c_act_we, c_wgt_we, c_bias_we, c_conn_we, c_conn_data;
    reg [C_ACT_AW-1:0]  c_act_addr;
    reg [C_WGT_AW-1:0]  c_wgt_addr;
    reg [C_BIAS_AW-1:0] c_bias_addr;
    reg [C_CONN_AW-1:0] c_conn_addr;
    reg signed [DATA_WIDTH-1:0] c_act_data, c_wgt_data;
    reg signed [ACC_WIDTH-1:0]  c_bias_data;

    integer c_idx = 0;

    conv2d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .MAX_IN_W  (`TV_EX_IN_W),
        .MAX_IN_H  (`TV_EX_IN_H),
        .MAX_IN_CH (`TV_EX_IN_CH),
        .MAX_OUT_CH(`TV_EX_OUT_CH),
        .ACT_DEPTH (C_ACT_DEPTH),
        .WGT_DEPTH (C_WGT_DEPTH),
        .BIAS_DEPTH(C_BIAS_DEPTH),
        .CONN_DEPTH(C_CONN_DEPTH)
    ) u_conv (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(c_start),
        .cfg_in_w_i(8'(`TV_EX_IN_W)), .cfg_in_h_i(8'(`TV_EX_IN_H)),
        .cfg_in_ch_i(8'(`TV_EX_IN_CH)), .cfg_out_ch_i(8'(`TV_EX_OUT_CH)),
        .cfg_shift_i(c_shift), .cfg_relu_en_i(`TV_EX_RELU),
        .busy_o(c_busy), .done_o(c_done), .config_error_o(c_config_error),
        .load_act_we_i(c_act_we),   .load_act_addr_i(c_act_addr),   .load_act_data_i(c_act_data),
        .load_wgt_we_i(c_wgt_we),   .load_wgt_addr_i(c_wgt_addr),   .load_wgt_data_i(c_wgt_data),
        .load_bias_we_i(c_bias_we), .load_bias_addr_i(c_bias_addr), .load_bias_data_i(c_bias_data),
        .load_conn_we_i(c_conn_we), .load_conn_addr_i(c_conn_addr), .load_conn_data_i(c_conn_data),
        .out_valid_o(c_out_valid), .out_ready_i(1'b1),
        .out_data_o(c_out_data), .out_channel_o(c_out_channel),
        .out_y_o(c_out_y), .out_x_o(c_out_x)
    );

    always @(posedge clk) begin
        if (rst_n && c_out_valid) begin
            if (c_idx >= C_OUT_COUNT) begin
                $fatal(1, "conv2d_engine produced more than %0d beats", C_OUT_COUNT);
            end
            if (!c_cancel_phase
                && $signed(c_out_data) !== $signed(c_exp_vec[c_idx])) begin
                $fatal(1,
                    "conv2d_engine extreme beat %0d: got %0d, golden model says %0d",
                    c_idx, $signed(c_out_data), $signed(c_exp_vec[c_idx]));
            end
            if (c_cancel_phase
                && $signed(c_out_data) !== $signed(cc_exp_vec[c_idx])) begin
                $fatal(1,
                    "conv2d_engine cancelling beat %0d: got %0d, golden model says %0d -- at shift 0 the output is the accumulator, so this is a wrong product, not a rounding step",
                    c_idx, $signed(c_out_data), $signed(cc_exp_vec[c_idx]));
            end
            c_idx <= c_idx + 1;
        end
    end

    // ==================================================================
    // conv2d_engine at the smallest legal configuration
    // 1 input channel, 1 output channel, 5x5 -> 1x1: a single output beat.
    // Nested counters that are off by one at the bottom of their range have
    // nowhere to hide in a one-beat stream, and no other testbench in the
    // project runs a shape this small.
    // ==================================================================
    localparam integer M_ACT_DEPTH  = `TV_EXMIN_ACT_COUNT;
    localparam integer M_WGT_DEPTH  = `TV_EXMIN_WGT_COUNT;
    localparam integer M_BIAS_DEPTH = `TV_EXMIN_BIAS_COUNT;
    localparam integer M_CONN_DEPTH = `TV_EXMIN_CONN_COUNT;
    localparam integer M_OUT_COUNT  = `TV_EXMIN_OUT_COUNT;
    localparam integer M_ACT_AW  = (M_ACT_DEPTH  <= 1) ? 1 : $clog2(M_ACT_DEPTH);
    localparam integer M_WGT_AW  = (M_WGT_DEPTH  <= 1) ? 1 : $clog2(M_WGT_DEPTH);
    localparam integer M_BIAS_AW = (M_BIAS_DEPTH <= 1) ? 1 : $clog2(M_BIAS_DEPTH);
    localparam integer M_CONN_AW = (M_CONN_DEPTH <= 1) ? 1 : $clog2(M_CONN_DEPTH);

    reg signed [DATA_WIDTH-1:0] m_act_vec  [0:M_ACT_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] m_wgt_vec  [0:M_WGT_DEPTH-1];
    reg signed [ACC_WIDTH-1:0]  m_bias_vec [0:M_BIAS_DEPTH-1];
    reg        [7:0]            m_conn_vec [0:M_CONN_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] m_exp_vec  [0:M_OUT_COUNT-1];

    logic m_start, m_busy, m_done, m_config_error;
    logic m_out_valid;
    logic signed [DATA_WIDTH-1:0] m_out_data;
    logic [7:0] m_out_channel, m_out_y, m_out_x;

    reg m_act_we, m_wgt_we, m_bias_we, m_conn_we, m_conn_data;
    reg [M_ACT_AW-1:0]  m_act_addr;
    reg [M_WGT_AW-1:0]  m_wgt_addr;
    reg [M_BIAS_AW-1:0] m_bias_addr;
    reg [M_CONN_AW-1:0] m_conn_addr;
    reg signed [DATA_WIDTH-1:0] m_act_data, m_wgt_data;
    reg signed [ACC_WIDTH-1:0]  m_bias_data;

    integer m_idx = 0;

    conv2d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .MAX_IN_W  (`TV_EXMIN_IN_W),
        .MAX_IN_H  (`TV_EXMIN_IN_H),
        .MAX_IN_CH (`TV_EXMIN_IN_CH),
        .MAX_OUT_CH(`TV_EXMIN_OUT_CH),
        .ACT_DEPTH (M_ACT_DEPTH),
        .WGT_DEPTH (M_WGT_DEPTH),
        .BIAS_DEPTH(M_BIAS_DEPTH),
        .CONN_DEPTH(M_CONN_DEPTH)
    ) u_min (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(m_start),
        .cfg_in_w_i(8'(`TV_EXMIN_IN_W)), .cfg_in_h_i(8'(`TV_EXMIN_IN_H)),
        .cfg_in_ch_i(8'(`TV_EXMIN_IN_CH)), .cfg_out_ch_i(8'(`TV_EXMIN_OUT_CH)),
        .cfg_shift_i(6'(`TV_EXMIN_SHIFT)), .cfg_relu_en_i(`TV_EXMIN_RELU),
        .busy_o(m_busy), .done_o(m_done), .config_error_o(m_config_error),
        .load_act_we_i(m_act_we),   .load_act_addr_i(m_act_addr),   .load_act_data_i(m_act_data),
        .load_wgt_we_i(m_wgt_we),   .load_wgt_addr_i(m_wgt_addr),   .load_wgt_data_i(m_wgt_data),
        .load_bias_we_i(m_bias_we), .load_bias_addr_i(m_bias_addr), .load_bias_data_i(m_bias_data),
        .load_conn_we_i(m_conn_we), .load_conn_addr_i(m_conn_addr), .load_conn_data_i(m_conn_data),
        .out_valid_o(m_out_valid), .out_ready_i(1'b1),
        .out_data_o(m_out_data), .out_channel_o(m_out_channel),
        .out_y_o(m_out_y), .out_x_o(m_out_x)
    );

    always @(posedge clk) begin
        if (rst_n && m_out_valid) begin
            if (m_idx >= M_OUT_COUNT) begin
                $fatal(1, "minimum-config conv2d_engine produced more than %0d beats", M_OUT_COUNT);
            end
            if ($signed(m_out_data) !== $signed(m_exp_vec[m_idx])) begin
                $fatal(1,
                    "minimum-config conv2d_engine beat %0d: got %0d, golden model says %0d",
                    m_idx, $signed(m_out_data), $signed(m_exp_vec[m_idx]));
            end
            // The one output pixel of a 5x5 -> 1x1 valid convolution is at
            // (0,0) of channel 0. An off-by-one in the output coordinate
            // counters is invisible in a bigger stream where some beat
            // legitimately carries every value.
            if (m_out_channel !== 8'd0 || m_out_y !== 8'd0 || m_out_x !== 8'd0) begin
                $fatal(1,
                    "minimum-config conv2d_engine reported position ch %0d y %0d x %0d, expected 0/0/0",
                    m_out_channel, m_out_y, m_out_x);
            end
            m_idx <= m_idx + 1;
        end
    end

    // ==================================================================
    // dense_engine at MAX_IN_LEN, with out_acc_o checked against the model
    // ==================================================================
    localparam integer D_ACT_DEPTH  = `TV_EXD_ACT_COUNT;
    localparam integer D_WGT_DEPTH  = `TV_EXD_WGT_COUNT;
    localparam integer D_BIAS_DEPTH = `TV_EXD_BIAS_COUNT;
    localparam integer D_OUT_COUNT  = `TV_EXD_OUT_COUNT;
    localparam integer D_ACT_AW  = (D_ACT_DEPTH  <= 1) ? 1 : $clog2(D_ACT_DEPTH);
    localparam integer D_WGT_AW  = (D_WGT_DEPTH  <= 1) ? 1 : $clog2(D_WGT_DEPTH);
    localparam integer D_BIAS_AW = (D_BIAS_DEPTH <= 1) ? 1 : $clog2(D_BIAS_DEPTH);

    reg signed [DATA_WIDTH-1:0] d_act_vec  [0:D_ACT_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] d_wgt_vec  [0:D_WGT_DEPTH-1];
    reg signed [ACC_WIDTH-1:0]  d_bias_vec [0:D_BIAS_DEPTH-1];
    reg signed [DATA_WIDTH-1:0] d_exp_vec  [0:D_OUT_COUNT-1];
    // The oracle writes accumulators 64 bits wide so a value that overflowed
    // int32 would be visible as itself rather than silently wrapped into
    // agreement with a broken DUT.
    reg signed [63:0]           d_acc_vec  [0:D_OUT_COUNT-1];

    logic d_start, d_busy, d_done, d_config_error;
    logic d_out_valid;
    logic signed [DATA_WIDTH-1:0] d_out_data;
    logic signed [ACC_WIDTH-1:0]  d_out_acc;
    logic [7:0] d_out_index;

    reg d_act_we, d_wgt_we, d_bias_we;
    reg [D_ACT_AW-1:0]  d_act_addr;
    reg [D_WGT_AW-1:0]  d_wgt_addr;
    reg [D_BIAS_AW-1:0] d_bias_addr;
    reg signed [DATA_WIDTH-1:0] d_act_data, d_wgt_data;
    reg signed [ACC_WIDTH-1:0]  d_bias_data;

    integer d_idx = 0;
    integer d_peak_acc = 0;   // largest |out_acc_o| the RTL actually carried

    dense_engine #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .MAX_IN_LEN (`TV_EXD_IN_LEN),
        .MAX_OUT_LEN(`TV_EXD_OUT_LEN),
        .ACT_DEPTH  (D_ACT_DEPTH),
        .WGT_DEPTH  (D_WGT_DEPTH),
        .BIAS_DEPTH (D_BIAS_DEPTH)
    ) u_dense (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(d_start),
        .cfg_in_len_i(8'(`TV_EXD_IN_LEN)), .cfg_out_len_i(8'(`TV_EXD_OUT_LEN)),
        .cfg_shift_i(6'(`TV_EXD_SHIFT)), .cfg_relu_en_i(`TV_EXD_RELU),
        .busy_o(d_busy), .done_o(d_done), .config_error_o(d_config_error),
        .load_act_we_i(d_act_we),   .load_act_addr_i(d_act_addr),   .load_act_data_i(d_act_data),
        .load_wgt_we_i(d_wgt_we),   .load_wgt_addr_i(d_wgt_addr),   .load_wgt_data_i(d_wgt_data),
        .load_bias_we_i(d_bias_we), .load_bias_addr_i(d_bias_addr), .load_bias_data_i(d_bias_data),
        .out_valid_o(d_out_valid), .out_ready_i(1'b1),
        .out_data_o(d_out_data), .out_acc_o(d_out_acc), .out_index_o(d_out_index)
    );

    always @(posedge clk) begin
        if (rst_n && d_out_valid) begin
            if (d_idx >= D_OUT_COUNT) begin
                $fatal(1, "dense_engine produced more than %0d beats", D_OUT_COUNT);
            end
            if ($signed(d_out_data) !== $signed(d_exp_vec[d_idx])) begin
                $fatal(1,
                    "dense_engine extreme beat %0d: got %0d, golden model says %0d",
                    d_idx, $signed(d_out_data), $signed(d_exp_vec[d_idx]));
            end
            // Sign-extended to 64 bits before the compare, so an accumulator
            // that wrapped at 32 bits disagrees with the model instead of
            // being truncated into agreement with it.
            if ($signed({{32{d_out_acc[ACC_WIDTH-1]}}, d_out_acc}) !== d_acc_vec[d_idx]) begin
                $fatal(1,
                    "dense_engine extreme beat %0d: out_acc_o = %0d, golden model says %0d -- the raw accumulator classifier_argmax votes on is wrong",
                    d_idx, $signed(d_out_acc), d_acc_vec[d_idx]);
            end
            if (d_out_index !== d_idx[7:0]) begin
                $fatal(1, "dense_engine beat %0d reported out_index_o = %0d",
                       d_idx, d_out_index);
            end
            if (d_out_acc > 0 && d_out_acc > d_peak_acc) d_peak_acc <= d_out_acc;
            if (d_out_acc < 0 && -d_out_acc > d_peak_acc) d_peak_acc <= -d_out_acc;
            d_idx <= d_idx + 1;
        end
    end

    // ==================================================================
    // Load and run
    // ==================================================================
    task automatic load_conv;
        integer i;
        begin
            for (i = 0; i < C_ACT_DEPTH; i = i + 1) begin
                @(negedge clk);
                c_act_we = 1'b1; c_act_addr = i[C_ACT_AW-1:0]; c_act_data = c_act_vec[i];
            end
            @(negedge clk); c_act_we = 1'b0;
            for (i = 0; i < C_WGT_DEPTH; i = i + 1) begin
                @(negedge clk);
                c_wgt_we = 1'b1; c_wgt_addr = i[C_WGT_AW-1:0]; c_wgt_data = c_wgt_vec[i];
            end
            @(negedge clk); c_wgt_we = 1'b0;
            for (i = 0; i < C_BIAS_DEPTH; i = i + 1) begin
                @(negedge clk);
                c_bias_we = 1'b1; c_bias_addr = i[C_BIAS_AW-1:0]; c_bias_data = c_bias_vec[i];
            end
            @(negedge clk); c_bias_we = 1'b0;
            for (i = 0; i < C_CONN_DEPTH; i = i + 1) begin
                @(negedge clk);
                c_conn_we = 1'b1; c_conn_addr = i[C_CONN_AW-1:0]; c_conn_data = c_conn_vec[i][0];
            end
            @(negedge clk); c_conn_we = 1'b0;
        end
    endtask

    // Only the weights and biases change for the cancelling pass; the
    // activations are deliberately the same array, so any difference in the
    // result comes from the multiply and not from new stimulus.
    task automatic load_conv_cancel;
        integer i;
        begin
            for (i = 0; i < CC_WGT_DEPTH; i = i + 1) begin
                @(negedge clk);
                c_wgt_we = 1'b1; c_wgt_addr = i[C_WGT_AW-1:0]; c_wgt_data = cc_wgt_vec[i];
            end
            @(negedge clk); c_wgt_we = 1'b0;
            for (i = 0; i < CC_BIAS_DEPTH; i = i + 1) begin
                @(negedge clk);
                c_bias_we = 1'b1; c_bias_addr = i[C_BIAS_AW-1:0]; c_bias_data = cc_bias_vec[i];
            end
            @(negedge clk); c_bias_we = 1'b0;
        end
    endtask

    task automatic load_min;
        integer i;
        begin
            for (i = 0; i < M_ACT_DEPTH; i = i + 1) begin
                @(negedge clk);
                m_act_we = 1'b1; m_act_addr = i[M_ACT_AW-1:0]; m_act_data = m_act_vec[i];
            end
            @(negedge clk); m_act_we = 1'b0;
            for (i = 0; i < M_WGT_DEPTH; i = i + 1) begin
                @(negedge clk);
                m_wgt_we = 1'b1; m_wgt_addr = i[M_WGT_AW-1:0]; m_wgt_data = m_wgt_vec[i];
            end
            @(negedge clk); m_wgt_we = 1'b0;
            for (i = 0; i < M_BIAS_DEPTH; i = i + 1) begin
                @(negedge clk);
                m_bias_we = 1'b1; m_bias_addr = i[M_BIAS_AW-1:0]; m_bias_data = m_bias_vec[i];
            end
            @(negedge clk); m_bias_we = 1'b0;
            for (i = 0; i < M_CONN_DEPTH; i = i + 1) begin
                @(negedge clk);
                m_conn_we = 1'b1; m_conn_addr = i[M_CONN_AW-1:0]; m_conn_data = m_conn_vec[i][0];
            end
            @(negedge clk); m_conn_we = 1'b0;
        end
    endtask

    task automatic load_dense;
        integer i;
        begin
            for (i = 0; i < D_ACT_DEPTH; i = i + 1) begin
                @(negedge clk);
                d_act_we = 1'b1; d_act_addr = i[D_ACT_AW-1:0]; d_act_data = d_act_vec[i];
            end
            @(negedge clk); d_act_we = 1'b0;
            for (i = 0; i < D_WGT_DEPTH; i = i + 1) begin
                @(negedge clk);
                d_wgt_we = 1'b1; d_wgt_addr = i[D_WGT_AW-1:0]; d_wgt_data = d_wgt_vec[i];
            end
            @(negedge clk); d_wgt_we = 1'b0;
            for (i = 0; i < D_BIAS_DEPTH; i = i + 1) begin
                @(negedge clk);
                d_bias_we = 1'b1; d_bias_addr = i[D_BIAS_AW-1:0]; d_bias_data = d_bias_vec[i];
            end
            @(negedge clk); d_bias_we = 1'b0;
        end
    endtask

    // ------------------------------------------------------------------
    // Anti-vacuity: prove the operands really are extreme.
    //
    // Everything this file claims rests on the vectors containing -128 and
    // +127. If golden/generate_vectors.py is ever changed so they do not, the
    // testbench would keep passing while quietly testing ordinary values --
    // the exact failure mode this tier exists to close. Check it rather than
    // trust it.
    // ------------------------------------------------------------------
    task automatic assert_operands_are_extreme;
        integer i;
        reg saw_min, saw_max;
        begin
            saw_min = 1'b0; saw_max = 1'b0;
            for (i = 0; i < C_ACT_DEPTH; i = i + 1) begin
                if (c_act_vec[i] === 8'sh80) saw_min = 1'b1;
                if (c_act_vec[i] === 8'sh7f)  saw_max = 1'b1;
            end
            if (!saw_min || !saw_max) begin
                $fatal(1, "conv activations do not contain both -128 and +127 -- this testbench is not testing extremes");
            end
            saw_min = 1'b0; saw_max = 1'b0;
            for (i = 0; i < C_WGT_DEPTH; i = i + 1) begin
                if (c_wgt_vec[i] === 8'sh80) saw_min = 1'b1;
                if (c_wgt_vec[i] === 8'sh7f)  saw_max = 1'b1;
            end
            if (!saw_min || !saw_max) begin
                $fatal(1, "conv weights do not contain both -128 and +127 -- this testbench is not testing extremes");
            end
            saw_min = 1'b0; saw_max = 1'b0;
            for (i = 0; i < D_ACT_DEPTH; i = i + 1) begin
                if (d_act_vec[i] === 8'sh80) saw_min = 1'b1;
                if (d_act_vec[i] === 8'sh7f)  saw_max = 1'b1;
            end
            if (!saw_min || !saw_max) begin
                $fatal(1, "dense activations do not contain both -128 and +127 -- this testbench is not testing extremes");
            end
        end
    endtask

    // Every expected output must be strictly inside the int8 rails. A beat
    // that saturates carries no information about the accumulator that
    // produced it: the arithmetic could be wrong by millions and the
    // comparison would still hold. See the header.
    task automatic assert_outputs_are_sensitive;
        integer i;
        begin
            for (i = 0; i < C_OUT_COUNT; i = i + 1) begin
                if (c_exp_vec[i] === 8'sh7f || c_exp_vec[i] === 8'sh80) begin
                    $fatal(1, "conv expected beat %0d is at the int8 rail (%0d) -- that beat cannot detect a wrong accumulator",
                           i, $signed(c_exp_vec[i]));
                end
            end
            for (i = 0; i < M_OUT_COUNT; i = i + 1) begin
                if (m_exp_vec[i] === 8'sh7f || m_exp_vec[i] === 8'sh80) begin
                    $fatal(1, "minimum-config expected beat %0d is at the int8 rail (%0d)",
                           i, $signed(m_exp_vec[i]));
                end
            end
            // The cancelling pass has a second requirement beyond not being at
            // the rail: it only resolves single products if it sits far enough
            // from the rail that a 128-sized error is still representable.
            for (i = 0; i < `TV_EXC_OUT_COUNT; i = i + 1) begin
                if (cc_exp_vec[i] === 8'sh7f || cc_exp_vec[i] === 8'sh80) begin
                    $fatal(1, "cancelling expected beat %0d is at the int8 rail (%0d) -- it can no longer resolve a wrong product",
                           i, $signed(cc_exp_vec[i]));
                end
            end
        end
    endtask

    task automatic run_engine(input reg [24*8-1:0] who);
        begin
            case (who)
                "conv", "cancel":  begin
                    @(negedge clk); c_idx <= 0; @(negedge clk);
                    c_start = 1'b1; @(negedge clk); c_start = 1'b0;
                    wait (c_done === 1'b1); @(negedge clk);
                    if (c_idx != C_OUT_COUNT)
                        $fatal(1, "conv2d_engine %0s: done_o after %0d beats, expected %0d",
                               who, c_idx, C_OUT_COUNT);
                    if (c_config_error !== 1'b0)
                        $fatal(1, "conv2d_engine %0s: config_error_o asserted on a legal config", who);
                end
                "min": begin
                    @(negedge clk); m_start = 1'b1; @(negedge clk); m_start = 1'b0;
                    wait (m_done === 1'b1); @(negedge clk);
                    if (m_idx != M_OUT_COUNT)
                        $fatal(1, "minimum-config conv2d_engine: done_o after %0d beats, expected %0d", m_idx, M_OUT_COUNT);
                    if (m_config_error !== 1'b0)
                        $fatal(1, "minimum-config conv2d_engine: config_error_o asserted on a legal minimum-size config");
                end
                default: begin
                    @(negedge clk); d_start = 1'b1; @(negedge clk); d_start = 1'b0;
                    wait (d_done === 1'b1); @(negedge clk);
                    if (d_idx != D_OUT_COUNT)
                        $fatal(1, "dense_engine: done_o after %0d beats, expected %0d", d_idx, D_OUT_COUNT);
                    if (d_config_error !== 1'b0)
                        $fatal(1, "dense_engine: config_error_o asserted on a legal maximum-length config");
                end
            endcase
        end
    endtask

    initial begin
        $readmemh("vectors/extremes/act.hex",         c_act_vec);
        $readmemh("vectors/extremes/wgt.hex",         c_wgt_vec);
        $readmemh("vectors/extremes/bias.hex",        c_bias_vec);
        $readmemh("vectors/extremes/conn.hex",        c_conn_vec);
        $readmemh("vectors/extremes/expected.hex",    c_exp_vec);

        $readmemh("vectors/extremes/cancel/wgt.hex",      cc_wgt_vec);
        $readmemh("vectors/extremes/cancel/bias.hex",     cc_bias_vec);
        $readmemh("vectors/extremes/cancel/expected.hex", cc_exp_vec);

        $readmemh("vectors/extremes/min/act.hex",      m_act_vec);
        $readmemh("vectors/extremes/min/wgt.hex",      m_wgt_vec);
        $readmemh("vectors/extremes/min/bias.hex",     m_bias_vec);
        $readmemh("vectors/extremes/min/conn.hex",     m_conn_vec);
        $readmemh("vectors/extremes/min/expected.hex", m_exp_vec);

        $readmemh("vectors/extremes/dense/act.hex",         d_act_vec);
        $readmemh("vectors/extremes/dense/wgt.hex",         d_wgt_vec);
        $readmemh("vectors/extremes/dense/bias.hex",        d_bias_vec);
        $readmemh("vectors/extremes/dense/expected.hex",    d_exp_vec);
        $readmemh("vectors/extremes/dense/accumulator.hex", d_acc_vec);

        c_start = 1'b0; m_start = 1'b0; d_start = 1'b0;
        c_shift = 6'(`TV_EX_SHIFT); c_cancel_phase = 1'b0;
        c_act_we = 1'b0; c_wgt_we = 1'b0; c_bias_we = 1'b0; c_conn_we = 1'b0;
        m_act_we = 1'b0; m_wgt_we = 1'b0; m_bias_we = 1'b0; m_conn_we = 1'b0;
        d_act_we = 1'b0; d_wgt_we = 1'b0; d_bias_we = 1'b0;

        assert_operands_are_extreme();
        assert_outputs_are_sensitive();

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        load_conv();
        load_min();
        load_dense();

        run_engine("conv");
        $display("  conv2d_engine %0dx%0dx%0d -> %0dx1x1: %0d beats at int8 extremes, %0d MACs each, peak |acc| %0d",
                 `TV_EX_IN_CH, `TV_EX_IN_H, `TV_EX_IN_W, `TV_EX_OUT_CH,
                 c_idx, `TV_EX_IN_CH * 25, `TV_EX_PEAK_ACC);
        $display("PASS tb_extremes: conv2d_engine is bit-exact at the largest layer it accepts, with -128 and +127 at every operand position");

        load_conv_cancel();
        c_shift = 6'(`TV_EXC_SHIFT);
        c_cancel_phase = 1'b1;
        run_engine("cancel");
        $display("  same operands, weights cancelling to zero, shift 0: %0d beats, output equals the accumulator so one wrong product moves it by >= 128",
                 c_idx);
        $display("PASS tb_extremes: conv2d_engine resolves a single wrong product, not just a wrong magnitude");

        run_engine("min");
        $display("  minimum config 1x5x5 -> 1x1x1: %0d beat, position 0/0/0 confirmed", m_idx);
        $display("PASS tb_extremes: conv2d_engine is bit-exact at the smallest legal configuration");

        run_engine("dense");
        if (d_peak_acc !== `TV_EXD_PEAK_ACC) begin
            $fatal(1,
                "dense_engine peak |out_acc_o| was %0d, expected %0d -- the extreme case did not actually run",
                d_peak_acc, `TV_EXD_PEAK_ACC);
        end
        $display("  dense_engine %0d -> %0d: %0d beats, out_acc_o matched the model exactly, peak |acc| %0d of the %0d an int32 accumulator holds (%0dx headroom)",
                 `TV_EXD_IN_LEN, `TV_EXD_OUT_LEN, d_idx, d_peak_acc,
                 32'sh7FFFFFFF, 32'sh7FFFFFFF / d_peak_acc);
        $display("PASS tb_extremes: dense_engine raw accumulator matches the golden model -- the signal classifier_argmax votes on, previously unchecked");

        $display("PASS tb_extremes: int8 extremes and both dimension limits are bit-exact against the golden model");
        $finish;
    end

endmodule
