`timescale 1ns/1ps
`include "vectors/config.svh"

// ---------------------------------------------------------------------------
// Stall-invariance and reset-interruption tier for the three streaming
// engines (conv2d_engine, avg_pool2x2_stream, dense_engine).
//
// Closes three items from docs/VERIFICATION_PLAN.md's "Required before
// tapeout" list: random output stalls, reset interruption policy, and
// assertions for stable output under stall.
//
// What this tier adds that the per-module testbenches do not
// -------------------------------------------------------------
// The existing engine testbenches check *what* an engine computes, against
// the bit-exact Python model. They say nothing about whether that result
// survives the two things a real system does to an engine and a directed
// testbench never does:
//
//   * a consumer that accepts beats irregularly. tb_dense_engine holds
//     out_ready_i high for the whole run, so dense has never been stalled at
//     all; tb_conv2d_engine and tb_avg_pool2x2_stream use a fixed modulo-7
//     pattern, which is better but still periodic -- a bug that only shows up
//     on two consecutive stalled beats, or on a stall landing exactly at a
//     row or channel rollover, can hide from a fixed period indefinitely;
//
//   * a reset that arrives mid-operation. Nothing in this project has ever
//     asserted rst_ni while an engine was busy. An aborted engine that leaves
//     a stale counter behind, hangs instead of returning to idle, or emits a
//     done_o pulse for work it never finished would pass the entire existing
//     regression.
//
// The oracle
// ----------
// Each engine is run three times over the same operands:
//
//   REF    out_ready_i tied high. Every beat is checked against the golden
//          vectors, so this run is oracle-anchored, and the full beat stream
//          is captured.
//   STALL  out_ready_i driven from an LFSR. Every beat must match the REF
//          capture exactly -- data and side-band alike.
//   ABORT  rst_ni asserted mid-stream, then the engine restarted from scratch.
//          Every beat of the restarted run must again match REF.
//
// Using REF as the oracle for the other two is the point rather than a
// weakness: a functional error that changed REF as well would be caught by
// the per-module testbenches, which compare against Python. What is being
// asserted here is *invariance* -- that backpressure and reset are not
// allowed to change the answer -- and the design's own unstalled behaviour is
// exactly the right reference for that.
//
// Anti-vacuity
// ------------
// Both new properties can pass without testing anything, so both are guarded:
//
//   * the STALL run asserts that the protocol checker actually observed
//     stall cycles. An LFSR tap that happened to hold ready high would
//     otherwise turn this into a second REF run that trivially passes;
//   * the ABORT run asserts that reset landed with the engine busy and with
//     some, but not all, of the beats delivered. Resetting an engine that had
//     already finished proves nothing, and is what this check would silently
//     decay into if a geometry change made the abort point fall past the end
//     of the stream.
// ---------------------------------------------------------------------------
module tb_robustness;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;

    // Generous: the longest of the three engines' runs is conv2d's 48 outputs
    // at 3 input channels x 5 kernel rows each, and this testbench performs
    // ten runs in total. Sized to catch a hang, not to be tight.
    localparam integer WATCHDOG_CYCLES = 50000;

    // Beat index at which each ABORT run pulls rst_ni. Asserted to be strictly
    // inside the stream -- see the anti-vacuity note in the header.
    localparam integer C_ABORT_AT = 20;   // of 48
    localparam integer P_ABORT_AT = 5;    // of 12
    localparam integer D_ABORT_AT = 2;    // of 5

    logic clk;
    logic rst_n;

    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // Pseudorandom backpressure.
    //
    // A 16-bit maximal-length LFSR, ANDed down to roughly a one-in-four duty
    // cycle so multi-cycle stalls are common rather than incidental. Advanced
    // on the negative edge so out_ready_i is settled well before the DUT
    // samples it, and reseeded by reset so every run of this testbench --
    // including the restart inside the ABORT phase -- sees the identical
    // pattern in both simulators. Nothing here is $random: a robustness
    // failure that cannot be reproduced from the log is not much use.
    // ------------------------------------------------------------------
    reg [15:0] lfsr;
    reg        stall_en;

    always @(negedge clk) begin
        if (!rst_n) begin
            lfsr <= 16'hACE1;
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
        end
    end

    wire rnd_ready = stall_en ? (lfsr[0] & lfsr[4]) : 1'b1;

    // ==================================================================
    // conv2d_engine
    // ==================================================================
    localparam integer C_ACT_DEPTH  = `TV_ACT_COUNT;
    localparam integer C_WGT_DEPTH  = `TV_WGT_COUNT;
    localparam integer C_BIAS_DEPTH = `TV_BIAS_COUNT;
    localparam integer C_CONN_DEPTH = `TV_CONN_COUNT;
    localparam integer C_ACT_AW  = (C_ACT_DEPTH  <= 1) ? 1 : $clog2(C_ACT_DEPTH);
    localparam integer C_WGT_AW  = (C_WGT_DEPTH  <= 1) ? 1 : $clog2(C_WGT_DEPTH);
    localparam integer C_BIAS_AW = (C_BIAS_DEPTH <= 1) ? 1 : $clog2(C_BIAS_DEPTH);
    localparam integer C_CONN_AW = (C_CONN_DEPTH <= 1) ? 1 : $clog2(C_CONN_DEPTH);
    localparam integer C_OUT_COUNT = `TV_OUT_COUNT;

    logic c_start, c_busy, c_done, c_config_error;
    logic c_out_valid, c_out_ready;
    logic signed [DATA_WIDTH-1:0] c_out_data;
    logic [7:0] c_out_channel, c_out_y, c_out_x;

    logic c_act_we, c_wgt_we, c_bias_we, c_conn_we;
    logic [C_ACT_AW-1:0]  c_act_addr;
    logic [C_WGT_AW-1:0]  c_wgt_addr;
    logic [C_BIAS_AW-1:0] c_bias_addr;
    logic [C_CONN_AW-1:0] c_conn_addr;
    logic signed [DATA_WIDTH-1:0] c_act_data, c_wgt_data;
    logic signed [ACC_WIDTH-1:0]  c_bias_data;
    logic c_conn_data;

    logic [7:0]  c_act_vec  [0:C_ACT_DEPTH-1];
    logic [7:0]  c_wgt_vec  [0:C_WGT_DEPTH-1];
    logic [31:0] c_bias_vec [0:C_BIAS_DEPTH-1];
    logic [7:0]  c_conn_vec [0:C_CONN_DEPTH-1];
    logic [7:0]  c_exp_vec  [0:C_OUT_COUNT-1];

    // Captured reference stream.
    logic [7:0] c_ref_data [0:C_OUT_COUNT-1];
    logic [7:0] c_ref_ch   [0:C_OUT_COUNT-1];
    logic [7:0] c_ref_y    [0:C_OUT_COUNT-1];
    logic [7:0] c_ref_x    [0:C_OUT_COUNT-1];

    integer c_idx;
    integer c_done_count;
    reg     c_capture;

    conv2d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .MAX_IN_W  (`TV_IN_W),
        .MAX_IN_H  (`TV_IN_H),
        .MAX_IN_CH (`TV_IN_CH),
        .MAX_OUT_CH(`TV_OUT_CH),
        .ACT_DEPTH (C_ACT_DEPTH),
        .WGT_DEPTH (C_WGT_DEPTH),
        .BIAS_DEPTH(C_BIAS_DEPTH),
        .CONN_DEPTH(C_CONN_DEPTH)
    ) u_conv (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(c_start),
        .cfg_in_w_i(8'(`TV_IN_W)), .cfg_in_h_i(8'(`TV_IN_H)),
        .cfg_in_ch_i(8'(`TV_IN_CH)), .cfg_out_ch_i(8'(`TV_OUT_CH)),
        .cfg_shift_i(6'(`TV_SHIFT)), .cfg_relu_en_i(`TV_RELU),
        .busy_o(c_busy), .done_o(c_done), .config_error_o(c_config_error),
        .load_act_we_i(c_act_we),   .load_act_addr_i(c_act_addr),   .load_act_data_i(c_act_data),
        .load_wgt_we_i(c_wgt_we),   .load_wgt_addr_i(c_wgt_addr),   .load_wgt_data_i(c_wgt_data),
        .load_bias_we_i(c_bias_we), .load_bias_addr_i(c_bias_addr), .load_bias_data_i(c_bias_data),
        .load_conn_we_i(c_conn_we), .load_conn_addr_i(c_conn_addr), .load_conn_data_i(c_conn_data),
        .out_valid_o(c_out_valid), .out_ready_i(c_out_ready),
        .out_data_o(c_out_data), .out_channel_o(c_out_channel),
        .out_y_o(c_out_y), .out_x_o(c_out_x)
    );

    stream_hold_check #(.PAYLOAD_WIDTH(32)) u_c_hold (
        .clk_i(clk), .rst_ni(rst_n),
        .valid_i(c_out_valid), .ready_i(c_out_ready),
        .payload_i({c_out_channel, c_out_y, c_out_x, c_out_data})
    );

    // c_idx is incremented here on posedge and cleared by the stimulus process
    // on negedge, both non-blocking, so the two never write it in the same
    // timestep. Same discipline as tb_config_guard's done flags.
    always @(posedge clk) begin
        if (c_done) c_done_count <= c_done_count + 1;

        if (rst_n && c_out_valid && c_out_ready) begin
            if (c_idx >= C_OUT_COUNT) begin
                $fatal(1, "conv2d_engine produced more than %0d beats", C_OUT_COUNT);
            end
            if (c_capture) begin
                c_ref_data[c_idx] <= c_out_data;
                c_ref_ch[c_idx]   <= c_out_channel;
                c_ref_y[c_idx]    <= c_out_y;
                c_ref_x[c_idx]    <= c_out_x;
                if ($signed(c_out_data) !== $signed(c_exp_vec[c_idx])) begin
                    $fatal(1,
                        "conv2d_engine REF beat %0d: got %0d, golden vector says %0d",
                        c_idx, $signed(c_out_data), $signed(c_exp_vec[c_idx]));
                end
            end else begin
                if (c_out_data !== c_ref_data[c_idx]
                    || c_out_channel !== c_ref_ch[c_idx]
                    || c_out_y !== c_ref_y[c_idx]
                    || c_out_x !== c_ref_x[c_idx]) begin
                    $fatal(1,
                        "conv2d_engine beat %0d diverged from the unstalled reference: got data %0d ch %0d y %0d x %0d, expected data %0d ch %0d y %0d x %0d",
                        c_idx, $signed(c_out_data), c_out_channel, c_out_y, c_out_x,
                        $signed(c_ref_data[c_idx]), c_ref_ch[c_idx], c_ref_y[c_idx], c_ref_x[c_idx]);
                end
            end
            c_idx <= c_idx + 1;
        end
    end

    assign c_out_ready = rnd_ready;

    // ==================================================================
    // avg_pool2x2_stream
    // ==================================================================
    localparam integer P_ACT_DEPTH = `TV_POOL_ACT_COUNT;
    localparam integer P_ACT_AW = (P_ACT_DEPTH <= 1) ? 1 : $clog2(P_ACT_DEPTH);
    localparam integer P_OUT_COUNT = `TV_POOL_OUT_COUNT;

    logic p_start, p_busy, p_done, p_config_error;
    logic p_out_valid, p_out_ready;
    logic signed [DATA_WIDTH-1:0] p_out_data;
    logic [7:0] p_out_channel, p_out_y, p_out_x;

    logic p_act_we;
    logic [P_ACT_AW-1:0] p_act_addr;
    logic signed [DATA_WIDTH-1:0] p_act_data;

    logic [7:0] p_act_vec [0:P_ACT_DEPTH-1];
    logic [7:0] p_exp_vec [0:P_OUT_COUNT-1];

    logic [7:0] p_ref_data [0:P_OUT_COUNT-1];
    logic [7:0] p_ref_ch   [0:P_OUT_COUNT-1];
    logic [7:0] p_ref_y    [0:P_OUT_COUNT-1];
    logic [7:0] p_ref_x    [0:P_OUT_COUNT-1];

    integer p_idx;
    integer p_done_count;
    reg     p_capture;

    avg_pool2x2_stream #(
        .DATA_WIDTH(DATA_WIDTH),
        .MAX_IN_W  (`TV_POOL_IN_W),
        .MAX_IN_H  (`TV_POOL_IN_H),
        .MAX_IN_CH (`TV_POOL_IN_CH),
        .ACT_DEPTH (P_ACT_DEPTH)
    ) u_pool (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(p_start),
        .cfg_in_w_i(8'(`TV_POOL_IN_W)), .cfg_in_h_i(8'(`TV_POOL_IN_H)),
        .cfg_in_ch_i(8'(`TV_POOL_IN_CH)),
        .busy_o(p_busy), .done_o(p_done), .config_error_o(p_config_error),
        .load_act_we_i(p_act_we), .load_act_addr_i(p_act_addr), .load_act_data_i(p_act_data),
        .out_valid_o(p_out_valid), .out_ready_i(p_out_ready),
        .out_data_o(p_out_data), .out_channel_o(p_out_channel),
        .out_y_o(p_out_y), .out_x_o(p_out_x)
    );

    stream_hold_check #(.PAYLOAD_WIDTH(32)) u_p_hold (
        .clk_i(clk), .rst_ni(rst_n),
        .valid_i(p_out_valid), .ready_i(p_out_ready),
        .payload_i({p_out_channel, p_out_y, p_out_x, p_out_data})
    );

    always @(posedge clk) begin
        if (p_done) p_done_count <= p_done_count + 1;

        if (rst_n && p_out_valid && p_out_ready) begin
            if (p_idx >= P_OUT_COUNT) begin
                $fatal(1, "avg_pool2x2_stream produced more than %0d beats", P_OUT_COUNT);
            end
            if (p_capture) begin
                p_ref_data[p_idx] <= p_out_data;
                p_ref_ch[p_idx]   <= p_out_channel;
                p_ref_y[p_idx]    <= p_out_y;
                p_ref_x[p_idx]    <= p_out_x;
                if ($signed(p_out_data) !== $signed(p_exp_vec[p_idx])) begin
                    $fatal(1,
                        "avg_pool2x2_stream REF beat %0d: got %0d, golden vector says %0d",
                        p_idx, $signed(p_out_data), $signed(p_exp_vec[p_idx]));
                end
            end else begin
                if (p_out_data !== p_ref_data[p_idx]
                    || p_out_channel !== p_ref_ch[p_idx]
                    || p_out_y !== p_ref_y[p_idx]
                    || p_out_x !== p_ref_x[p_idx]) begin
                    $fatal(1,
                        "avg_pool2x2_stream beat %0d diverged from the unstalled reference: got data %0d ch %0d y %0d x %0d, expected data %0d ch %0d y %0d x %0d",
                        p_idx, $signed(p_out_data), p_out_channel, p_out_y, p_out_x,
                        $signed(p_ref_data[p_idx]), p_ref_ch[p_idx], p_ref_y[p_idx], p_ref_x[p_idx]);
                end
            end
            p_idx <= p_idx + 1;
        end
    end

    assign p_out_ready = rnd_ready;

    // ==================================================================
    // dense_engine (F6 configuration)
    // ==================================================================
    localparam integer D_ACT_DEPTH  = `TV_F6_ACT_COUNT;
    localparam integer D_WGT_DEPTH  = `TV_F6_WGT_COUNT;
    localparam integer D_BIAS_DEPTH = `TV_F6_BIAS_COUNT;
    localparam integer D_ACT_AW  = (D_ACT_DEPTH  <= 1) ? 1 : $clog2(D_ACT_DEPTH);
    localparam integer D_WGT_AW  = (D_WGT_DEPTH  <= 1) ? 1 : $clog2(D_WGT_DEPTH);
    localparam integer D_BIAS_AW = (D_BIAS_DEPTH <= 1) ? 1 : $clog2(D_BIAS_DEPTH);
    localparam integer D_OUT_COUNT = `TV_F6_OUT_COUNT;

    logic d_start, d_busy, d_done, d_config_error;
    logic d_out_valid, d_out_ready;
    logic signed [DATA_WIDTH-1:0] d_out_data;
    logic signed [ACC_WIDTH-1:0]  d_out_acc;
    logic [7:0] d_out_index;

    logic d_act_we, d_wgt_we, d_bias_we;
    logic [D_ACT_AW-1:0]  d_act_addr;
    logic [D_WGT_AW-1:0]  d_wgt_addr;
    logic [D_BIAS_AW-1:0] d_bias_addr;
    logic signed [DATA_WIDTH-1:0] d_act_data, d_wgt_data;
    logic signed [ACC_WIDTH-1:0]  d_bias_data;

    logic [7:0]  d_act_vec  [0:D_ACT_DEPTH-1];
    logic [7:0]  d_wgt_vec  [0:D_WGT_DEPTH-1];
    logic [31:0] d_bias_vec [0:D_BIAS_DEPTH-1];
    logic [7:0]  d_exp_vec  [0:D_OUT_COUNT-1];

    logic [7:0]  d_ref_data  [0:D_OUT_COUNT-1];
    logic [31:0] d_ref_acc   [0:D_OUT_COUNT-1];
    logic [7:0]  d_ref_index [0:D_OUT_COUNT-1];

    integer d_idx;
    integer d_done_count;
    reg     d_capture;

    dense_engine #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .MAX_IN_LEN (`TV_F6_IN_LEN),
        .MAX_OUT_LEN(`TV_F6_OUT_LEN),
        .ACT_DEPTH  (D_ACT_DEPTH),
        .WGT_DEPTH  (D_WGT_DEPTH),
        .BIAS_DEPTH (D_BIAS_DEPTH)
    ) u_dense (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(d_start),
        .cfg_in_len_i(8'(`TV_F6_IN_LEN)), .cfg_out_len_i(8'(`TV_F6_OUT_LEN)),
        .cfg_shift_i(6'(`TV_F6_SHIFT)), .cfg_relu_en_i(`TV_F6_RELU),
        .busy_o(d_busy), .done_o(d_done), .config_error_o(d_config_error),
        .load_act_we_i(d_act_we),   .load_act_addr_i(d_act_addr),   .load_act_data_i(d_act_data),
        .load_wgt_we_i(d_wgt_we),   .load_wgt_addr_i(d_wgt_addr),   .load_wgt_data_i(d_wgt_data),
        .load_bias_we_i(d_bias_we), .load_bias_addr_i(d_bias_addr), .load_bias_data_i(d_bias_data),
        .out_valid_o(d_out_valid), .out_ready_i(d_out_ready),
        .out_data_o(d_out_data), .out_acc_o(d_out_acc), .out_index_o(d_out_index)
    );

    stream_hold_check #(.PAYLOAD_WIDTH(48)) u_d_hold (
        .clk_i(clk), .rst_ni(rst_n),
        .valid_i(d_out_valid), .ready_i(d_out_ready),
        .payload_i({d_out_index, d_out_acc, d_out_data})
    );

    always @(posedge clk) begin
        if (d_done) d_done_count <= d_done_count + 1;

        if (rst_n && d_out_valid && d_out_ready) begin
            if (d_idx >= D_OUT_COUNT) begin
                $fatal(1, "dense_engine produced more than %0d beats", D_OUT_COUNT);
            end
            if (d_capture) begin
                d_ref_data[d_idx]  <= d_out_data;
                d_ref_acc[d_idx]   <= d_out_acc;
                d_ref_index[d_idx] <= d_out_index;
                if ($signed(d_out_data) !== $signed(d_exp_vec[d_idx])) begin
                    $fatal(1,
                        "dense_engine REF beat %0d: got %0d, golden vector says %0d",
                        d_idx, $signed(d_out_data), $signed(d_exp_vec[d_idx]));
                end
            end else begin
                // out_acc_o is compared as well as out_data_o: classifier_argmax
                // decides the predicted class from the raw accumulator, so an
                // accumulator that shifted under backpressure while the
                // saturated int8 score happened not to would change the
                // network's answer without changing this stream's data beats.
                if (d_out_data !== d_ref_data[d_idx]
                    || d_out_acc !== d_ref_acc[d_idx]
                    || d_out_index !== d_ref_index[d_idx]) begin
                    $fatal(1,
                        "dense_engine beat %0d diverged from the unstalled reference: got data %0d acc %0d index %0d, expected data %0d acc %0d index %0d",
                        d_idx, $signed(d_out_data), $signed(d_out_acc), d_out_index,
                        $signed(d_ref_data[d_idx]), $signed(d_ref_acc[d_idx]), d_ref_index[d_idx]);
                end
            end
            d_idx <= d_idx + 1;
        end
    end

    assign d_out_ready = rnd_ready;

    // ==================================================================
    // Stimulus
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

    task automatic load_pool;
        integer i;
        begin
            for (i = 0; i < P_ACT_DEPTH; i = i + 1) begin
                @(negedge clk);
                p_act_we = 1'b1; p_act_addr = i[P_ACT_AW-1:0]; p_act_data = p_act_vec[i];
            end
            @(negedge clk); p_act_we = 1'b0;
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
    // Advance to a cycle on which the engine has announced a beat that the
    // consumer has not accepted, so the abort lands on a pending output.
    //
    // This is not incidental sequencing. With out_ready_i tied high every
    // beat is accepted the cycle it appears, so out_valid_o is high for one
    // cycle per beat and an abort taken at an arbitrary moment almost always
    // finds it already low -- which makes the out_valid_o half of the
    // "went quiet" check below pass without ever testing anything. A mutation
    // that deleted out_valid_o from conv2d_engine's reset branch survived the
    // whole regression until the abort was pinned to this state.
    //
    // Sampled just after the posedge so both out_valid_o (updated on the
    // previous posedge) and out_ready_i (updated on the previous negedge) are
    // settled, and reset is then asserted on the following negedge, before any
    // clock edge can retire the pending beat.
    // ------------------------------------------------------------------
    // Bounded rather than a bare wait: if the abort point is ever moved past
    // the end of a stream, an unbounded loop here would hang and be reported
    // by the global watchdog as "an engine did not recover from reset", which
    // sends the reader looking at the design instead of at this constant.
    localparam integer PENDING_TIMEOUT = 500;

    task automatic wait_pending_conv;
        integer waited;
        begin
            waited = 0;
            while (!(c_out_valid === 1'b1 && c_out_ready === 1'b0)) begin
                @(posedge clk);
                #1;
                waited = waited + 1;
                if (waited > PENDING_TIMEOUT) begin
                    $fatal(1, "conv2d_engine never presented a stalled beat within %0d cycles of C_ABORT_AT -- the abort point is not inside the stream",
                           PENDING_TIMEOUT);
                end
            end
        end
    endtask

    task automatic wait_pending_pool;
        integer waited;
        begin
            waited = 0;
            while (!(p_out_valid === 1'b1 && p_out_ready === 1'b0)) begin
                @(posedge clk);
                #1;
                waited = waited + 1;
                if (waited > PENDING_TIMEOUT) begin
                    $fatal(1, "avg_pool2x2_stream never presented a stalled beat within %0d cycles of P_ABORT_AT -- the abort point is not inside the stream",
                           PENDING_TIMEOUT);
                end
            end
        end
    endtask

    task automatic wait_pending_dense;
        integer waited;
        begin
            waited = 0;
            while (!(d_out_valid === 1'b1 && d_out_ready === 1'b0)) begin
                @(posedge clk);
                #1;
                waited = waited + 1;
                if (waited > PENDING_TIMEOUT) begin
                    $fatal(1, "dense_engine never presented a stalled beat within %0d cycles of D_ABORT_AT -- the abort point is not inside the stream",
                           PENDING_TIMEOUT);
                end
            end
        end
    endtask

    // Assert reset while an engine is mid-stream, hold it, and release. The
    // "went quiet" checks are the reset-interruption policy itself: an
    // aborted engine must drop busy_o and out_valid_o, and must not emit a
    // done_o for work it never completed.
    task automatic abort_with_reset(input reg [24*8-1:0] who);
        begin
            @(negedge clk);
            rst_n = 1'b0;
            @(negedge clk);
            if (c_busy !== 1'b0 || p_busy !== 1'b0 || d_busy !== 1'b0
                || c_out_valid !== 1'b0 || p_out_valid !== 1'b0 || d_out_valid !== 1'b0
                || c_done !== 1'b0 || p_done !== 1'b0 || d_done !== 1'b0) begin
                $fatal(1, "%0s: engines did not go quiet under reset at time %0t", who, $time);
            end
            repeat (2) @(negedge clk);
            rst_n = 1'b1;
            repeat (2) @(negedge clk);
            if (c_busy !== 1'b0 || p_busy !== 1'b0 || d_busy !== 1'b0
                || c_out_valid !== 1'b0 || p_out_valid !== 1'b0 || d_out_valid !== 1'b0) begin
                $fatal(1, "%0s: engines are not idle after reset release at time %0t", who, $time);
            end
        end
    endtask

    integer c_stalls_before, c_stalls_after;
    integer p_stalls_before, p_stalls_after;
    integer d_stalls_before, d_stalls_after;
    integer c_done_before, p_done_before, d_done_before;
    integer c_beats_at_abort, p_beats_at_abort, d_beats_at_abort;

    initial begin
        $readmemh("vectors/act.hex",       c_act_vec);
        $readmemh("vectors/wgt.hex",       c_wgt_vec);
        $readmemh("vectors/bias.hex",      c_bias_vec);
        $readmemh("vectors/conn.hex",      c_conn_vec);
        $readmemh("vectors/expected.hex",  c_exp_vec);
        $readmemh("vectors/pool/act.hex",      p_act_vec);
        $readmemh("vectors/pool/expected.hex", p_exp_vec);
        $readmemh("vectors/f6/act.hex",      d_act_vec);
        $readmemh("vectors/f6/wgt.hex",      d_wgt_vec);
        $readmemh("vectors/f6/bias.hex",     d_bias_vec);
        $readmemh("vectors/f6/expected.hex", d_exp_vec);

        u_c_hold.set_label("conv2d_engine");
        u_p_hold.set_label("avg_pool2x2_stream");
        u_d_hold.set_label("dense_engine");

        clk = 1'b0;
        rst_n = 1'b0;
        stall_en = 1'b0;

        c_start = 1'b0; c_act_we = 1'b0; c_wgt_we = 1'b0; c_bias_we = 1'b0; c_conn_we = 1'b0;
        c_act_addr = '0; c_wgt_addr = '0; c_bias_addr = '0; c_conn_addr = '0;
        c_act_data = '0; c_wgt_data = '0; c_bias_data = '0; c_conn_data = 1'b0;
        c_idx = 0; c_done_count = 0; c_capture = 1'b1;

        p_start = 1'b0; p_act_we = 1'b0; p_act_addr = '0; p_act_data = '0;
        p_idx = 0; p_done_count = 0; p_capture = 1'b1;

        d_start = 1'b0; d_act_we = 1'b0; d_wgt_we = 1'b0; d_bias_we = 1'b0;
        d_act_addr = '0; d_wgt_addr = '0; d_bias_addr = '0;
        d_act_data = '0; d_wgt_data = '0; d_bias_data = '0;
        d_idx = 0; d_done_count = 0; d_capture = 1'b1;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        // Memories are written by unreset always blocks, so their contents
        // survive every reset this testbench asserts. Loading once up front is
        // deliberate: it means the ABORT phases also prove that a mid-operation
        // reset does not disturb loaded operands.
        load_conv();
        load_pool();
        load_dense();

        // ---------------- conv2d_engine ----------------
        run_conv("REF");
        $display("  conv2d_engine REF: %0d beats, oracle-checked against vectors/expected.hex", c_idx);

        stall_en = 1'b1;
        c_stalls_before = u_c_hold.stall_cycles;
        c_capture = 1'b0;
        run_conv("STALL");
        c_stalls_after = u_c_hold.stall_cycles;
        if ((c_stalls_after - c_stalls_before) <= 0) begin
            $fatal(1, "conv2d_engine STALL run never stalled -- the backpressure test is vacuous");
        end
        $display("  conv2d_engine STALL: %0d beats identical to REF across %0d stalled cycles",
                 c_idx, c_stalls_after - c_stalls_before);

        @(negedge clk);
        c_idx <= 0;
        @(negedge clk);
        // Snapshotted here rather than earlier on purpose. done_o is a
        // one-cycle pulse, and the counter above increments on the clock edge
        // *after* the edge that raised it -- so a snapshot taken the moment
        // run_conv returns still misses the run's own done, and the abort
        // check then blames this run for the previous one's pulse. Taking it
        // immediately before start_i removes the window entirely.
        c_done_before = c_done_count;
        c_start = 1'b1;
        @(negedge clk);
        c_start = 1'b0;
        wait (c_idx >= C_ABORT_AT);
        // Settle before reading busy_o: the wait above releases in the
        // non-blocking update region of the same edge that retires the beat,
        // where busy_o is being written too.
        @(negedge clk);
        if (c_busy !== 1'b1) begin
            $fatal(1, "conv2d_engine was not busy at the abort point -- the reset test is vacuous");
        end
        wait_pending_conv();
        c_beats_at_abort = c_idx;
        if (c_out_valid !== 1'b1) begin
            $fatal(1, "conv2d_engine had no pending beat at the abort point -- the out_valid_o half of the quiet check would be vacuous");
        end
        abort_with_reset("conv2d_engine");
        stall_en = 1'b0;
        if (c_done_count != c_done_before) begin
            $fatal(1, "conv2d_engine pulsed done_o for an aborted operation");
        end
        if (c_beats_at_abort <= 0 || c_beats_at_abort >= C_OUT_COUNT) begin
            $fatal(1, "conv2d_engine abort landed at beat %0d, outside the 1..%0d stream -- the reset test is vacuous",
                   c_beats_at_abort, C_OUT_COUNT - 1);
        end
        run_conv("RESTART");
        $display("  conv2d_engine ABORT: reset at beat %0d of %0d, restarted run identical to REF",
                 c_beats_at_abort, C_OUT_COUNT);
        $display("PASS tb_robustness: conv2d_engine is stall-invariant and recovers from a mid-stream reset");

        // ---------------- avg_pool2x2_stream ----------------
        run_pool("REF");
        $display("  avg_pool2x2_stream REF: %0d beats, oracle-checked against vectors/pool/expected.hex", p_idx);

        stall_en = 1'b1;
        p_stalls_before = u_p_hold.stall_cycles;
        p_capture = 1'b0;
        run_pool("STALL");
        p_stalls_after = u_p_hold.stall_cycles;
        if ((p_stalls_after - p_stalls_before) <= 0) begin
            $fatal(1, "avg_pool2x2_stream STALL run never stalled -- the backpressure test is vacuous");
        end
        $display("  avg_pool2x2_stream STALL: %0d beats identical to REF across %0d stalled cycles",
                 p_idx, p_stalls_after - p_stalls_before);

        @(negedge clk);
        p_idx <= 0;
        @(negedge clk);
        p_done_before = p_done_count;   // see the note in the conv2d abort above
        p_start = 1'b1;
        @(negedge clk);
        p_start = 1'b0;
        wait (p_idx >= P_ABORT_AT);
        @(negedge clk);   // settle -- see the note in the conv2d abort above
        if (p_busy !== 1'b1) begin
            $fatal(1, "avg_pool2x2_stream was not busy at the abort point -- the reset test is vacuous");
        end
        wait_pending_pool();
        p_beats_at_abort = p_idx;
        if (p_out_valid !== 1'b1) begin
            $fatal(1, "avg_pool2x2_stream had no pending beat at the abort point -- the out_valid_o half of the quiet check would be vacuous");
        end
        abort_with_reset("avg_pool2x2_stream");
        stall_en = 1'b0;
        if (p_done_count != p_done_before) begin
            $fatal(1, "avg_pool2x2_stream pulsed done_o for an aborted operation");
        end
        if (p_beats_at_abort <= 0 || p_beats_at_abort >= P_OUT_COUNT) begin
            $fatal(1, "avg_pool2x2_stream abort landed at beat %0d, outside the 1..%0d stream -- the reset test is vacuous",
                   p_beats_at_abort, P_OUT_COUNT - 1);
        end
        run_pool("RESTART");
        $display("  avg_pool2x2_stream ABORT: reset at beat %0d of %0d, restarted run identical to REF",
                 p_beats_at_abort, P_OUT_COUNT);
        $display("PASS tb_robustness: avg_pool2x2_stream is stall-invariant and recovers from a mid-stream reset");

        // ---------------- dense_engine ----------------
        run_dense("REF");
        $display("  dense_engine REF: %0d beats, oracle-checked against vectors/f6/expected.hex", d_idx);

        stall_en = 1'b1;
        d_stalls_before = u_d_hold.stall_cycles;
        d_capture = 1'b0;
        run_dense("STALL");
        d_stalls_after = u_d_hold.stall_cycles;
        if ((d_stalls_after - d_stalls_before) <= 0) begin
            $fatal(1, "dense_engine STALL run never stalled -- the backpressure test is vacuous");
        end
        $display("  dense_engine STALL: %0d beats identical to REF across %0d stalled cycles",
                 d_idx, d_stalls_after - d_stalls_before);

        @(negedge clk);
        d_idx <= 0;
        @(negedge clk);
        d_done_before = d_done_count;   // see the note in the conv2d abort above
        d_start = 1'b1;
        @(negedge clk);
        d_start = 1'b0;
        wait (d_idx >= D_ABORT_AT);
        @(negedge clk);   // settle -- see the note in the conv2d abort above
        if (d_busy !== 1'b1) begin
            $fatal(1, "dense_engine was not busy at the abort point -- the reset test is vacuous");
        end
        wait_pending_dense();
        d_beats_at_abort = d_idx;
        if (d_out_valid !== 1'b1) begin
            $fatal(1, "dense_engine had no pending beat at the abort point -- the out_valid_o half of the quiet check would be vacuous");
        end
        abort_with_reset("dense_engine");
        stall_en = 1'b0;
        if (d_done_count != d_done_before) begin
            $fatal(1, "dense_engine pulsed done_o for an aborted operation");
        end
        if (d_beats_at_abort <= 0 || d_beats_at_abort >= D_OUT_COUNT) begin
            $fatal(1, "dense_engine abort landed at beat %0d, outside the 1..%0d stream -- the reset test is vacuous",
                   d_beats_at_abort, D_OUT_COUNT - 1);
        end
        run_dense("RESTART");
        $display("  dense_engine ABORT: reset at beat %0d of %0d, restarted run identical to REF",
                 d_beats_at_abort, D_OUT_COUNT);
        $display("PASS tb_robustness: dense_engine is stall-invariant and recovers from a mid-stream reset");

        $display("PASS tb_robustness: 3 engines x {unstalled, randomly stalled, reset mid-stream} all agree beat for beat");
        $finish;
    end

    // One run of each engine: clear the beat index, pulse start, wait for
    // done, and check the full stream arrived. The comparison itself lives in
    // the per-engine always blocks above.
    task automatic run_conv(input reg [24*8-1:0] phase);
        begin
            @(negedge clk);
            c_idx <= 0;
            @(negedge clk);
            c_start = 1'b1;
            @(negedge clk);
            c_start = 1'b0;
            wait (c_done === 1'b1);
            @(negedge clk);
            if (c_idx != C_OUT_COUNT) begin
                $fatal(1, "conv2d_engine %0s: done_o after %0d beats, expected %0d",
                       phase, c_idx, C_OUT_COUNT);
            end
            if (c_config_error !== 1'b0) begin
                $fatal(1, "conv2d_engine %0s: config_error_o asserted on a legal config", phase);
            end
        end
    endtask

    task automatic run_pool(input reg [24*8-1:0] phase);
        begin
            @(negedge clk);
            p_idx <= 0;
            @(negedge clk);
            p_start = 1'b1;
            @(negedge clk);
            p_start = 1'b0;
            wait (p_done === 1'b1);
            @(negedge clk);
            if (p_idx != P_OUT_COUNT) begin
                $fatal(1, "avg_pool2x2_stream %0s: done_o after %0d beats, expected %0d",
                       phase, p_idx, P_OUT_COUNT);
            end
            if (p_config_error !== 1'b0) begin
                $fatal(1, "avg_pool2x2_stream %0s: config_error_o asserted on a legal config", phase);
            end
        end
    endtask

    task automatic run_dense(input reg [24*8-1:0] phase);
        begin
            @(negedge clk);
            d_idx <= 0;
            @(negedge clk);
            d_start = 1'b1;
            @(negedge clk);
            d_start = 1'b0;
            wait (d_done === 1'b1);
            @(negedge clk);
            if (d_idx != D_OUT_COUNT) begin
                $fatal(1, "dense_engine %0s: done_o after %0d beats, expected %0d",
                       phase, d_idx, D_OUT_COUNT);
            end
            if (d_config_error !== 1'b0) begin
                $fatal(1, "dense_engine %0s: config_error_o asserted on a legal config", phase);
            end
        end
    endtask

    initial begin
        repeat (WATCHDOG_CYCLES) @(posedge clk);
        $fatal(1, "Simulation timeout -- an engine did not recover from reset or a stream stalled forever");
    end
endmodule
