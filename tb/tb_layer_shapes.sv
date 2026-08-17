// ---------------------------------------------------------------------------
// tb_layer_shapes.sv -- conv2d_engine and dense_engine reconfigured across
// the real network's own C1/C3/C5/F6/classifier dimensions, each layer's
// beat stream checked against the golden model directly.
//
// What this adds over the existing per-module testbenches
// ---------------------------------------------------------------------
// Every other standalone conv2d_engine/dense_engine testbench in this project
// uses either a tiny synthetic shape (7x8x3->4, 10->5) or a channel/operand
// *extreme* (16 input channels, MAX_IN_LEN=120) -- never the shapes the real
// network is actually built from. C1 (1->6, 32x32), C3 (6->16, 14x14, the
// real sparse LeCun-98 connectivity), C5 (16->120, 5x5), F6 (120->84) and the
// classifier stage (84->10) are only ever driven by tb_lenet5_top.sv, and
// there the only oracle is "did the final 10-way argmax pick the same class"
// -- an argmax over 10 classes tolerates a great deal of intermediate error,
// so a layer can be quietly wrong and the top-level check still passes. This
// testbench reuses the *exact* image and weights tb_lenet5_top.sv checks
// (same seed, same random_deploy_parameters -- see golden/generate_vectors.py
// generate_layer_shapes_vectors()) but compares every layer's own beat stream
// against golden/deploy.py:deploy_forward_int8's intermediate C1/C3/C5/F6
// arrays.
//
// It also reconfigures one conv2d_engine instance from C1 -> C3 -> C5 and one
// dense_engine instance from F6 -> classifier, each transition a genuinely
// different shape. No existing testbench does this: tb_lenet5_top's two
// back-to-back inferences rerun the *same* five shapes both times, so cfg_*
// re-latching a config that differs from the previous run has never been
// exercised on either engine before this file.
//
// Two untested corners fall out of using the real shapes rather than inventing
// new ones: C5 (16->120) is the first *standalone* conv2d_engine test to
// reach MAX_OUT_CH=120, and F6 (120->84) is the first standalone dense_engine
// test to reach MAX_OUT_LEN=84 -- tb_extremes.sv reaches MAX_IN_CH and
// MAX_IN_LEN, never the *_OUT_* ceiling on either engine.
//
// Not claimed: this is not a trained network. random_deploy_parameters is
// explicit that its weights are shape-correct and not trained on MNIST -- see
// docs/VERIFICATION_PLAN.md's "trained-network C1/C3/C5" item, which stays
// open. This closes the dimension-coverage half of that gap, not the
// training half.
// ---------------------------------------------------------------------------

`timescale 1ns / 1ps

`include "vectors/config.svh"

module tb_layer_shapes;

    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;

    // Five loads (the biggest is C5's 48,000-weight ROM) plus five full
    // layer runs; ~163k cycles by hand estimate. Generous margin so a
    // deadlocked engine fails loudly instead of hanging a CI job.
    localparam integer WATCHDOG_CYCLES = 400000;

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
    // One conv2d_engine instance at its true default maximum (32x32x16
    // in, 120 out) -- the same sizing lenet5_top.sv itself uses -- driven
    // through C1, then C3, then C5.
    // ==================================================================
    localparam integer CONV_ACT_DEPTH  = 32 * 32 * 16;
    localparam integer CONV_WGT_DEPTH  = 120 * 16 * 25;
    localparam integer CONV_BIAS_DEPTH = 120;
    localparam integer CONV_CONN_DEPTH = 120 * 16;
    localparam integer CONV_ACT_AW  = $clog2(CONV_ACT_DEPTH);
    localparam integer CONV_WGT_AW  = $clog2(CONV_WGT_DEPTH);
    localparam integer CONV_BIAS_AW = $clog2(CONV_BIAS_DEPTH);
    localparam integer CONV_CONN_AW = $clog2(CONV_CONN_DEPTH);

    logic c_start, c_busy, c_done, c_config_error;
    logic [7:0] c_cfg_in_w, c_cfg_in_h, c_cfg_in_ch, c_cfg_out_ch;
    logic [5:0] c_cfg_shift;
    logic       c_cfg_relu;
    logic c_out_valid;
    logic signed [DATA_WIDTH-1:0] c_out_data;
    logic [7:0] c_out_channel, c_out_y, c_out_x;

    reg c_act_we, c_wgt_we, c_bias_we, c_conn_we, c_conn_data;
    reg [CONV_ACT_AW-1:0]  c_act_addr;
    reg [CONV_WGT_AW-1:0]  c_wgt_addr;
    reg [CONV_BIAS_AW-1:0] c_bias_addr;
    reg [CONV_CONN_AW-1:0] c_conn_addr;
    reg signed [DATA_WIDTH-1:0] c_act_data, c_wgt_data;
    reg signed [ACC_WIDTH-1:0]  c_bias_data;

    conv2d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_conv (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(c_start),
        .cfg_in_w_i(c_cfg_in_w), .cfg_in_h_i(c_cfg_in_h),
        .cfg_in_ch_i(c_cfg_in_ch), .cfg_out_ch_i(c_cfg_out_ch),
        .cfg_shift_i(c_cfg_shift), .cfg_relu_en_i(c_cfg_relu),
        .busy_o(c_busy), .done_o(c_done), .config_error_o(c_config_error),
        .load_act_we_i(c_act_we),   .load_act_addr_i(c_act_addr),   .load_act_data_i(c_act_data),
        .load_wgt_we_i(c_wgt_we),   .load_wgt_addr_i(c_wgt_addr),   .load_wgt_data_i(c_wgt_data),
        .load_bias_we_i(c_bias_we), .load_bias_addr_i(c_bias_addr), .load_bias_data_i(c_bias_data),
        .load_conn_we_i(c_conn_we), .load_conn_addr_i(c_conn_addr), .load_conn_data_i(c_conn_data),
        .out_valid_o(c_out_valid), .out_ready_i(1'b1),
        .out_data_o(c_out_data), .out_channel_o(c_out_channel),
        .out_y_o(c_out_y), .out_x_o(c_out_x)
    );

    // Live C3 sparse-connectivity generation -- the same RTL module
    // lenet5_top.sv itself walks, not a hand-transcribed copy. Any
    // disagreement with golden/lenet5.py:c3_connectivity() (which produced
    // the expected vectors) shows up as a wrong C3 beat, not a silent
    // mismatch between two independent transcriptions.
    logic [3:0] c3_output_map;
    logic [2:0] c3_input_map;
    logic       c3_conn_bit;

    lenet5_c3_connectivity u_c3_connectivity (
        .output_map_i(c3_output_map),
        .input_map_i (c3_input_map),
        .connected_o (c3_conn_bit)
    );

    // ==================================================================
    // One dense_engine instance at its true default maximum (120 in,
    // 84 out) driven through F6, then the classifier stage.
    // ==================================================================
    localparam integer DENSE_ACT_DEPTH  = 120;
    localparam integer DENSE_WGT_DEPTH  = 84 * 120;
    localparam integer DENSE_BIAS_DEPTH = 84;
    localparam integer DENSE_ACT_AW  = $clog2(DENSE_ACT_DEPTH);
    localparam integer DENSE_WGT_AW  = $clog2(DENSE_WGT_DEPTH);
    localparam integer DENSE_BIAS_AW = $clog2(DENSE_BIAS_DEPTH);

    logic d_start, d_busy, d_done, d_config_error;
    logic [7:0] d_cfg_in_len, d_cfg_out_len;
    logic [5:0] d_cfg_shift;
    logic       d_cfg_relu;
    logic d_out_valid;
    logic signed [DATA_WIDTH-1:0] d_out_data;
    logic signed [ACC_WIDTH-1:0]  d_out_acc;
    logic [7:0] d_out_index;

    reg d_act_we, d_wgt_we, d_bias_we;
    reg [DENSE_ACT_AW-1:0]  d_act_addr;
    reg [DENSE_WGT_AW-1:0]  d_wgt_addr;
    reg [DENSE_BIAS_AW-1:0] d_bias_addr;
    reg signed [DATA_WIDTH-1:0] d_act_data, d_wgt_data;
    reg signed [ACC_WIDTH-1:0]  d_bias_data;

    dense_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_dense (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(d_start),
        .cfg_in_len_i(d_cfg_in_len), .cfg_out_len_i(d_cfg_out_len),
        .cfg_shift_i(d_cfg_shift), .cfg_relu_en_i(d_cfg_relu),
        .busy_o(d_busy), .done_o(d_done), .config_error_o(d_config_error),
        .load_act_we_i(d_act_we),   .load_act_addr_i(d_act_addr),   .load_act_data_i(d_act_data),
        .load_wgt_we_i(d_wgt_we),   .load_wgt_addr_i(d_wgt_addr),   .load_wgt_data_i(d_wgt_data),
        .load_bias_we_i(d_bias_we), .load_bias_addr_i(d_bias_addr), .load_bias_data_i(d_bias_data),
        .out_valid_o(d_out_valid), .out_ready_i(1'b1),
        .out_data_o(d_out_data), .out_acc_o(d_out_acc), .out_index_o(d_out_index)
    );

    // ==================================================================
    // Per-layer vector storage and expected-beat comparators
    // ==================================================================
    reg signed [DATA_WIDTH-1:0] c1_act_vec  [0:`TV_LS_C1_ACT_COUNT-1];
    reg signed [DATA_WIDTH-1:0] c1_wgt_vec  [0:`TV_LS_C1_WGT_COUNT-1];
    reg signed [ACC_WIDTH-1:0]  c1_bias_vec [0:`TV_LS_C1_BIAS_COUNT-1];
    reg signed [DATA_WIDTH-1:0] c1_exp_vec  [0:`TV_LS_C1_OUT_COUNT-1];

    reg signed [DATA_WIDTH-1:0] c3_act_vec  [0:`TV_LS_C3_ACT_COUNT-1];
    reg signed [DATA_WIDTH-1:0] c3_wgt_vec  [0:`TV_LS_C3_WGT_COUNT-1];
    reg signed [ACC_WIDTH-1:0]  c3_bias_vec [0:`TV_LS_C3_BIAS_COUNT-1];
    reg signed [DATA_WIDTH-1:0] c3_exp_vec  [0:`TV_LS_C3_OUT_COUNT-1];

    reg signed [DATA_WIDTH-1:0] c5_act_vec  [0:`TV_LS_C5_ACT_COUNT-1];
    reg signed [DATA_WIDTH-1:0] c5_wgt_vec  [0:`TV_LS_C5_WGT_COUNT-1];
    reg signed [ACC_WIDTH-1:0]  c5_bias_vec [0:`TV_LS_C5_BIAS_COUNT-1];
    reg signed [DATA_WIDTH-1:0] c5_exp_vec  [0:`TV_LS_C5_OUT_COUNT-1];

    reg signed [DATA_WIDTH-1:0] f6_act_vec  [0:`TV_LS_F6_ACT_COUNT-1];
    reg signed [DATA_WIDTH-1:0] f6_wgt_vec  [0:`TV_LS_F6_WGT_COUNT-1];
    reg signed [ACC_WIDTH-1:0]  f6_bias_vec [0:`TV_LS_F6_BIAS_COUNT-1];
    reg signed [DATA_WIDTH-1:0] f6_exp_vec  [0:`TV_LS_F6_OUT_COUNT-1];

    reg signed [DATA_WIDTH-1:0] cls_act_vec  [0:`TV_LS_CLS_ACT_COUNT-1];
    reg signed [DATA_WIDTH-1:0] cls_wgt_vec  [0:`TV_LS_CLS_WGT_COUNT-1];
    reg signed [ACC_WIDTH-1:0]  cls_bias_vec [0:`TV_LS_CLS_BIAS_COUNT-1];
    reg signed [DATA_WIDTH-1:0] cls_exp_vec  [0:`TV_LS_CLS_OUT_COUNT-1];
    reg signed [63:0]           cls_acc_vec  [0:`TV_LS_CLS_OUT_COUNT-1];

    // conv_phase/dense_phase select which expected array the comparator
    // below reads from; set once before each run() call, matching
    // tb_extremes.sv's c_cancel_phase idiom generalised to five layers.
    integer conv_phase;   // 0 = C1, 1 = C3, 2 = C5
    integer dense_phase;  // 0 = F6, 1 = classifier
    integer c_idx = 0;
    integer d_idx = 0;
    integer c3_conn_ones_seen = 0;
    integer cls_winner;
    integer cls_best_acc;

    always @(posedge clk) begin
        if (rst_n && c_out_valid) begin
            case (conv_phase)
                0: begin
                    if (c_idx >= `TV_LS_C1_OUT_COUNT)
                        $fatal(1, "C1: conv2d_engine produced more than %0d beats", `TV_LS_C1_OUT_COUNT);
                    if ($signed(c_out_data) !== $signed(c1_exp_vec[c_idx]))
                        $fatal(1, "C1 beat %0d: got %0d, golden model says %0d",
                               c_idx, $signed(c_out_data), $signed(c1_exp_vec[c_idx]));
                end
                1: begin
                    if (c_idx >= `TV_LS_C3_OUT_COUNT)
                        $fatal(1, "C3: conv2d_engine produced more than %0d beats", `TV_LS_C3_OUT_COUNT);
                    if ($signed(c_out_data) !== $signed(c3_exp_vec[c_idx]))
                        $fatal(1, "C3 beat %0d: got %0d, golden model says %0d -- reconfiguring conv2d_engine from C1 to C3 (or the real sparse connectivity) is suspect",
                               c_idx, $signed(c_out_data), $signed(c3_exp_vec[c_idx]));
                end
                default: begin
                    if (c_idx >= `TV_LS_C5_OUT_COUNT)
                        $fatal(1, "C5: conv2d_engine produced more than %0d beats", `TV_LS_C5_OUT_COUNT);
                    if ($signed(c_out_data) !== $signed(c5_exp_vec[c_idx]))
                        $fatal(1, "C5 beat %0d: got %0d, golden model says %0d -- reconfiguring conv2d_engine from C3 to C5 is suspect",
                               c_idx, $signed(c_out_data), $signed(c5_exp_vec[c_idx]));
                end
            endcase
            c_idx <= c_idx + 1;
        end
    end

    always @(posedge clk) begin
        if (rst_n && d_out_valid) begin
            if (dense_phase == 0) begin
                if (d_idx >= `TV_LS_F6_OUT_COUNT)
                    $fatal(1, "F6: dense_engine produced more than %0d beats", `TV_LS_F6_OUT_COUNT);
                if ($signed(d_out_data) !== $signed(f6_exp_vec[d_idx]))
                    $fatal(1, "F6 beat %0d: got %0d, golden model says %0d",
                           d_idx, $signed(d_out_data), $signed(f6_exp_vec[d_idx]));
            end else begin
                if (d_idx >= `TV_LS_CLS_OUT_COUNT)
                    $fatal(1, "classifier: dense_engine produced more than %0d beats", `TV_LS_CLS_OUT_COUNT);
                if ($signed(d_out_data) !== $signed(cls_exp_vec[d_idx]))
                    $fatal(1, "classifier beat %0d: got %0d, golden model says %0d -- reconfiguring dense_engine from F6 to the classifier shape is suspect",
                           d_idx, $signed(d_out_data), $signed(cls_exp_vec[d_idx]));
                if ($signed({{32{d_out_acc[ACC_WIDTH-1]}}, d_out_acc}) !== cls_acc_vec[d_idx])
                    $fatal(1, "classifier beat %0d: out_acc_o = %0d, golden model says %0d",
                           d_idx, $signed(d_out_acc), cls_acc_vec[d_idx]);
                if (d_idx == 0 || d_out_acc > cls_best_acc) begin
                    cls_best_acc <= d_out_acc;
                    cls_winner   <= d_idx;
                end
            end
            d_idx <= d_idx + 1;
        end
    end

    // ==================================================================
    // Load tasks -- one per layer, mirroring tb_extremes.sv's per-shape
    // load_* tasks rather than a generically-sized helper, since each
    // array has its own compile-time size.
    // ==================================================================
    task automatic load_c1;
        integer i;
        begin
            for (i = 0; i < `TV_LS_C1_ACT_COUNT; i = i + 1) begin
                @(negedge clk);
                c_act_we = 1'b1; c_act_addr = i[CONV_ACT_AW-1:0]; c_act_data = c1_act_vec[i];
            end
            @(negedge clk); c_act_we = 1'b0;
            for (i = 0; i < `TV_LS_C1_WGT_COUNT; i = i + 1) begin
                @(negedge clk);
                c_wgt_we = 1'b1; c_wgt_addr = i[CONV_WGT_AW-1:0]; c_wgt_data = c1_wgt_vec[i];
            end
            @(negedge clk); c_wgt_we = 1'b0;
            for (i = 0; i < `TV_LS_C1_BIAS_COUNT; i = i + 1) begin
                @(negedge clk);
                c_bias_we = 1'b1; c_bias_addr = i[CONV_BIAS_AW-1:0]; c_bias_data = c1_bias_vec[i];
            end
            @(negedge clk); c_bias_we = 1'b0;
            // C1 is fully connected (1 input channel): constant-1 fill,
            // exactly like lenet5_top.sv drives its own C1 stage.
            for (i = 0; i < `TV_LS_C1_OUT_CH; i = i + 1) begin
                @(negedge clk);
                c_conn_we = 1'b1; c_conn_addr = i[CONV_CONN_AW-1:0]; c_conn_data = 1'b1;
            end
            @(negedge clk); c_conn_we = 1'b0;
        end
    endtask

    task automatic load_c3;
        integer i;
        begin
            for (i = 0; i < `TV_LS_C3_ACT_COUNT; i = i + 1) begin
                @(negedge clk);
                c_act_we = 1'b1; c_act_addr = i[CONV_ACT_AW-1:0]; c_act_data = c3_act_vec[i];
            end
            @(negedge clk); c_act_we = 1'b0;
            for (i = 0; i < `TV_LS_C3_WGT_COUNT; i = i + 1) begin
                @(negedge clk);
                c_wgt_we = 1'b1; c_wgt_addr = i[CONV_WGT_AW-1:0]; c_wgt_data = c3_wgt_vec[i];
            end
            @(negedge clk); c_wgt_we = 1'b0;
            for (i = 0; i < `TV_LS_C3_BIAS_COUNT; i = i + 1) begin
                @(negedge clk);
                c_bias_we = 1'b1; c_bias_addr = i[CONV_BIAS_AW-1:0]; c_bias_data = c3_bias_vec[i];
            end
            @(negedge clk); c_bias_we = 1'b0;
            // Real sparse LeCun-98 table, walked live via lenet5_c3_connectivity
            // -- same addressing lenet5_top.sv uses (output_map = idx/6,
            // input_map = idx%6).
            c3_conn_ones_seen = 0;
            for (i = 0; i < `TV_LS_C3_IN_CH * `TV_LS_C3_OUT_CH; i = i + 1) begin
                c3_output_map = (i / `TV_LS_C3_IN_CH);
                c3_input_map  = (i % `TV_LS_C3_IN_CH);
                #1;
                if (c3_conn_bit) c3_conn_ones_seen = c3_conn_ones_seen + 1;
                @(negedge clk);
                c_conn_we = 1'b1; c_conn_addr = i[CONV_CONN_AW-1:0]; c_conn_data = c3_conn_bit;
            end
            @(negedge clk); c_conn_we = 1'b0;
            if (c3_conn_ones_seen != `TV_LS_C3_CONN_ONES) begin
                $fatal(1, "C3 connectivity: live lenet5_c3_connectivity produced %0d connections, golden/lenet5.py:c3_connectivity() says %0d -- this tier's central claim (the real sparse table gets a strong check) does not hold",
                       c3_conn_ones_seen, `TV_LS_C3_CONN_ONES);
            end
        end
    endtask

    task automatic load_c5;
        integer i;
        begin
            for (i = 0; i < `TV_LS_C5_ACT_COUNT; i = i + 1) begin
                @(negedge clk);
                c_act_we = 1'b1; c_act_addr = i[CONV_ACT_AW-1:0]; c_act_data = c5_act_vec[i];
            end
            @(negedge clk); c_act_we = 1'b0;
            for (i = 0; i < `TV_LS_C5_WGT_COUNT; i = i + 1) begin
                @(negedge clk);
                c_wgt_we = 1'b1; c_wgt_addr = i[CONV_WGT_AW-1:0]; c_wgt_data = c5_wgt_vec[i];
            end
            @(negedge clk); c_wgt_we = 1'b0;
            for (i = 0; i < `TV_LS_C5_BIAS_COUNT; i = i + 1) begin
                @(negedge clk);
                c_bias_we = 1'b1; c_bias_addr = i[CONV_BIAS_AW-1:0]; c_bias_data = c5_bias_vec[i];
            end
            @(negedge clk); c_bias_we = 1'b0;
            // C5 is fully connected (real LeNet-5 topology): constant-1 fill.
            for (i = 0; i < `TV_LS_C5_IN_CH * `TV_LS_C5_OUT_CH; i = i + 1) begin
                @(negedge clk);
                c_conn_we = 1'b1; c_conn_addr = i[CONV_CONN_AW-1:0]; c_conn_data = 1'b1;
            end
            @(negedge clk); c_conn_we = 1'b0;
        end
    endtask

    task automatic load_f6;
        integer i;
        begin
            for (i = 0; i < `TV_LS_F6_ACT_COUNT; i = i + 1) begin
                @(negedge clk);
                d_act_we = 1'b1; d_act_addr = i[DENSE_ACT_AW-1:0]; d_act_data = f6_act_vec[i];
            end
            @(negedge clk); d_act_we = 1'b0;
            for (i = 0; i < `TV_LS_F6_WGT_COUNT; i = i + 1) begin
                @(negedge clk);
                d_wgt_we = 1'b1; d_wgt_addr = i[DENSE_WGT_AW-1:0]; d_wgt_data = f6_wgt_vec[i];
            end
            @(negedge clk); d_wgt_we = 1'b0;
            for (i = 0; i < `TV_LS_F6_BIAS_COUNT; i = i + 1) begin
                @(negedge clk);
                d_bias_we = 1'b1; d_bias_addr = i[DENSE_BIAS_AW-1:0]; d_bias_data = f6_bias_vec[i];
            end
            @(negedge clk); d_bias_we = 1'b0;
        end
    endtask

    task automatic load_cls;
        integer i;
        begin
            for (i = 0; i < `TV_LS_CLS_ACT_COUNT; i = i + 1) begin
                @(negedge clk);
                d_act_we = 1'b1; d_act_addr = i[DENSE_ACT_AW-1:0]; d_act_data = cls_act_vec[i];
            end
            @(negedge clk); d_act_we = 1'b0;
            for (i = 0; i < `TV_LS_CLS_WGT_COUNT; i = i + 1) begin
                @(negedge clk);
                d_wgt_we = 1'b1; d_wgt_addr = i[DENSE_WGT_AW-1:0]; d_wgt_data = cls_wgt_vec[i];
            end
            @(negedge clk); d_wgt_we = 1'b0;
            for (i = 0; i < `TV_LS_CLS_BIAS_COUNT; i = i + 1) begin
                @(negedge clk);
                d_bias_we = 1'b1; d_bias_addr = i[DENSE_BIAS_AW-1:0]; d_bias_data = cls_bias_vec[i];
            end
            @(negedge clk); d_bias_we = 1'b0;
        end
    endtask

    // ==================================================================
    // Run tasks
    // ==================================================================
    task automatic run_conv(
        input [7:0] in_w, input [7:0] in_h, input [7:0] in_ch, input [7:0] out_ch,
        input [5:0] shift, input relu, input integer expected_count, input [24*8-1:0] who
    );
        begin
            c_idx <= 0;
            @(negedge clk);
            c_cfg_in_w = in_w; c_cfg_in_h = in_h;
            c_cfg_in_ch = in_ch; c_cfg_out_ch = out_ch;
            c_cfg_shift = shift; c_cfg_relu = relu;
            c_start = 1'b1; @(negedge clk); c_start = 1'b0;
            wait (c_done === 1'b1); @(negedge clk);
            if (c_config_error !== 1'b0)
                $fatal(1, "%0s: config_error_o asserted on a legal config", who);
            if (c_idx != expected_count)
                $fatal(1, "%0s: done_o after %0d beats, expected %0d", who, c_idx, expected_count);
        end
    endtask

    task automatic run_dense(
        input [7:0] in_len, input [7:0] out_len,
        input [5:0] shift, input relu, input integer expected_count, input [24*8-1:0] who
    );
        begin
            d_idx <= 0;
            @(negedge clk);
            d_cfg_in_len = in_len; d_cfg_out_len = out_len;
            d_cfg_shift = shift; d_cfg_relu = relu;
            d_start = 1'b1; @(negedge clk); d_start = 1'b0;
            wait (d_done === 1'b1); @(negedge clk);
            if (d_config_error !== 1'b0)
                $fatal(1, "%0s: config_error_o asserted on a legal config", who);
            if (d_idx != expected_count)
                $fatal(1, "%0s: done_o after %0d beats, expected %0d", who, d_idx, expected_count);
        end
    endtask

    initial begin
        $readmemh("vectors/layer_shapes/c1_act.hex",  c1_act_vec);
        $readmemh("vectors/layer_shapes/c1_wgt.hex",  c1_wgt_vec);
        $readmemh("vectors/layer_shapes/c1_bias.hex", c1_bias_vec);
        $readmemh("vectors/layer_shapes/c1_expected.hex", c1_exp_vec);

        $readmemh("vectors/layer_shapes/c3_act.hex",  c3_act_vec);
        $readmemh("vectors/layer_shapes/c3_wgt.hex",  c3_wgt_vec);
        $readmemh("vectors/layer_shapes/c3_bias.hex", c3_bias_vec);
        $readmemh("vectors/layer_shapes/c3_expected.hex", c3_exp_vec);

        $readmemh("vectors/layer_shapes/c5_act.hex",  c5_act_vec);
        $readmemh("vectors/layer_shapes/c5_wgt.hex",  c5_wgt_vec);
        $readmemh("vectors/layer_shapes/c5_bias.hex", c5_bias_vec);
        $readmemh("vectors/layer_shapes/c5_expected.hex", c5_exp_vec);

        $readmemh("vectors/layer_shapes/f6_act.hex",  f6_act_vec);
        $readmemh("vectors/layer_shapes/f6_wgt.hex",  f6_wgt_vec);
        $readmemh("vectors/layer_shapes/f6_bias.hex", f6_bias_vec);
        $readmemh("vectors/layer_shapes/f6_expected.hex", f6_exp_vec);

        $readmemh("vectors/layer_shapes/cls_act.hex",  cls_act_vec);
        $readmemh("vectors/layer_shapes/cls_wgt.hex",  cls_wgt_vec);
        $readmemh("vectors/layer_shapes/cls_bias.hex", cls_bias_vec);
        $readmemh("vectors/layer_shapes/cls_expected.hex", cls_exp_vec);
        $readmemh("vectors/layer_shapes/cls_accumulator.hex", cls_acc_vec);

        c_start = 1'b0; d_start = 1'b0;
        c_act_we = 1'b0; c_wgt_we = 1'b0; c_bias_we = 1'b0; c_conn_we = 1'b0;
        d_act_we = 1'b0; d_wgt_we = 1'b0; d_bias_we = 1'b0;
        conv_phase = 0; dense_phase = 0;
        cls_best_acc = 0; cls_winner = 0;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // ---- C1: 1x32x32 -> 6x28x28 -------------------------------------
        conv_phase = 0;
        load_c1();
        run_conv(8'(`TV_LS_C1_IN_W), 8'(`TV_LS_C1_IN_H), 8'(`TV_LS_C1_IN_CH), 8'(`TV_LS_C1_OUT_CH),
                 6'd7, 1'b1, `TV_LS_C1_OUT_COUNT, "C1");
        $display("  C1 1x32x32 -> 6x28x28: %0d beats bit-exact against deploy_forward_int8", c_idx);
        $display("PASS tb_layer_shapes: conv2d_engine matches the golden model at the real C1 shape");

        // ---- C3: 6x14x14 -> 16x10x10, real sparse connectivity ----------
        conv_phase = 1;
        load_c3();
        run_conv(8'(`TV_LS_C3_IN_W), 8'(`TV_LS_C3_IN_H), 8'(`TV_LS_C3_IN_CH), 8'(`TV_LS_C3_OUT_CH),
                 6'd7, 1'b1, `TV_LS_C3_OUT_COUNT, "C3");
        $display("  C3 6x14x14 -> 16x10x10: %0d beats bit-exact, reconfigured from C1 with no reset, %0d/%0d real connections live",
                 c_idx, c3_conn_ones_seen, `TV_LS_C3_IN_CH * `TV_LS_C3_OUT_CH);
        $display("PASS tb_layer_shapes: conv2d_engine reconfigures cleanly to a new shape and the real sparse C3 connectivity checks out");

        // ---- C5: 16x5x5 -> 120x1x1, MAX_OUT_CH reached standalone -------
        conv_phase = 2;
        load_c5();
        run_conv(8'(`TV_LS_C5_IN_W), 8'(`TV_LS_C5_IN_H), 8'(`TV_LS_C5_IN_CH), 8'(`TV_LS_C5_OUT_CH),
                 6'd7, 1'b1, `TV_LS_C5_OUT_COUNT, "C5");
        $display("  C5 16x5x5 -> 120x1x1: %0d beats bit-exact, reconfigured from C3 with no reset", c_idx);
        $display("PASS tb_layer_shapes: conv2d_engine reaches MAX_OUT_CH=120 standalone and matches the golden model");

        // ---- F6: dense_engine 120 -> 84, MAX_OUT_LEN reached standalone -
        dense_phase = 0;
        load_f6();
        run_dense(8'(`TV_LS_F6_IN_LEN), 8'(`TV_LS_F6_OUT_LEN), 6'd7, 1'b1, `TV_LS_F6_OUT_COUNT, "F6");
        $display("  F6 120 -> 84: %0d beats bit-exact against deploy_forward_int8", d_idx);
        $display("PASS tb_layer_shapes: dense_engine reaches MAX_OUT_LEN=84 standalone and matches the golden model");

        // ---- classifier: dense_engine 84 -> 10, shift 0 / relu off ------
        dense_phase = 1;
        load_cls();
        run_dense(8'(`TV_LS_CLS_IN_LEN), 8'(`TV_LS_CLS_OUT_LEN), 6'd0, 1'b0, `TV_LS_CLS_OUT_COUNT, "classifier");
        if (cls_winner !== `TV_LS_EXPECTED_CLASS) begin
            $fatal(1, "classifier: argmax over out_acc_o picked class %0d, expected %0d",
                   cls_winner, `TV_LS_EXPECTED_CLASS);
        end
        $display("  classifier 84 -> 10: %0d beats bit-exact, reconfigured from F6 with no reset, argmax picks class %0d",
                 d_idx, cls_winner);
        $display("PASS tb_layer_shapes: dense_engine reconfigures cleanly to the classifier shape and out_acc_o argmax matches the golden prediction");

        $display("PASS tb_layer_shapes: C1/C3/C5/F6/classifier all bit-exact against the golden model on one reconfigured conv2d_engine and one reconfigured dense_engine");
        $finish;
    end

endmodule
