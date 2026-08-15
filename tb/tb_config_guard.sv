`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// Config-validation (reject-path) testbench.
//
// Why this exists
// ---------------
// Four modules implement runtime config validation and drive config_error_o.
// Every other testbench in this project only ever asserts that config_error_o
// stays *low* -- that is, they verify the accept path and leave the reject
// path entirely unexecuted. The rejection logic was written, elaborated and
// synthesized without a single stimulus ever reaching it, so any one of these
// comparisons could have been inverted, dropped, or wired to the wrong config
// port and every test in the suite would still have passed.
//
// This testbench drives all 22 reject conditions:
//
//   conv2d_engine       8   in_w<5, in_h<5, in_w>MAX, in_h>MAX,
//                           in_ch==0, in_ch>MAX, out_ch==0, out_ch>MAX
//   avg_pool2x2_stream  8   in_w==0, in_h==0, in_w odd, in_h odd,
//                           in_w>MAX, in_h>MAX, in_ch==0, in_ch>MAX
//   dense_engine        4   in_len==0, in_len>MAX, out_len==0, out_len>MAX
//   classifier_argmax   2   in_len==0, in_len>MAX
//
// classifier_argmax contributes 2 rather than 4 because it hardwires its
// inner dense_engine's cfg_out_len_i to NUM_CLASSES, so the two out_len
// reject conditions are structurally unreachable through its port list. That
// is a property of the wrapper, not an untested gap, and it is asserted below
// rather than assumed.
//
// Each stimulus is chosen to isolate exactly one condition: the pool ">MAX"
// cases use even values so they cannot also trip the odd-size check, and the
// pool "odd" cases stay within MAX so they cannot also trip the range check.
//
// Beyond "config_error_o rose", each reject is checked to be inert -- busy_o
// must stay low and done_o must never pulse, because an engine that flags an
// error *and* starts anyway is the more dangerous failure. Each DUT then
// takes a legal config to prove the error is cleared rather than latched: a
// design that bricks itself after one bad config would otherwise pass every
// check above.
// ---------------------------------------------------------------------------
module tb_config_guard;

    localparam integer CONV_MAX_IN_W   = 32;
    localparam integer CONV_MAX_IN_H   = 32;
    localparam integer CONV_MAX_IN_CH  = 16;
    localparam integer CONV_MAX_OUT_CH = 120;

    localparam integer POOL_MAX_IN_W  = 28;
    localparam integer POOL_MAX_IN_H  = 28;
    localparam integer POOL_MAX_IN_CH = 16;

    localparam integer DENSE_MAX_IN_LEN  = 120;
    localparam integer DENSE_MAX_OUT_LEN = 84;

    localparam integer CLS_MAX_IN_LEN = 84;
    localparam integer CLS_NUM_CLASSES = 10;

    logic clk = 1'b0;
    logic rst_n;
    always #5 clk = ~clk;

    integer rejects_checked = 0;
    integer accepts_checked = 0;

    // ---------------------------------------------------------------- conv
    logic       conv_start, conv_busy, conv_done, conv_cfg_err;
    logic [7:0] conv_in_w, conv_in_h, conv_in_ch, conv_out_ch;
    logic       conv_act_we, conv_wgt_we, conv_bias_we, conv_conn_we;
    logic [13:0] conv_act_addr;
    logic [15:0] conv_wgt_addr;
    logic [6:0]  conv_bias_addr;
    logic [10:0] conv_conn_addr;
    logic signed [7:0] conv_act_data, conv_wgt_data;
    logic signed [31:0] conv_bias_data;
    logic conv_conn_data;

    conv2d_engine #(
        .MAX_IN_W  (CONV_MAX_IN_W),
        .MAX_IN_H  (CONV_MAX_IN_H),
        .MAX_IN_CH (CONV_MAX_IN_CH),
        .MAX_OUT_CH(CONV_MAX_OUT_CH)
    ) u_conv (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(conv_start),
        .cfg_in_w_i(conv_in_w), .cfg_in_h_i(conv_in_h),
        .cfg_in_ch_i(conv_in_ch), .cfg_out_ch_i(conv_out_ch),
        .cfg_shift_i(6'd0), .cfg_relu_en_i(1'b0),
        .busy_o(conv_busy), .done_o(conv_done), .config_error_o(conv_cfg_err),
        .load_act_we_i(conv_act_we), .load_act_addr_i(conv_act_addr),
        .load_act_data_i(conv_act_data),
        .load_wgt_we_i(conv_wgt_we), .load_wgt_addr_i(conv_wgt_addr),
        .load_wgt_data_i(conv_wgt_data),
        .load_bias_we_i(conv_bias_we), .load_bias_addr_i(conv_bias_addr),
        .load_bias_data_i(conv_bias_data),
        .load_conn_we_i(conv_conn_we), .load_conn_addr_i(conv_conn_addr),
        .load_conn_data_i(conv_conn_data),
        .out_valid_o(), .out_ready_i(1'b1), .out_data_o(),
        .out_channel_o(), .out_y_o(), .out_x_o()
    );

    // ---------------------------------------------------------------- pool
    logic       pool_start, pool_busy, pool_done, pool_cfg_err;
    logic [7:0] pool_in_w, pool_in_h, pool_in_ch;
    logic       pool_act_we;
    logic [13:0] pool_act_addr;
    logic signed [7:0] pool_act_data;

    avg_pool2x2_stream #(
        .MAX_IN_W (POOL_MAX_IN_W),
        .MAX_IN_H (POOL_MAX_IN_H),
        .MAX_IN_CH(POOL_MAX_IN_CH)
    ) u_pool (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(pool_start),
        .cfg_in_w_i(pool_in_w), .cfg_in_h_i(pool_in_h),
        .cfg_in_ch_i(pool_in_ch),
        .busy_o(pool_busy), .done_o(pool_done), .config_error_o(pool_cfg_err),
        .load_act_we_i(pool_act_we), .load_act_addr_i(pool_act_addr),
        .load_act_data_i(pool_act_data),
        .out_valid_o(), .out_ready_i(1'b1), .out_data_o(),
        .out_channel_o(), .out_y_o(), .out_x_o()
    );

    // --------------------------------------------------------------- dense
    logic       den_start, den_busy, den_done, den_cfg_err;
    logic [7:0] den_in_len, den_out_len;
    logic       den_act_we, den_wgt_we, den_bias_we;
    logic [6:0]  den_act_addr;
    logic [13:0] den_wgt_addr;
    logic [6:0]  den_bias_addr;
    logic signed [7:0] den_act_data, den_wgt_data;
    logic signed [31:0] den_bias_data;

    dense_engine #(
        .MAX_IN_LEN (DENSE_MAX_IN_LEN),
        .MAX_OUT_LEN(DENSE_MAX_OUT_LEN)
    ) u_dense (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(den_start),
        .cfg_in_len_i(den_in_len), .cfg_out_len_i(den_out_len),
        .cfg_shift_i(6'd0), .cfg_relu_en_i(1'b0),
        .busy_o(den_busy), .done_o(den_done), .config_error_o(den_cfg_err),
        .load_act_we_i(den_act_we), .load_act_addr_i(den_act_addr),
        .load_act_data_i(den_act_data),
        .load_wgt_we_i(den_wgt_we), .load_wgt_addr_i(den_wgt_addr),
        .load_wgt_data_i(den_wgt_data),
        .load_bias_we_i(den_bias_we), .load_bias_addr_i(den_bias_addr),
        .load_bias_data_i(den_bias_data),
        .out_valid_o(), .out_ready_i(1'b1), .out_data_o(),
        .out_acc_o(), .out_index_o()
    );

    // ---------------------------------------------------------- classifier
    logic       cls_start, cls_busy, cls_done, cls_cfg_err;
    logic [7:0] cls_in_len;
    logic       cls_act_we, cls_wgt_we, cls_bias_we;
    logic [6:0]  cls_act_addr;
    logic [9:0]  cls_wgt_addr;
    logic [3:0]  cls_bias_addr;
    logic signed [7:0] cls_act_data, cls_wgt_data;
    logic signed [31:0] cls_bias_data;

    classifier_argmax #(
        .MAX_IN_LEN (CLS_MAX_IN_LEN),
        .NUM_CLASSES(CLS_NUM_CLASSES)
    ) u_cls (
        .clk_i(clk), .rst_ni(rst_n),
        .start_i(cls_start),
        .cfg_in_len_i(cls_in_len),
        .busy_o(cls_busy), .done_o(cls_done), .config_error_o(cls_cfg_err),
        .load_act_we_i(cls_act_we), .load_act_addr_i(cls_act_addr),
        .load_act_data_i(cls_act_data),
        .load_wgt_we_i(cls_wgt_we), .load_wgt_addr_i(cls_wgt_addr),
        .load_wgt_data_i(cls_wgt_data),
        .load_bias_we_i(cls_bias_we), .load_bias_addr_i(cls_bias_addr),
        .load_bias_data_i(cls_bias_data),
        .class_o(), .valid_o()
    );

    // A rejected start must be completely inert. done_o pulsing even once
    // after a rejected config would mean the engine ran anyway, so it is
    // latched here rather than sampled at one arbitrary instant.
    logic conv_done_seen, pool_done_seen, den_done_seen, cls_done_seen;
    always @(posedge clk) begin
        if (conv_done) conv_done_seen <= 1'b1;
        if (pool_done) pool_done_seen <= 1'b1;
        if (den_done)  den_done_seen  <= 1'b1;
        if (cls_done)  cls_done_seen  <= 1'b1;
    end

    // Cleared with non-blocking assignment to match the clocked block above.
    // Driving one variable with both blocking and non-blocking assignments
    // from two processes is simulator-dependent, and this testbench runs
    // under Icarus and ModelSim/Questa alike. The clear always happens on a
    // negedge and the set always on a posedge, so the two never contend.
    task automatic clear_done_flags;
        begin
            conv_done_seen <= 1'b0;
            pool_done_seen <= 1'b0;
            den_done_seen  <= 1'b0;
            cls_done_seen  <= 1'b0;
        end
    endtask

    // ------------------------------------------------------------ checkers
    // Each expect_reject task pulses start for exactly one cycle, then holds
    // the engine idle for a few cycles to prove nothing started behind the
    // error flag.
    task automatic check_reject(
        input reg [8*48-1:0] label,
        input                cfg_err,
        input                busy,
        input                done_seen
    );
        begin
            if (cfg_err !== 1'b1) begin
                $fatal(1, "REJECT MISSED [%0s]: config_error_o=%b, expected 1",
                       label, cfg_err);
            end
            if (busy !== 1'b0) begin
                $fatal(1, "REJECT UNSAFE [%0s]: busy_o=%b, engine started despite the error",
                       label, busy);
            end
            if (done_seen !== 1'b0) begin
                $fatal(1, "REJECT UNSAFE [%0s]: done_o pulsed after a rejected config",
                       label);
            end
            rejects_checked = rejects_checked + 1;
        end
    endtask

    task automatic conv_reject(
        input [7:0] w, input [7:0] h, input [7:0] ich, input [7:0] och,
        input reg [8*48-1:0] label
    );
        begin
            clear_done_flags();
            @(negedge clk);
            conv_in_w = w; conv_in_h = h; conv_in_ch = ich; conv_out_ch = och;
            conv_start = 1'b1;
            @(negedge clk);
            conv_start = 1'b0;
            repeat (3) @(negedge clk);
            check_reject(label, conv_cfg_err, conv_busy, conv_done_seen);
        end
    endtask

    task automatic pool_reject(
        input [7:0] w, input [7:0] h, input [7:0] ich,
        input reg [8*48-1:0] label
    );
        begin
            clear_done_flags();
            @(negedge clk);
            pool_in_w = w; pool_in_h = h; pool_in_ch = ich;
            pool_start = 1'b1;
            @(negedge clk);
            pool_start = 1'b0;
            repeat (3) @(negedge clk);
            check_reject(label, pool_cfg_err, pool_busy, pool_done_seen);
        end
    endtask

    task automatic dense_reject(
        input [7:0] ilen, input [7:0] olen,
        input reg [8*48-1:0] label
    );
        begin
            clear_done_flags();
            @(negedge clk);
            den_in_len = ilen; den_out_len = olen;
            den_start = 1'b1;
            @(negedge clk);
            den_start = 1'b0;
            repeat (3) @(negedge clk);
            check_reject(label, den_cfg_err, den_busy, den_done_seen);
        end
    endtask

    task automatic cls_reject(
        input [7:0] ilen,
        input reg [8*48-1:0] label
    );
        begin
            clear_done_flags();
            @(negedge clk);
            cls_in_len = ilen;
            cls_start = 1'b1;
            @(negedge clk);
            cls_start = 1'b0;
            repeat (3) @(negedge clk);
            check_reject(label, cls_cfg_err, cls_busy, cls_done_seen);
        end
    endtask

    // ------------------------------------------------------------- preload
    // Minimal legal payloads for the recovery runs. The reject cases never
    // read memory, but the accept cases do, and an X read out of an
    // uninitialised connection table would make control flow indeterminate
    // rather than merely making the data wrong.
    integer k;
    task automatic preload_all;
        begin
            @(negedge clk);
            for (k = 0; k < 25; k = k + 1) begin
                conv_act_we = 1'b1; conv_act_addr = k[13:0]; conv_act_data = 8'sd1;
                conv_wgt_we = 1'b1; conv_wgt_addr = k[15:0]; conv_wgt_data = 8'sd1;
                @(negedge clk);
            end
            conv_act_we = 1'b0; conv_wgt_we = 1'b0;

            conv_bias_we = 1'b1; conv_bias_addr = 7'd0; conv_bias_data = 32'sd0;
            conv_conn_we = 1'b1; conv_conn_addr = 11'd0; conv_conn_data = 1'b1;
            @(negedge clk);
            conv_bias_we = 1'b0; conv_conn_we = 1'b0;

            for (k = 0; k < 4; k = k + 1) begin
                pool_act_we = 1'b1; pool_act_addr = k[13:0]; pool_act_data = 8'sd4;
                @(negedge clk);
            end
            pool_act_we = 1'b0;

            den_act_we = 1'b1; den_act_addr = 7'd0; den_act_data = 8'sd1;
            den_wgt_we = 1'b1; den_wgt_addr = 14'd0; den_wgt_data = 8'sd1;
            den_bias_we = 1'b1; den_bias_addr = 7'd0; den_bias_data = 32'sd0;
            @(negedge clk);
            den_act_we = 1'b0; den_wgt_we = 1'b0; den_bias_we = 1'b0;

            for (k = 0; k < CLS_NUM_CLASSES; k = k + 1) begin
                cls_wgt_we = 1'b1; cls_wgt_addr = k[9:0]; cls_wgt_data = 8'sd1;
                cls_bias_we = 1'b1; cls_bias_addr = k[3:0]; cls_bias_data = 32'sd0;
                @(negedge clk);
            end
            cls_wgt_we = 1'b0; cls_bias_we = 1'b0;

            cls_act_we = 1'b1; cls_act_addr = 7'd0; cls_act_data = 8'sd1;
            @(negedge clk);
            cls_act_we = 1'b0;
        end
    endtask

    // ---------------------------------------------------------------- main
    initial begin
        rst_n = 1'b0;
        conv_start = 1'b0; pool_start = 1'b0;
        den_start  = 1'b0; cls_start  = 1'b0;
        conv_in_w = 8'd32; conv_in_h = 8'd32; conv_in_ch = 8'd1; conv_out_ch = 8'd6;
        pool_in_w = 8'd28; pool_in_h = 8'd28; pool_in_ch = 8'd6;
        den_in_len = 8'd120; den_out_len = 8'd84;
        cls_in_len = 8'd84;
        conv_act_we = 1'b0; conv_wgt_we = 1'b0;
        conv_bias_we = 1'b0; conv_conn_we = 1'b0;
        conv_act_addr = '0; conv_wgt_addr = '0;
        conv_bias_addr = '0; conv_conn_addr = '0;
        conv_act_data = '0; conv_wgt_data = '0;
        conv_bias_data = '0; conv_conn_data = 1'b0;
        pool_act_we = 1'b0; pool_act_addr = '0; pool_act_data = '0;
        den_act_we = 1'b0; den_wgt_we = 1'b0; den_bias_we = 1'b0;
        den_act_addr = '0; den_wgt_addr = '0; den_bias_addr = '0;
        den_act_data = '0; den_wgt_data = '0; den_bias_data = '0;
        cls_act_we = 1'b0; cls_wgt_we = 1'b0; cls_bias_we = 1'b0;
        cls_act_addr = '0; cls_wgt_addr = '0; cls_bias_addr = '0;
        cls_act_data = '0; cls_wgt_data = '0; cls_bias_data = '0;
        clear_done_flags();

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // ---- conv2d_engine: 8 conditions ------------------------------
        // K=5 with no padding, so anything under 5x5 has no valid window.
        conv_reject(8'd4,  8'd32, 8'd1, 8'd6,  "conv in_w < 5");
        conv_reject(8'd32, 8'd4,  8'd1, 8'd6,  "conv in_h < 5");
        conv_reject(8'(CONV_MAX_IN_W + 1), 8'd32, 8'd1, 8'd6, "conv in_w > MAX_IN_W");
        conv_reject(8'd32, 8'(CONV_MAX_IN_H + 1), 8'd1, 8'd6, "conv in_h > MAX_IN_H");
        conv_reject(8'd32, 8'd32, 8'd0, 8'd6,  "conv in_ch == 0");
        conv_reject(8'd32, 8'd32, 8'(CONV_MAX_IN_CH + 1), 8'd6, "conv in_ch > MAX_IN_CH");
        conv_reject(8'd32, 8'd32, 8'd1, 8'd0,  "conv out_ch == 0");
        conv_reject(8'd32, 8'd32, 8'd1, 8'(CONV_MAX_OUT_CH + 1), "conv out_ch > MAX_OUT_CH");

        // ---- avg_pool2x2_stream: 8 conditions --------------------------
        // The >MAX values are even and the odd values are within MAX, so each
        // stimulus trips exactly one comparison.
        pool_reject(8'd0,  8'd28, 8'd6, "pool in_w == 0");
        pool_reject(8'd28, 8'd0,  8'd6, "pool in_h == 0");
        pool_reject(8'd27, 8'd28, 8'd6, "pool in_w odd");
        pool_reject(8'd28, 8'd27, 8'd6, "pool in_h odd");
        pool_reject(8'(POOL_MAX_IN_W + 2), 8'd28, 8'd6, "pool in_w > MAX_IN_W");
        pool_reject(8'd28, 8'(POOL_MAX_IN_H + 2), 8'd6, "pool in_h > MAX_IN_H");
        pool_reject(8'd28, 8'd28, 8'd0, "pool in_ch == 0");
        pool_reject(8'd28, 8'd28, 8'(POOL_MAX_IN_CH + 1), "pool in_ch > MAX_IN_CH");

        // ---- dense_engine: 4 conditions --------------------------------
        dense_reject(8'd0,   8'd84, "dense in_len == 0");
        dense_reject(8'(DENSE_MAX_IN_LEN + 1), 8'd84, "dense in_len > MAX_IN_LEN");
        dense_reject(8'd120, 8'd0,  "dense out_len == 0");
        dense_reject(8'd120, 8'(DENSE_MAX_OUT_LEN + 1), "dense out_len > MAX_OUT_LEN");

        // ---- classifier_argmax: 2 reachable conditions ------------------
        cls_reject(8'd0, "classifier in_len == 0");
        cls_reject(8'(CLS_MAX_IN_LEN + 1), "classifier in_len > MAX_IN_LEN");

        if (rejects_checked != 22) begin
            $fatal(1, "expected 22 reject conditions, exercised %0d", rejects_checked);
        end
        $display("PASS tb_config_guard: all %0d reject conditions flagged and inert",
                 rejects_checked);

        // ---- recovery: a legal config after a rejected one --------------
        // config_error_o is cleared at the top of the accept branch, so an
        // engine that latched the error permanently would fail here and
        // nowhere else. Minimal legal shapes keep each run a few cycles.
        preload_all();
        clear_done_flags();

        @(negedge clk);
        conv_in_w = 8'd5; conv_in_h = 8'd5; conv_in_ch = 8'd1; conv_out_ch = 8'd1;
        conv_start = 1'b1;
        @(negedge clk);
        conv_start = 1'b0;
        if (conv_cfg_err !== 1'b0) begin
            $fatal(1, "conv2d_engine latched config_error_o across a legal restart");
        end
        if (conv_busy !== 1'b1) begin
            $fatal(1, "conv2d_engine refused a legal config after a rejected one");
        end
        wait (conv_done === 1'b1);
        accepts_checked = accepts_checked + 1;

        @(negedge clk);
        pool_in_w = 8'd2; pool_in_h = 8'd2; pool_in_ch = 8'd1;
        pool_start = 1'b1;
        @(negedge clk);
        pool_start = 1'b0;
        if (pool_cfg_err !== 1'b0) begin
            $fatal(1, "avg_pool2x2_stream latched config_error_o across a legal restart");
        end
        if (pool_busy !== 1'b1) begin
            $fatal(1, "avg_pool2x2_stream refused a legal config after a rejected one");
        end
        wait (pool_done === 1'b1);
        accepts_checked = accepts_checked + 1;

        @(negedge clk);
        den_in_len = 8'd1; den_out_len = 8'd1;
        den_start = 1'b1;
        @(negedge clk);
        den_start = 1'b0;
        if (den_cfg_err !== 1'b0) begin
            $fatal(1, "dense_engine latched config_error_o across a legal restart");
        end
        if (den_busy !== 1'b1) begin
            $fatal(1, "dense_engine refused a legal config after a rejected one");
        end
        wait (den_done === 1'b1);
        accepts_checked = accepts_checked + 1;

        @(negedge clk);
        cls_in_len = 8'd1;
        cls_start = 1'b1;
        @(negedge clk);
        cls_start = 1'b0;
        if (cls_cfg_err !== 1'b0) begin
            $fatal(1, "classifier_argmax latched config_error_o across a legal restart");
        end
        if (cls_busy !== 1'b1) begin
            $fatal(1, "classifier_argmax refused a legal config after a rejected one");
        end
        wait (cls_done === 1'b1);
        accepts_checked = accepts_checked + 1;

        if (accepts_checked != 4) begin
            $fatal(1, "expected 4 recovery runs, completed %0d", accepts_checked);
        end
        $display("PASS tb_config_guard: all %0d engines accepted a legal config after rejection",
                 accepts_checked);

        // classifier_argmax ties its inner dense_engine's cfg_out_len_i to
        // NUM_CLASSES, which is what makes the two out_len reject conditions
        // unreachable from its port list. Asserted rather than assumed, so
        // that rewiring the wrapper reopens this file rather than silently
        // shrinking its coverage.
        if (u_cls.u_dense_engine.cfg_out_len_q !== 8'(CLS_NUM_CLASSES)) begin
            $fatal(1,
                "classifier_argmax inner cfg_out_len is %0d, expected NUM_CLASSES=%0d: the two out_len reject conditions may now be reachable and untested",
                u_cls.u_dense_engine.cfg_out_len_q, CLS_NUM_CLASSES);
        end
        $display("PASS tb_config_guard: classifier out_len is hardwired to %0d, so its out_len rejects stay unreachable",
                 CLS_NUM_CLASSES);

        $finish;
    end

    initial begin
        repeat (20000) @(posedge clk);
        $fatal(1, "tb_config_guard: simulation timeout");
    end
endmodule
