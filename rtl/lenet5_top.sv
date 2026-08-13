`timescale 1ns/1ps

// Top-level LeNet-5 sequencer: C1 -> S2 -> C3 -> S4 -> C5 -> F6 -> classifier.
//
// Resource-shared: one conv2d_engine instance runs C1, C3, and C5 in turn;
// one avg_pool2x2_stream instance runs S2 and S4; one dense_engine runs F6;
// one classifier_argmax (with its own internal dense_engine) produces the
// final class index. Every stage's activation input is bridged straight
// from the previous stage's output stream while the previous stage is
// running (both engines' load ports accept writes independent of busy_o,
// and output data/valid are held stable until accepted, so no separate
// buffering stage or shared memory is needed). Each stage's weights/bias
// are staged in a persistent ROM owned by this module and copied into the
// active engine's scratch memory immediately before that engine starts.
//
// C1/C5 use full connectivity (a constant-1 fill, no ROM needed). C3 uses
// the sparse LeCun-98 table: lenet5_c3_connectivity is instantiated live
// here and walked by a counter to drive the fill, rather than storing the
// 96-bit table in a ROM.
//
// All activations/weights/biases are preloaded through one demuxed ROM
// bus (cfg_rom_sel_i + load_we_i/addr_i/data_i) before start_i, exactly
// like conv2d_engine's own load_*_we_i ports -- see docs/INTERFACES.md.
//
// This module is a behavioral/verification integration, not a synthesis
// target: like conv2d_engine, its ROM arrays stand in for future SRAM
// macros (see docs/SEMICUSTOM_FLOW.md).
module lenet5_top #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32,
    parameter integer SHIFT_C1   = 7,
    parameter integer SHIFT_C3   = 7,
    parameter integer SHIFT_C5   = 7,
    parameter integer SHIFT_F6   = 7,
    localparam integer C5_WGT_COUNT   = 120 * 16 * 25,
    localparam integer ROM_ADDR_WIDTH = $clog2(C5_WGT_COUNT)
) (
    input  logic clk_i,
    input  logic rst_ni,

    input  logic       start_i,
    output logic       busy_o,
    output logic       done_o,
    output logic [3:0] class_o,

    // Demuxed preload bus for every persistent per-layer ROM.
    input  logic                         load_we_i,
    input  logic [3:0]                   cfg_rom_sel_i,
    input  logic [ROM_ADDR_WIDTH-1:0]    load_addr_i,
    input  logic signed [ACC_WIDTH-1:0]  load_data_i
);
    // ------------------------------------------------------------------
    // Per-layer tensor counts (canonical LeNet-5 dimensions).
    // ------------------------------------------------------------------
    localparam integer IMG_COUNT      = 1 * 32 * 32;
    localparam integer C1_WGT_COUNT   = 6 * 1 * 25;
    localparam integer C1_BIAS_COUNT  = 6;
    localparam integer C1_CONN_COUNT  = 6 * 1;
    localparam integer C3_WGT_COUNT   = 16 * 6 * 25;
    localparam integer C3_BIAS_COUNT  = 16;
    localparam integer C3_CONN_COUNT  = 16 * 6;
    localparam integer C5_BIAS_COUNT  = 120;
    localparam integer C5_CONN_COUNT  = 120 * 16;
    localparam integer F6_WGT_COUNT   = 84 * 120;
    localparam integer F6_BIAS_COUNT  = 84;
    localparam integer CLS_WGT_COUNT  = 10 * 84;
    localparam integer CLS_BIAS_COUNT = 10;

    localparam integer MAX_LOAD_C1  = (IMG_COUNT > C1_WGT_COUNT) ? IMG_COUNT : C1_WGT_COUNT;
    localparam integer MAX_LOAD_C3  = C3_WGT_COUNT;
    localparam integer MAX_LOAD_C5  = C5_WGT_COUNT;
    localparam integer MAX_LOAD_F6  = F6_WGT_COUNT;
    localparam integer MAX_LOAD_CLS = CLS_WGT_COUNT;

    localparam integer ROM_SEL_IMAGE    = 0;
    localparam integer ROM_SEL_C1_WGT   = 1;
    localparam integer ROM_SEL_C1_BIAS  = 2;
    localparam integer ROM_SEL_C3_WGT   = 3;
    localparam integer ROM_SEL_C3_BIAS  = 4;
    localparam integer ROM_SEL_C5_WGT   = 5;
    localparam integer ROM_SEL_C5_BIAS  = 6;
    localparam integer ROM_SEL_F6_WGT   = 7;
    localparam integer ROM_SEL_F6_BIAS  = 8;
    localparam integer ROM_SEL_CLS_WGT  = 9;
    localparam integer ROM_SEL_CLS_BIAS = 10;

    // ------------------------------------------------------------------
    // Persistent per-layer ROMs, loaded once via the external preload bus.
    // ------------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] image_rom    [0:IMG_COUNT-1];
    logic signed [DATA_WIDTH-1:0] c1_wgt_rom   [0:C1_WGT_COUNT-1];
    logic signed [ACC_WIDTH-1:0]  c1_bias_rom  [0:C1_BIAS_COUNT-1];
    logic signed [DATA_WIDTH-1:0] c3_wgt_rom   [0:C3_WGT_COUNT-1];
    logic signed [ACC_WIDTH-1:0]  c3_bias_rom  [0:C3_BIAS_COUNT-1];
    logic signed [DATA_WIDTH-1:0] c5_wgt_rom   [0:C5_WGT_COUNT-1];
    logic signed [ACC_WIDTH-1:0]  c5_bias_rom  [0:C5_BIAS_COUNT-1];
    logic signed [DATA_WIDTH-1:0] f6_wgt_rom   [0:F6_WGT_COUNT-1];
    logic signed [ACC_WIDTH-1:0]  f6_bias_rom  [0:F6_BIAS_COUNT-1];
    logic signed [DATA_WIDTH-1:0] cls_wgt_rom  [0:CLS_WGT_COUNT-1];
    logic signed [ACC_WIDTH-1:0]  cls_bias_rom [0:CLS_BIAS_COUNT-1];

    always_ff @(posedge clk_i) begin
        if (load_we_i) begin
            case (cfg_rom_sel_i)
                ROM_SEL_IMAGE:    image_rom[load_addr_i]    <= load_data_i[DATA_WIDTH-1:0];
                ROM_SEL_C1_WGT:   c1_wgt_rom[load_addr_i]   <= load_data_i[DATA_WIDTH-1:0];
                ROM_SEL_C1_BIAS:  c1_bias_rom[load_addr_i]  <= load_data_i;
                ROM_SEL_C3_WGT:   c3_wgt_rom[load_addr_i]   <= load_data_i[DATA_WIDTH-1:0];
                ROM_SEL_C3_BIAS:  c3_bias_rom[load_addr_i]  <= load_data_i;
                ROM_SEL_C5_WGT:   c5_wgt_rom[load_addr_i]   <= load_data_i[DATA_WIDTH-1:0];
                ROM_SEL_C5_BIAS:  c5_bias_rom[load_addr_i]  <= load_data_i;
                ROM_SEL_F6_WGT:   f6_wgt_rom[load_addr_i]   <= load_data_i[DATA_WIDTH-1:0];
                ROM_SEL_F6_BIAS:  f6_bias_rom[load_addr_i]  <= load_data_i;
                ROM_SEL_CLS_WGT:  cls_wgt_rom[load_addr_i]  <= load_data_i[DATA_WIDTH-1:0];
                ROM_SEL_CLS_BIAS: cls_bias_rom[load_addr_i] <= load_data_i;
                default: ; // no-op
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Shared engine instances.
    // ------------------------------------------------------------------
    localparam integer CONV_ACT_ADDR_W  = $clog2(32*32*16);
    localparam integer CONV_WGT_ADDR_W  = $clog2(120*16*25);
    localparam integer CONV_BIAS_ADDR_W = $clog2(120);
    localparam integer CONV_CONN_ADDR_W = $clog2(120*16);
    localparam integer POOL_ACT_ADDR_W  = $clog2(28*28*16);
    localparam integer F6_ACT_ADDR_W    = $clog2(120);
    localparam integer F6_WGT_ADDR_W    = $clog2(84*120);
    localparam integer F6_BIAS_ADDR_W   = $clog2(84);
    localparam integer CLS_ACT_ADDR_W   = $clog2(84);
    localparam integer CLS_WGT_ADDR_W   = $clog2(840);
    localparam integer CLS_BIAS_ADDR_W  = $clog2(10);

    logic        conv_start;
    logic [7:0]  conv_cfg_in_w, conv_cfg_in_h, conv_cfg_in_ch, conv_cfg_out_ch;
    logic [5:0]  conv_cfg_shift;
    logic        conv_cfg_relu_en;
    logic        conv_busy, conv_done, conv_config_error;
    logic        conv_load_act_we;
    logic [CONV_ACT_ADDR_W-1:0]  conv_load_act_addr;
    logic signed [DATA_WIDTH-1:0] conv_load_act_data;
    logic        conv_load_wgt_we;
    logic [CONV_WGT_ADDR_W-1:0]  conv_load_wgt_addr;
    logic signed [DATA_WIDTH-1:0] conv_load_wgt_data;
    logic        conv_load_bias_we;
    logic [CONV_BIAS_ADDR_W-1:0] conv_load_bias_addr;
    logic signed [ACC_WIDTH-1:0]  conv_load_bias_data;
    logic        conv_load_conn_we;
    logic [CONV_CONN_ADDR_W-1:0] conv_load_conn_addr;
    logic        conv_load_conn_data;
    logic        conv_out_valid, conv_out_ready;
    logic signed [DATA_WIDTH-1:0] conv_out_data;
    logic [7:0]  conv_out_channel, conv_out_y, conv_out_x;

    conv2d_engine #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_conv2d_engine (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .start_i       (conv_start),
        .cfg_in_w_i    (conv_cfg_in_w),
        .cfg_in_h_i    (conv_cfg_in_h),
        .cfg_in_ch_i   (conv_cfg_in_ch),
        .cfg_out_ch_i  (conv_cfg_out_ch),
        .cfg_shift_i   (conv_cfg_shift),
        .cfg_relu_en_i (conv_cfg_relu_en),
        .busy_o        (conv_busy),
        .done_o        (conv_done),
        .config_error_o(conv_config_error),
        .load_act_we_i  (conv_load_act_we),
        .load_act_addr_i(conv_load_act_addr),
        .load_act_data_i(conv_load_act_data),
        .load_wgt_we_i  (conv_load_wgt_we),
        .load_wgt_addr_i(conv_load_wgt_addr),
        .load_wgt_data_i(conv_load_wgt_data),
        .load_bias_we_i  (conv_load_bias_we),
        .load_bias_addr_i(conv_load_bias_addr),
        .load_bias_data_i(conv_load_bias_data),
        .load_conn_we_i  (conv_load_conn_we),
        .load_conn_addr_i(conv_load_conn_addr),
        .load_conn_data_i(conv_load_conn_data),
        .out_valid_o  (conv_out_valid),
        .out_ready_i  (conv_out_ready),
        .out_data_o   (conv_out_data),
        .out_channel_o(conv_out_channel),
        .out_y_o      (conv_out_y),
        .out_x_o      (conv_out_x)
    );

    logic       pool_start;
    logic [7:0] pool_cfg_in_w, pool_cfg_in_h, pool_cfg_in_ch;
    logic       pool_busy, pool_done, pool_config_error;
    logic       pool_load_act_we;
    logic [POOL_ACT_ADDR_W-1:0] pool_load_act_addr;
    logic signed [DATA_WIDTH-1:0] pool_load_act_data;
    logic       pool_out_valid, pool_out_ready;
    logic signed [DATA_WIDTH-1:0] pool_out_data;
    logic [7:0] pool_out_channel, pool_out_y, pool_out_x;

    avg_pool2x2_stream #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_avg_pool2x2_stream (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .start_i       (pool_start),
        .cfg_in_w_i    (pool_cfg_in_w),
        .cfg_in_h_i    (pool_cfg_in_h),
        .cfg_in_ch_i   (pool_cfg_in_ch),
        .busy_o        (pool_busy),
        .done_o        (pool_done),
        .config_error_o(pool_config_error),
        .load_act_we_i  (pool_load_act_we),
        .load_act_addr_i(pool_load_act_addr),
        .load_act_data_i(pool_load_act_data),
        .out_valid_o  (pool_out_valid),
        .out_ready_i  (pool_out_ready),
        .out_data_o   (pool_out_data),
        .out_channel_o(pool_out_channel),
        .out_y_o      (pool_out_y),
        .out_x_o      (pool_out_x)
    );

    logic       f6_start;
    logic [7:0] f6_cfg_in_len, f6_cfg_out_len;
    logic [5:0] f6_cfg_shift;
    logic       f6_cfg_relu_en;
    logic       f6_busy, f6_done, f6_config_error;
    logic       f6_load_act_we;
    logic [F6_ACT_ADDR_W-1:0] f6_load_act_addr;
    logic signed [DATA_WIDTH-1:0] f6_load_act_data;
    logic       f6_load_wgt_we;
    logic [F6_WGT_ADDR_W-1:0] f6_load_wgt_addr;
    logic signed [DATA_WIDTH-1:0] f6_load_wgt_data;
    logic       f6_load_bias_we;
    logic [F6_BIAS_ADDR_W-1:0] f6_load_bias_addr;
    logic signed [ACC_WIDTH-1:0] f6_load_bias_data;
    logic       f6_out_valid, f6_out_ready;
    logic signed [DATA_WIDTH-1:0] f6_out_data;
    logic signed [ACC_WIDTH-1:0]  f6_out_acc;
    logic [7:0] f6_out_index;

    dense_engine #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .MAX_IN_LEN (120),
        .MAX_OUT_LEN(84)
    ) u_dense_engine_f6 (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .start_i       (f6_start),
        .cfg_in_len_i  (f6_cfg_in_len),
        .cfg_out_len_i (f6_cfg_out_len),
        .cfg_shift_i   (f6_cfg_shift),
        .cfg_relu_en_i (f6_cfg_relu_en),
        .busy_o        (f6_busy),
        .done_o        (f6_done),
        .config_error_o(f6_config_error),
        .load_act_we_i  (f6_load_act_we),
        .load_act_addr_i(f6_load_act_addr),
        .load_act_data_i(f6_load_act_data),
        .load_wgt_we_i  (f6_load_wgt_we),
        .load_wgt_addr_i(f6_load_wgt_addr),
        .load_wgt_data_i(f6_load_wgt_data),
        .load_bias_we_i  (f6_load_bias_we),
        .load_bias_addr_i(f6_load_bias_addr),
        .load_bias_data_i(f6_load_bias_data),
        .out_valid_o(f6_out_valid),
        .out_ready_i(f6_out_ready),
        .out_data_o (f6_out_data),
        .out_acc_o  (f6_out_acc),
        .out_index_o(f6_out_index)
    );

    logic       cls_start;
    logic [7:0] cls_cfg_in_len;
    logic       cls_busy, cls_done, cls_config_error;
    logic       cls_load_act_we;
    logic [CLS_ACT_ADDR_W-1:0] cls_load_act_addr;
    logic signed [DATA_WIDTH-1:0] cls_load_act_data;
    logic       cls_load_wgt_we;
    logic [CLS_WGT_ADDR_W-1:0] cls_load_wgt_addr;
    logic signed [DATA_WIDTH-1:0] cls_load_wgt_data;
    logic       cls_load_bias_we;
    logic [CLS_BIAS_ADDR_W-1:0] cls_load_bias_addr;
    logic signed [ACC_WIDTH-1:0] cls_load_bias_data;
    logic [3:0] cls_class;
    logic       cls_valid;

    classifier_argmax #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .MAX_IN_LEN (84),
        .NUM_CLASSES(10)
    ) u_classifier_argmax (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .start_i       (cls_start),
        .cfg_in_len_i  (cls_cfg_in_len),
        .busy_o        (cls_busy),
        .done_o        (cls_done),
        .config_error_o(cls_config_error),
        .load_act_we_i  (cls_load_act_we),
        .load_act_addr_i(cls_load_act_addr),
        .load_act_data_i(cls_load_act_data),
        .load_wgt_we_i  (cls_load_wgt_we),
        .load_wgt_addr_i(cls_load_wgt_addr),
        .load_wgt_data_i(cls_load_wgt_data),
        .load_bias_we_i  (cls_load_bias_we),
        .load_bias_addr_i(cls_load_bias_addr),
        .load_bias_data_i(cls_load_bias_data),
        .class_o(cls_class),
        .valid_o(cls_valid)
    );

    // ------------------------------------------------------------------
    // Top control FSM state (declared early: referenced by the live C3
    // connectivity generator below before the FSM itself is defined).
    // A plain counter register with localparam stage constants (not
    // typedef enum), since the stage sequence is a flat list rather than
    // conv2d_engine's nested loop-counter shape.
    // ------------------------------------------------------------------
    localparam integer S_IDLE     = 0;
    localparam integer S_LOAD_C1  = 1;
    localparam integer S_KICK_C1  = 2;
    localparam integer S_RUN_C1   = 3;
    localparam integer S_KICK_S2  = 4;
    localparam integer S_RUN_S2   = 5;
    localparam integer S_LOAD_C3  = 6;
    localparam integer S_KICK_C3  = 7;
    localparam integer S_RUN_C3   = 8;
    localparam integer S_KICK_S4  = 9;
    localparam integer S_RUN_S4   = 10;
    localparam integer S_LOAD_C5  = 11;
    localparam integer S_KICK_C5  = 12;
    localparam integer S_RUN_C5   = 13;
    localparam integer S_LOAD_F6  = 14;
    localparam integer S_KICK_F6  = 15;
    localparam integer S_RUN_F6   = 16;
    localparam integer S_LOAD_CLS = 17;
    localparam integer S_KICK_CLS = 18;
    localparam integer S_RUN_CLS  = 19;

    logic [4:0]  top_stage_q;
    logic [31:0] load_idx_q;

    // ------------------------------------------------------------------
    // Live C3 sparse-connectivity generation (walks the existing
    // lenet5_c3_connectivity table instead of storing it in a ROM).
    // ------------------------------------------------------------------
    integer c3_output_map_i, c3_input_map_i;
    logic   c3_conn_bit;

    always_comb begin
        c3_output_map_i = load_idx_q / 6;
        c3_input_map_i  = load_idx_q % 6;
    end

    lenet5_c3_connectivity u_lenet5_c3_connectivity (
        .output_map_i(c3_output_map_i[3:0]),
        .input_map_i (c3_input_map_i[2:0]),
        .connected_o (c3_conn_bit)
    );

    // ------------------------------------------------------------------
    // Inter-stage bridge addressing (CHW, matching each downstream
    // engine's own addressing convention).
    // ------------------------------------------------------------------
    integer c1_to_s2_addr, s2_to_c3_addr, c3_to_s4_addr, s4_to_c5_addr;
    always_comb begin
        c1_to_s2_addr = (integer'(conv_out_channel) * 28 + integer'(conv_out_y)) * 28
                        + integer'(conv_out_x);
        s2_to_c3_addr = (integer'(pool_out_channel) * 14 + integer'(pool_out_y)) * 14
                        + integer'(pool_out_x);
        c3_to_s4_addr = (integer'(conv_out_channel) * 10 + integer'(conv_out_y)) * 10
                        + integer'(conv_out_x);
        s4_to_c5_addr = (integer'(pool_out_channel) * 5 + integer'(pool_out_y)) * 5
                        + integer'(pool_out_x);
    end

    // Runtime layer configuration + start pulses, valid only in their
    // respective KICK_* cycle.
    always_comb begin
        conv_cfg_in_w    = 8'd0;
        conv_cfg_in_h    = 8'd0;
        conv_cfg_in_ch   = 8'd0;
        conv_cfg_out_ch  = 8'd0;
        conv_cfg_shift   = 6'd0;
        conv_cfg_relu_en = 1'b0;
        conv_start       = 1'b0;

        pool_cfg_in_w  = 8'd0;
        pool_cfg_in_h  = 8'd0;
        pool_cfg_in_ch = 8'd0;
        pool_start     = 1'b0;

        f6_cfg_in_len  = 8'd0;
        f6_cfg_out_len = 8'd0;
        f6_cfg_shift   = 6'd0;
        f6_cfg_relu_en = 1'b0;
        f6_start       = 1'b0;

        cls_cfg_in_len = 8'd0;
        cls_start      = 1'b0;

        case (top_stage_q)
            S_KICK_C1: begin
                conv_cfg_in_w    = 8'd32;
                conv_cfg_in_h    = 8'd32;
                conv_cfg_in_ch   = 8'd1;
                conv_cfg_out_ch  = 8'd6;
                conv_cfg_shift   = 6'(SHIFT_C1);
                conv_cfg_relu_en = 1'b1;
                conv_start       = 1'b1;
            end
            S_KICK_S2: begin
                pool_cfg_in_w  = 8'd28;
                pool_cfg_in_h  = 8'd28;
                pool_cfg_in_ch = 8'd6;
                pool_start     = 1'b1;
            end
            S_KICK_C3: begin
                conv_cfg_in_w    = 8'd14;
                conv_cfg_in_h    = 8'd14;
                conv_cfg_in_ch   = 8'd6;
                conv_cfg_out_ch  = 8'd16;
                conv_cfg_shift   = 6'(SHIFT_C3);
                conv_cfg_relu_en = 1'b1;
                conv_start       = 1'b1;
            end
            S_KICK_S4: begin
                pool_cfg_in_w  = 8'd10;
                pool_cfg_in_h  = 8'd10;
                pool_cfg_in_ch = 8'd16;
                pool_start     = 1'b1;
            end
            S_KICK_C5: begin
                conv_cfg_in_w    = 8'd5;
                conv_cfg_in_h    = 8'd5;
                conv_cfg_in_ch   = 8'd16;
                conv_cfg_out_ch  = 8'd120;
                conv_cfg_shift   = 6'(SHIFT_C5);
                conv_cfg_relu_en = 1'b1;
                conv_start       = 1'b1;
            end
            S_KICK_F6: begin
                f6_cfg_in_len  = 8'd120;
                f6_cfg_out_len = 8'd84;
                f6_cfg_shift   = 6'(SHIFT_F6);
                f6_cfg_relu_en = 1'b1;
                f6_start       = 1'b1;
            end
            S_KICK_CLS: begin
                cls_cfg_in_len = 8'd84;
                cls_start      = 1'b1;
            end
            default: ;
        endcase
    end

    // Load-stage ROM->engine copy ports and inter-stage bridges. Every
    // load-port count is checked independently against the same shared
    // load_idx_q counter, so activation/weight/bias/connectivity copies
    // for a stage run in lockstep across the widest of the four counts
    // rather than sequential phases.
    always_comb begin
        conv_load_act_we    = 1'b0;
        conv_load_act_addr  = '0;
        conv_load_act_data  = '0;
        conv_load_wgt_we    = 1'b0;
        conv_load_wgt_addr  = '0;
        conv_load_wgt_data  = '0;
        conv_load_bias_we   = 1'b0;
        conv_load_bias_addr = '0;
        conv_load_bias_data = '0;
        conv_load_conn_we   = 1'b0;
        conv_load_conn_addr = '0;
        conv_load_conn_data = 1'b0;
        conv_out_ready      = 1'b0;

        pool_load_act_we   = 1'b0;
        pool_load_act_addr = '0;
        pool_load_act_data = '0;
        pool_out_ready     = 1'b0;

        f6_load_act_we    = 1'b0;
        f6_load_act_addr  = '0;
        f6_load_act_data  = '0;
        f6_load_wgt_we    = 1'b0;
        f6_load_wgt_addr  = '0;
        f6_load_wgt_data  = '0;
        f6_load_bias_we   = 1'b0;
        f6_load_bias_addr = '0;
        f6_load_bias_data = '0;
        f6_out_ready       = 1'b0;

        cls_load_act_we    = 1'b0;
        cls_load_act_addr  = '0;
        cls_load_act_data  = '0;
        cls_load_wgt_we    = 1'b0;
        cls_load_wgt_addr  = '0;
        cls_load_wgt_data  = '0;
        cls_load_bias_we   = 1'b0;
        cls_load_bias_addr = '0;
        cls_load_bias_data = '0;

        case (top_stage_q)
            S_LOAD_C1: begin
                if (load_idx_q < IMG_COUNT) begin
                    conv_load_act_we   = 1'b1;
                    conv_load_act_addr = load_idx_q[CONV_ACT_ADDR_W-1:0];
                    conv_load_act_data = image_rom[load_idx_q];
                end
                if (load_idx_q < C1_WGT_COUNT) begin
                    conv_load_wgt_we   = 1'b1;
                    conv_load_wgt_addr = load_idx_q[CONV_WGT_ADDR_W-1:0];
                    conv_load_wgt_data = c1_wgt_rom[load_idx_q];
                end
                if (load_idx_q < C1_BIAS_COUNT) begin
                    conv_load_bias_we   = 1'b1;
                    conv_load_bias_addr = load_idx_q[CONV_BIAS_ADDR_W-1:0];
                    conv_load_bias_data = c1_bias_rom[load_idx_q];
                end
                if (load_idx_q < C1_CONN_COUNT) begin
                    conv_load_conn_we   = 1'b1;
                    conv_load_conn_addr = load_idx_q[CONV_CONN_ADDR_W-1:0];
                    conv_load_conn_data = 1'b1;
                end
            end

            S_RUN_C1: begin
                conv_out_ready      = 1'b1;
                pool_load_act_we    = conv_out_valid;
                pool_load_act_addr  = c1_to_s2_addr[POOL_ACT_ADDR_W-1:0];
                pool_load_act_data  = conv_out_data;
            end

            S_RUN_S2: begin
                pool_out_ready      = 1'b1;
                conv_load_act_we    = pool_out_valid;
                conv_load_act_addr  = s2_to_c3_addr[CONV_ACT_ADDR_W-1:0];
                conv_load_act_data  = pool_out_data;
            end

            S_LOAD_C3: begin
                if (load_idx_q < C3_WGT_COUNT) begin
                    conv_load_wgt_we   = 1'b1;
                    conv_load_wgt_addr = load_idx_q[CONV_WGT_ADDR_W-1:0];
                    conv_load_wgt_data = c3_wgt_rom[load_idx_q];
                end
                if (load_idx_q < C3_BIAS_COUNT) begin
                    conv_load_bias_we   = 1'b1;
                    conv_load_bias_addr = load_idx_q[CONV_BIAS_ADDR_W-1:0];
                    conv_load_bias_data = c3_bias_rom[load_idx_q];
                end
                if (load_idx_q < C3_CONN_COUNT) begin
                    conv_load_conn_we   = 1'b1;
                    conv_load_conn_addr = load_idx_q[CONV_CONN_ADDR_W-1:0];
                    conv_load_conn_data = c3_conn_bit;
                end
            end

            S_RUN_C3: begin
                conv_out_ready      = 1'b1;
                pool_load_act_we    = conv_out_valid;
                pool_load_act_addr  = c3_to_s4_addr[POOL_ACT_ADDR_W-1:0];
                pool_load_act_data  = conv_out_data;
            end

            S_RUN_S4: begin
                pool_out_ready      = 1'b1;
                conv_load_act_we    = pool_out_valid;
                conv_load_act_addr  = s4_to_c5_addr[CONV_ACT_ADDR_W-1:0];
                conv_load_act_data  = pool_out_data;
            end

            S_LOAD_C5: begin
                if (load_idx_q < C5_WGT_COUNT) begin
                    conv_load_wgt_we   = 1'b1;
                    conv_load_wgt_addr = load_idx_q[CONV_WGT_ADDR_W-1:0];
                    conv_load_wgt_data = c5_wgt_rom[load_idx_q];
                end
                if (load_idx_q < C5_BIAS_COUNT) begin
                    conv_load_bias_we   = 1'b1;
                    conv_load_bias_addr = load_idx_q[CONV_BIAS_ADDR_W-1:0];
                    conv_load_bias_data = c5_bias_rom[load_idx_q];
                end
                if (load_idx_q < C5_CONN_COUNT) begin
                    conv_load_conn_we   = 1'b1;
                    conv_load_conn_addr = load_idx_q[CONV_CONN_ADDR_W-1:0];
                    conv_load_conn_data = 1'b1;
                end
            end

            S_RUN_C5: begin
                conv_out_ready   = 1'b1;
                f6_load_act_we   = conv_out_valid;
                f6_load_act_addr = conv_out_channel[F6_ACT_ADDR_W-1:0];
                f6_load_act_data = conv_out_data;
            end

            S_LOAD_F6: begin
                if (load_idx_q < F6_WGT_COUNT) begin
                    f6_load_wgt_we   = 1'b1;
                    f6_load_wgt_addr = load_idx_q[F6_WGT_ADDR_W-1:0];
                    f6_load_wgt_data = f6_wgt_rom[load_idx_q];
                end
                if (load_idx_q < F6_BIAS_COUNT) begin
                    f6_load_bias_we   = 1'b1;
                    f6_load_bias_addr = load_idx_q[F6_BIAS_ADDR_W-1:0];
                    f6_load_bias_data = f6_bias_rom[load_idx_q];
                end
            end

            S_RUN_F6: begin
                f6_out_ready      = 1'b1;
                cls_load_act_we   = f6_out_valid;
                cls_load_act_addr = f6_out_index[CLS_ACT_ADDR_W-1:0];
                cls_load_act_data = f6_out_data;
            end

            S_LOAD_CLS: begin
                if (load_idx_q < CLS_WGT_COUNT) begin
                    cls_load_wgt_we   = 1'b1;
                    cls_load_wgt_addr = load_idx_q[CLS_WGT_ADDR_W-1:0];
                    cls_load_wgt_data = cls_wgt_rom[load_idx_q];
                end
                if (load_idx_q < CLS_BIAS_COUNT) begin
                    cls_load_bias_we   = 1'b1;
                    cls_load_bias_addr = load_idx_q[CLS_BIAS_ADDR_W-1:0];
                    cls_load_bias_data = cls_bias_rom[load_idx_q];
                end
            end

            default: ;
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            top_stage_q <= S_IDLE;
            load_idx_q  <= '0;
            busy_o      <= 1'b0;
            done_o      <= 1'b0;
            class_o     <= '0;
        end else begin
            done_o <= 1'b0;

            case (top_stage_q)
                S_IDLE: begin
                    if (start_i) begin
                        busy_o      <= 1'b1;
                        load_idx_q  <= '0;
                        top_stage_q <= S_LOAD_C1;
                    end
                end

                S_LOAD_C1: begin
                    if (load_idx_q >= (MAX_LOAD_C1 - 1)) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_KICK_C1;
                    end else begin
                        load_idx_q <= load_idx_q + 1'b1;
                    end
                end
                S_KICK_C1: top_stage_q <= S_RUN_C1;
                S_RUN_C1: begin
                    if (conv_done) top_stage_q <= S_KICK_S2;
                end

                S_KICK_S2: top_stage_q <= S_RUN_S2;
                S_RUN_S2: begin
                    if (pool_done) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_LOAD_C3;
                    end
                end

                S_LOAD_C3: begin
                    if (load_idx_q >= (MAX_LOAD_C3 - 1)) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_KICK_C3;
                    end else begin
                        load_idx_q <= load_idx_q + 1'b1;
                    end
                end
                S_KICK_C3: top_stage_q <= S_RUN_C3;
                S_RUN_C3: begin
                    if (conv_done) top_stage_q <= S_KICK_S4;
                end

                S_KICK_S4: top_stage_q <= S_RUN_S4;
                S_RUN_S4: begin
                    if (pool_done) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_LOAD_C5;
                    end
                end

                S_LOAD_C5: begin
                    if (load_idx_q >= (MAX_LOAD_C5 - 1)) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_KICK_C5;
                    end else begin
                        load_idx_q <= load_idx_q + 1'b1;
                    end
                end
                S_KICK_C5: top_stage_q <= S_RUN_C5;
                S_RUN_C5: begin
                    if (conv_done) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_LOAD_F6;
                    end
                end

                S_LOAD_F6: begin
                    if (load_idx_q >= (MAX_LOAD_F6 - 1)) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_KICK_F6;
                    end else begin
                        load_idx_q <= load_idx_q + 1'b1;
                    end
                end
                S_KICK_F6: top_stage_q <= S_RUN_F6;
                S_RUN_F6: begin
                    if (f6_done) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_LOAD_CLS;
                    end
                end

                S_LOAD_CLS: begin
                    if (load_idx_q >= (MAX_LOAD_CLS - 1)) begin
                        load_idx_q  <= '0;
                        top_stage_q <= S_KICK_CLS;
                    end else begin
                        load_idx_q <= load_idx_q + 1'b1;
                    end
                end
                S_KICK_CLS: top_stage_q <= S_RUN_CLS;
                S_RUN_CLS: begin
                    if (cls_valid) begin
                        class_o     <= cls_class;
                        busy_o      <= 1'b0;
                        done_o      <= 1'b1;
                        top_stage_q <= S_IDLE;
                    end
                end

                default: top_stage_q <= S_IDLE;
            endcase
        end
    end
endmodule
