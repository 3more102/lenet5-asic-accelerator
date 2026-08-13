`timescale 1ns/1ps

// Dense(IN->NUM_CLASSES) classifier head with an integrated running-max
// argmax comparator. Wraps one internal dense_engine (ReLU disabled,
// output backpressure tied high so every one of the NUM_CLASSES scores is
// consumed as soon as it is produced) and compares its raw pre-requantize
// accumulator (out_acc_o), not its saturated int8 score -- matching
// golden/quantized_conv.py:argmax_classifier. See that function's
// docstring for why an int8-saturated score is unsafe to argmax over.
// Ties resolve to the lowest class index: the comparator below uses a
// strict '>' so a later, equal score never overwrites an earlier max.
module classifier_argmax #(
    parameter integer DATA_WIDTH      = 8,
    parameter integer ACC_WIDTH       = 32,
    parameter integer LANES           = 8,
    parameter integer MAX_IN_LEN      = 84,
    parameter integer NUM_CLASSES     = 10,
    parameter integer ACT_DEPTH       = MAX_IN_LEN,
    parameter integer WGT_DEPTH       = NUM_CLASSES * MAX_IN_LEN,
    parameter integer BIAS_DEPTH      = NUM_CLASSES,
    parameter integer ACT_ADDR_WIDTH  = (ACT_DEPTH  <= 1) ? 1 : $clog2(ACT_DEPTH),
    parameter integer WGT_ADDR_WIDTH  = (WGT_DEPTH  <= 1) ? 1 : $clog2(WGT_DEPTH),
    parameter integer BIAS_ADDR_WIDTH = (BIAS_DEPTH <= 1) ? 1 : $clog2(BIAS_DEPTH)
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,

    input  logic                         start_i,
    input  logic [7:0]                   cfg_in_len_i,
    output logic                         busy_o,
    output logic                         done_o,
    output logic                         config_error_o,

    input  logic                         load_act_we_i,
    input  logic [ACT_ADDR_WIDTH-1:0]    load_act_addr_i,
    input  logic signed [DATA_WIDTH-1:0] load_act_data_i,

    input  logic                         load_wgt_we_i,
    input  logic [WGT_ADDR_WIDTH-1:0]    load_wgt_addr_i,
    input  logic signed [DATA_WIDTH-1:0] load_wgt_data_i,

    input  logic                         load_bias_we_i,
    input  logic [BIAS_ADDR_WIDTH-1:0]   load_bias_addr_i,
    input  logic signed [ACC_WIDTH-1:0]  load_bias_data_i,

    output logic [3:0]                   class_o,
    output logic                         valid_o
);
    logic dense_busy, dense_done, dense_config_error;
    logic dense_out_valid;
    logic signed [ACC_WIDTH-1:0] dense_out_acc;
    logic [7:0] dense_out_index;

    dense_engine #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .LANES      (LANES),
        .MAX_IN_LEN (MAX_IN_LEN),
        .MAX_OUT_LEN(NUM_CLASSES)
    ) u_dense_engine (
        .clk_i         (clk_i),
        .rst_ni        (rst_ni),
        .start_i       (start_i),
        .cfg_in_len_i  (cfg_in_len_i),
        .cfg_out_len_i (8'(NUM_CLASSES)),
        .cfg_shift_i   (6'd0),
        .cfg_relu_en_i (1'b0),
        .busy_o        (dense_busy),
        .done_o        (dense_done),
        .config_error_o(dense_config_error),

        .load_act_we_i  (load_act_we_i),
        .load_act_addr_i(load_act_addr_i),
        .load_act_data_i(load_act_data_i),

        .load_wgt_we_i  (load_wgt_we_i),
        .load_wgt_addr_i(load_wgt_addr_i),
        .load_wgt_data_i(load_wgt_data_i),

        .load_bias_we_i  (load_bias_we_i),
        .load_bias_addr_i(load_bias_addr_i),
        .load_bias_data_i(load_bias_data_i),

        .out_valid_o (dense_out_valid),
        .out_ready_i (1'b1),
        .out_data_o  (),
        .out_acc_o   (dense_out_acc),
        .out_index_o (dense_out_index)
    );

    assign busy_o         = dense_busy;
    assign done_o         = dense_done;
    assign config_error_o = dense_config_error;

    logic signed [ACC_WIDTH-1:0] best_score_q;
    logic [3:0]                  best_index_q;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            best_score_q <= '0;
            best_index_q <= '0;
            class_o      <= '0;
            valid_o      <= 1'b0;
        end else begin
            if (!dense_busy && start_i) begin
                best_score_q <= {1'b1, {(ACC_WIDTH-1){1'b0}}}; // most-negative value
                best_index_q <= '0;
                valid_o      <= 1'b0;
            end else if (dense_busy && dense_out_valid) begin
                if (dense_out_acc > best_score_q) begin
                    best_score_q <= dense_out_acc;
                    best_index_q <= dense_out_index[3:0];
                end
            end

            if (dense_done) begin
                class_o <= best_index_q;
                valid_o <= 1'b1;
            end
        end
    end
endmodule
