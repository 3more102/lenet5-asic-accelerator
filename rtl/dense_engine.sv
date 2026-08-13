`timescale 1ns/1ps

// Memory-backed reference engine for a fully-connected (dense) layer, used
// for F6 and (instantiated a second time internally) the output
// classifier's dense stage. Mirrors conv2d_engine's control idiom: this
// module is intended for architecture verification and small prototypes,
// not a production SRAM-backed block. Tensor layout is documented in
// docs/INTERFACES.md.
//
// In addition to the normal requantized int8 output (out_data_o), this
// engine exposes the raw pre-shift/pre-ReLU/pre-saturate accumulator
// (out_acc_o) for each neuron. classifier_argmax compares out_acc_o rather
// than out_data_o -- see golden/quantized_conv.py:argmax_classifier's
// docstring for why an int8-saturated score is unsafe to argmax over.
module dense_engine #(
    parameter integer DATA_WIDTH      = 8,
    parameter integer ACC_WIDTH       = 32,
    parameter integer LANES           = 8,
    parameter integer MAX_IN_LEN      = 120,
    parameter integer MAX_OUT_LEN     = 84,
    parameter integer ACT_DEPTH       = MAX_IN_LEN,
    parameter integer WGT_DEPTH       = MAX_OUT_LEN * MAX_IN_LEN,
    parameter integer BIAS_DEPTH      = MAX_OUT_LEN,
    parameter integer ACT_ADDR_WIDTH  = (ACT_DEPTH  <= 1) ? 1 : $clog2(ACT_DEPTH),
    parameter integer WGT_ADDR_WIDTH  = (WGT_DEPTH  <= 1) ? 1 : $clog2(WGT_DEPTH),
    parameter integer BIAS_ADDR_WIDTH = (BIAS_DEPTH <= 1) ? 1 : $clog2(BIAS_DEPTH)
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,

    // Runtime layer configuration.
    input  logic                         start_i,
    input  logic [7:0]                   cfg_in_len_i,
    input  logic [7:0]                   cfg_out_len_i,
    input  logic [5:0]                   cfg_shift_i,
    input  logic                         cfg_relu_en_i,
    output logic                         busy_o,
    output logic                         done_o,
    output logic                         config_error_o,

    // Preload interfaces. Writes are accepted independently of busy_o.
    input  logic                         load_act_we_i,
    input  logic [ACT_ADDR_WIDTH-1:0]    load_act_addr_i,
    input  logic signed [DATA_WIDTH-1:0] load_act_data_i,

    input  logic                         load_wgt_we_i,
    input  logic [WGT_ADDR_WIDTH-1:0]    load_wgt_addr_i,
    input  logic signed [DATA_WIDTH-1:0] load_wgt_data_i,

    input  logic                         load_bias_we_i,
    input  logic [BIAS_ADDR_WIDTH-1:0]   load_bias_addr_i,
    input  logic signed [ACC_WIDTH-1:0]  load_bias_data_i,

    // Output stream is one neuron per beat, in index order.
    output logic                         out_valid_o,
    input  logic                         out_ready_i,
    output logic signed [DATA_WIDTH-1:0] out_data_o,
    output logic signed [ACC_WIDTH-1:0]  out_acc_o,
    output logic [7:0]                   out_index_o
);
    logic signed [DATA_WIDTH-1:0] act_mem  [0:ACT_DEPTH-1];
    logic signed [DATA_WIDTH-1:0] wgt_mem  [0:WGT_DEPTH-1];
    logic signed [ACC_WIDTH-1:0]  bias_mem [0:BIAS_DEPTH-1];

    logic [7:0] cfg_in_len_q;
    logic [7:0] cfg_out_len_q;
    logic [5:0] cfg_shift_q;
    logic       cfg_relu_en_q;

    logic [7:0] out_idx_q;
    logic [7:0] lane_base_q;
    logic signed [ACC_WIDTH-1:0] accumulator_q;

    logic signed [(LANES*DATA_WIDTH)-1:0] act_lane;
    logic signed [(LANES*DATA_WIDTH)-1:0] wgt_lane;
    logic signed [ACC_WIDTH-1:0]          lane_sum;
    logic signed [ACC_WIDTH-1:0]          quant_source;
    logic signed [DATA_WIDTH-1:0]         quantized;
    logic                                 last_lane_group;

    integer lane_idx, in_idx;

    always_ff @(posedge clk_i) begin
        if (load_act_we_i) begin
            act_mem[load_act_addr_i] <= load_act_data_i;
        end
        if (load_wgt_we_i) begin
            wgt_mem[load_wgt_addr_i] <= load_wgt_data_i;
        end
        if (load_bias_we_i) begin
            bias_mem[load_bias_addr_i] <= load_bias_data_i;
        end
    end

    // Zero-padding out-of-range lanes makes the final partial lane group
    // of ceil(in_len/LANES) groups correct with no special-case logic.
    always_comb begin
        act_lane = '0;
        wgt_lane = '0;
        for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
            in_idx = lane_base_q + lane_idx;
            if (in_idx < cfg_in_len_q) begin
                act_lane[(lane_idx*DATA_WIDTH) +: DATA_WIDTH] = act_mem[in_idx];
                wgt_lane[(lane_idx*DATA_WIDTH) +: DATA_WIDTH] =
                    wgt_mem[(out_idx_q * cfg_in_len_q) + in_idx];
            end
        end
        last_lane_group = (lane_base_q + LANES) >= cfg_in_len_q;
    end

    dense_row_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .LANES     (LANES)
    ) u_dense_row_mac (
        .act_lane_i (act_lane),
        .wgt_lane_i (wgt_lane),
        .lane_sum_o (lane_sum)
    );

    always_comb begin
        quant_source = accumulator_q + lane_sum;
    end

    requantize #(
        .ACC_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(DATA_WIDTH)
    ) u_requantize (
        .acc_i     (quant_source),
        .shift_i   (cfg_shift_q),
        .relu_en_i (cfg_relu_en_q),
        .data_o    (quantized)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cfg_in_len_q   <= '0;
            cfg_out_len_q  <= '0;
            cfg_shift_q    <= '0;
            cfg_relu_en_q  <= 1'b0;
            out_idx_q      <= '0;
            lane_base_q    <= '0;
            accumulator_q  <= '0;
            out_valid_o    <= 1'b0;
            out_data_o     <= '0;
            out_acc_o      <= '0;
            out_index_o    <= '0;
            busy_o         <= 1'b0;
            done_o         <= 1'b0;
            config_error_o <= 1'b0;
        end else begin
            done_o <= 1'b0;

            if (!busy_o && start_i) begin
                config_error_o <= 1'b0;
                out_valid_o    <= 1'b0;

                if ((cfg_in_len_i == 0) || (cfg_in_len_i > MAX_IN_LEN)
                    || (cfg_out_len_i == 0) || (cfg_out_len_i > MAX_OUT_LEN)) begin
                    config_error_o <= 1'b1;
                    busy_o         <= 1'b0;
                end else begin
                    cfg_in_len_q  <= cfg_in_len_i;
                    cfg_out_len_q <= cfg_out_len_i;
                    cfg_shift_q   <= cfg_shift_i;
                    cfg_relu_en_q <= cfg_relu_en_i;
                    out_idx_q     <= '0;
                    lane_base_q   <= '0;
                    accumulator_q <= bias_mem[0];
                    busy_o        <= 1'b1;
                end
            end else if (busy_o) begin
                if (out_valid_o) begin
                    if (out_ready_i) begin
                        out_valid_o <= 1'b0;

                        if (out_idx_q == (cfg_out_len_q - 1)) begin
                            busy_o <= 1'b0;
                            done_o <= 1'b1;
                        end else begin
                            out_idx_q     <= out_idx_q + 1'b1;
                            lane_base_q   <= '0;
                            accumulator_q <= bias_mem[out_idx_q + 1'b1];
                        end
                    end
                end else if (last_lane_group) begin
                    out_data_o  <= quantized;
                    out_acc_o   <= quant_source;
                    out_index_o <= out_idx_q;
                    out_valid_o <= 1'b1;
                end else begin
                    accumulator_q <= quant_source;
                    lane_base_q   <= lane_base_q + LANES;
                end
            end
        end
    end
endmodule
