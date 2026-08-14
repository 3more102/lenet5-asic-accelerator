`timescale 1ns/1ps

// Memory-backed reference engine for valid 5x5 convolution.
//
// This module is intended for architecture verification and small prototypes.
// For a production ASIC, replace act_mem and wgt_mem with foundry SRAM macros
// and retain the controller/PE interface. Tensor layout is documented in
// docs/INTERFACES.md.
module conv2d_engine #(
    parameter integer DATA_WIDTH     = 8,
    parameter integer ACC_WIDTH      = 32,
    parameter integer MAX_IN_W       = 32,
    parameter integer MAX_IN_H       = 32,
    parameter integer MAX_IN_CH      = 16,
    parameter integer MAX_OUT_CH     = 120,
    parameter integer ACT_DEPTH      = MAX_IN_W * MAX_IN_H * MAX_IN_CH,
    parameter integer WGT_DEPTH      = MAX_OUT_CH * MAX_IN_CH * 25,
    parameter integer BIAS_DEPTH     = MAX_OUT_CH,
    parameter integer CONN_DEPTH     = MAX_OUT_CH * MAX_IN_CH,
    parameter integer ACT_ADDR_WIDTH = (ACT_DEPTH <= 1) ? 1 : $clog2(ACT_DEPTH),
    parameter integer WGT_ADDR_WIDTH = (WGT_DEPTH <= 1) ? 1 : $clog2(WGT_DEPTH),
    parameter integer BIAS_ADDR_WIDTH = (BIAS_DEPTH <= 1) ? 1 : $clog2(BIAS_DEPTH),
    parameter integer CONN_ADDR_WIDTH = (CONN_DEPTH <= 1) ? 1 : $clog2(CONN_DEPTH)
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,

    // Runtime layer configuration. The engine implements K=5, stride=1,
    // padding=0; therefore output size is (in_h-4) x (in_w-4).
    input  logic                         start_i,
    input  logic [7:0]                   cfg_in_w_i,
    input  logic [7:0]                   cfg_in_h_i,
    input  logic [7:0]                   cfg_in_ch_i,
    input  logic [7:0]                   cfg_out_ch_i,
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

    input  logic                         load_conn_we_i,
    input  logic [CONN_ADDR_WIDTH-1:0]   load_conn_addr_i,
    input  logic                         load_conn_data_i,

    // Output stream is ordered output-channel, row, column.
    output logic                         out_valid_o,
    input  logic                         out_ready_i,
    output logic signed [DATA_WIDTH-1:0] out_data_o,
    output logic [7:0]                   out_channel_o,
    output logic [7:0]                   out_y_o,
    output logic [7:0]                   out_x_o
);
    logic signed [DATA_WIDTH-1:0] act_mem  [0:ACT_DEPTH-1];
    logic signed [DATA_WIDTH-1:0] wgt_mem  [0:WGT_DEPTH-1];
    logic signed [ACC_WIDTH-1:0]  bias_mem [0:BIAS_DEPTH-1];
    logic                         conn_mem [0:CONN_DEPTH-1];

    logic [7:0] cfg_in_w_q;
    logic [7:0] cfg_in_h_q;
    logic [7:0] cfg_in_ch_q;
    logic [7:0] cfg_out_ch_q;
    logic [7:0] cfg_out_w_q;
    logic [7:0] cfg_out_h_q;
    logic [5:0] cfg_shift_q;
    logic       cfg_relu_en_q;

    logic [7:0] output_channel_q;
    logic [7:0] output_y_q;
    logic [7:0] output_x_q;
    logic [7:0] input_channel_q;
    logic [2:0] kernel_row_q;

    logic signed [(5*DATA_WIDTH)-1:0] act_row;
    logic signed [(5*DATA_WIDTH)-1:0] wgt_row;
    logic signed [ACC_WIDTH-1:0] row_sum;
    logic signed [ACC_WIDTH-1:0] accumulator_q;
    logic signed [ACC_WIDTH-1:0] quant_source;
    logic signed [DATA_WIDTH-1:0] quantized;
    logic current_connected;

    // Requantization pipeline stage. quant_source is registered here before it
    // reaches `requantize`, so the act_mem/wgt_mem read, the row-MAC adder tree
    // and the requantizer's carry chain no longer share one combinational
    // cycle. The coordinates travel alongside the accumulator so the output
    // stream stays correctly tagged one cycle later.
    logic signed [ACC_WIDTH-1:0] rq_acc_q;
    logic                        rq_pending_q;
    logic [7:0]                  rq_channel_q;
    logic [7:0]                  rq_y_q;
    logic [7:0]                  rq_x_q;

    integer lane_idx;
    integer act_linear_index;
    integer wgt_linear_index;
    integer conn_linear_index;

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
        if (load_conn_we_i) begin
            conn_mem[load_conn_addr_i] <= load_conn_data_i;
        end
    end

    always_comb begin
        act_row = '0;
        wgt_row = '0;
        for (lane_idx = 0; lane_idx < 5; lane_idx = lane_idx + 1) begin
            act_linear_index =
                (((input_channel_q * cfg_in_h_q) + (output_y_q + kernel_row_q))
                 * cfg_in_w_q) + output_x_q + lane_idx;
            wgt_linear_index =
                ((((output_channel_q * cfg_in_ch_q) + input_channel_q) * 25)
                 + (kernel_row_q * 5)) + lane_idx;
            act_row[(lane_idx*DATA_WIDTH) +: DATA_WIDTH] =
                act_mem[act_linear_index];
            wgt_row[(lane_idx*DATA_WIDTH) +: DATA_WIDTH] =
                wgt_mem[wgt_linear_index];
        end

        conn_linear_index =
            (output_channel_q * cfg_in_ch_q) + input_channel_q;
        current_connected = conn_mem[conn_linear_index];
    end

    conv5x5_row_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) u_row_mac (
        .act_row_i (act_row),
        .wgt_row_i (wgt_row),
        .row_sum_o (row_sum)
    );

    always_comb begin
        quant_source = current_connected
                     ? (accumulator_q + row_sum)
                     : accumulator_q;
    end

    requantize #(
        .ACC_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(DATA_WIDTH)
    ) u_requantize (
        .acc_i     (rq_acc_q),
        .shift_i   (cfg_shift_q),
        .relu_en_i (cfg_relu_en_q),
        .data_o    (quantized)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cfg_in_w_q       <= '0;
            cfg_in_h_q       <= '0;
            cfg_in_ch_q      <= '0;
            cfg_out_ch_q     <= '0;
            cfg_out_w_q      <= '0;
            cfg_out_h_q      <= '0;
            cfg_shift_q      <= '0;
            cfg_relu_en_q    <= 1'b0;
            output_channel_q <= '0;
            output_y_q       <= '0;
            output_x_q       <= '0;
            input_channel_q  <= '0;
            kernel_row_q     <= '0;
            accumulator_q    <= '0;
            rq_acc_q         <= '0;
            rq_pending_q     <= 1'b0;
            rq_channel_q     <= '0;
            rq_y_q           <= '0;
            rq_x_q           <= '0;
            out_valid_o      <= 1'b0;
            out_data_o       <= '0;
            out_channel_o    <= '0;
            out_y_o          <= '0;
            out_x_o          <= '0;
            busy_o           <= 1'b0;
            done_o           <= 1'b0;
            config_error_o   <= 1'b0;
        end else begin
            done_o <= 1'b0;

            if (!busy_o && start_i) begin
                config_error_o <= 1'b0;
                out_valid_o    <= 1'b0;

                if ((cfg_in_w_i < 5) || (cfg_in_h_i < 5)
                    || (cfg_in_w_i > MAX_IN_W) || (cfg_in_h_i > MAX_IN_H)
                    || (cfg_in_ch_i == 0) || (cfg_in_ch_i > MAX_IN_CH)
                    || (cfg_out_ch_i == 0) || (cfg_out_ch_i > MAX_OUT_CH)) begin
                    config_error_o <= 1'b1;
                    busy_o         <= 1'b0;
                end else begin
                    cfg_in_w_q       <= cfg_in_w_i;
                    cfg_in_h_q       <= cfg_in_h_i;
                    cfg_in_ch_q      <= cfg_in_ch_i;
                    cfg_out_ch_q     <= cfg_out_ch_i;
                    cfg_out_w_q      <= cfg_in_w_i - 4;
                    cfg_out_h_q      <= cfg_in_h_i - 4;
                    cfg_shift_q      <= cfg_shift_i;
                    cfg_relu_en_q    <= cfg_relu_en_i;
                    output_channel_q <= '0;
                    output_y_q       <= '0;
                    output_x_q       <= '0;
                    input_channel_q  <= '0;
                    kernel_row_q     <= '0;
                    accumulator_q    <= bias_mem[0];
                    rq_pending_q     <= 1'b0;
                    busy_o           <= 1'b1;
                end
            end else if (busy_o) begin
                // Second pipeline stage: the accumulator latched last cycle has
                // now been requantized, so publish it. Reached only with
                // out_valid_o low, since rq_pending_q is set exclusively from
                // branches that require that.
                if (rq_pending_q) begin
                    out_data_o    <= quantized;
                    out_channel_o <= rq_channel_q;
                    out_y_o       <= rq_y_q;
                    out_x_o       <= rq_x_q;
                    out_valid_o   <= 1'b1;
                    rq_pending_q  <= 1'b0;
                end else if (out_valid_o) begin
                    if (out_ready_i) begin
                        out_valid_o <= 1'b0;

                        if ((output_channel_q == (cfg_out_ch_q - 1))
                            && (output_y_q == (cfg_out_h_q - 1))
                            && (output_x_q == (cfg_out_w_q - 1))) begin
                            busy_o <= 1'b0;
                            done_o <= 1'b1;
                        end else begin
                            input_channel_q <= '0;
                            kernel_row_q    <= '0;

                            if (output_x_q != (cfg_out_w_q - 1)) begin
                                output_x_q    <= output_x_q + 1'b1;
                                accumulator_q <= bias_mem[output_channel_q];
                            end else if (output_y_q != (cfg_out_h_q - 1)) begin
                                output_x_q    <= '0;
                                output_y_q    <= output_y_q + 1'b1;
                                accumulator_q <= bias_mem[output_channel_q];
                            end else begin
                                output_x_q       <= '0;
                                output_y_q       <= '0;
                                output_channel_q <= output_channel_q + 1'b1;
                                accumulator_q    <= bias_mem[output_channel_q + 1'b1];
                            end
                        end
                    end
                end else if (!current_connected) begin
                    if (input_channel_q == (cfg_in_ch_q - 1)) begin
                        rq_acc_q     <= quant_source;
                        rq_channel_q <= output_channel_q;
                        rq_y_q       <= output_y_q;
                        rq_x_q       <= output_x_q;
                        rq_pending_q <= 1'b1;
                    end else begin
                        input_channel_q <= input_channel_q + 1'b1;
                        kernel_row_q    <= '0;
                    end
                end else if (kernel_row_q == 3'd4) begin
                    if (input_channel_q == (cfg_in_ch_q - 1)) begin
                        rq_acc_q     <= quant_source;
                        rq_channel_q <= output_channel_q;
                        rq_y_q       <= output_y_q;
                        rq_x_q       <= output_x_q;
                        rq_pending_q <= 1'b1;
                    end else begin
                        accumulator_q   <= accumulator_q + row_sum;
                        input_channel_q <= input_channel_q + 1'b1;
                        kernel_row_q    <= '0;
                    end
                end else begin
                    accumulator_q <= accumulator_q + row_sum;
                    kernel_row_q  <= kernel_row_q + 1'b1;
                end
            end
        end
    end
endmodule

