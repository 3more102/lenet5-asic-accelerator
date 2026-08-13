`timescale 1ns/1ps

// Streaming controller for 2x2 average pooling (S2/S4). Wraps the
// combinational avg_pool2x2_int8 primitive with a conv2d_engine-style
// load/start/stream control interface, reused at runtime for both S2 and
// S4 via cfg_in_w_i/cfg_in_h_i/cfg_in_ch_i. avg_pool2x2_int8 has no
// shift/ReLU ports (shift=2, no ReLU is fixed inside the primitive), so
// this controller exposes none either.
module avg_pool2x2_stream #(
    parameter integer DATA_WIDTH     = 8,
    parameter integer MAX_IN_W       = 28,
    parameter integer MAX_IN_H       = 28,
    parameter integer MAX_IN_CH      = 16,
    parameter integer ACT_DEPTH      = MAX_IN_W * MAX_IN_H * MAX_IN_CH,
    parameter integer ACT_ADDR_WIDTH = (ACT_DEPTH <= 1) ? 1 : $clog2(ACT_DEPTH)
) (
    input  logic                         clk_i,
    input  logic                         rst_ni,

    // Runtime layer configuration. Output size is (in_h/2) x (in_w/2).
    input  logic                         start_i,
    input  logic [7:0]                   cfg_in_w_i,
    input  logic [7:0]                   cfg_in_h_i,
    input  logic [7:0]                   cfg_in_ch_i,
    output logic                         busy_o,
    output logic                         done_o,
    output logic                         config_error_o,

    // Preload interface. Writes are accepted independently of busy_o.
    input  logic                         load_act_we_i,
    input  logic [ACT_ADDR_WIDTH-1:0]    load_act_addr_i,
    input  logic signed [DATA_WIDTH-1:0] load_act_data_i,

    // Output stream is ordered channel, row, column.
    output logic                         out_valid_o,
    input  logic                         out_ready_i,
    output logic signed [DATA_WIDTH-1:0] out_data_o,
    output logic [7:0]                   out_channel_o,
    output logic [7:0]                   out_y_o,
    output logic [7:0]                   out_x_o
);
    logic signed [DATA_WIDTH-1:0] act_mem [0:ACT_DEPTH-1];

    logic [7:0] cfg_in_w_q;
    logic [7:0] cfg_in_h_q;
    logic [7:0] cfg_in_ch_q;
    logic [7:0] cfg_out_w_q;
    logic [7:0] cfg_out_h_q;

    logic [7:0] channel_q;
    logic [7:0] out_y_q;
    logic [7:0] out_x_q;

    logic signed [(4*DATA_WIDTH)-1:0] samples;
    logic signed [DATA_WIDTH-1:0]     average;

    integer base_idx, addr00, addr01, addr10, addr11;

    always_ff @(posedge clk_i) begin
        if (load_act_we_i) begin
            act_mem[load_act_addr_i] <= load_act_data_i;
        end
    end

    // 2x2 window addressing in CHW layout, matching conv2d_engine's
    // addressing convention. Sample lane order (0..3) is documented here
    // for auditability; the sum itself is order-independent.
    always_comb begin
        base_idx = ((integer'(channel_q) * integer'(cfg_in_h_q))
                    + (integer'(out_y_q) * 2)) * integer'(cfg_in_w_q)
                   + (integer'(out_x_q) * 2);
        addr00 = base_idx;                       // (y+0, x+0)
        addr01 = base_idx + 1;                   // (y+0, x+1)
        addr10 = base_idx + integer'(cfg_in_w_q); // (y+1, x+0)
        addr11 = addr10 + 1;                     // (y+1, x+1)

        samples = '0;
        samples[(0*DATA_WIDTH) +: DATA_WIDTH] = act_mem[addr00];
        samples[(1*DATA_WIDTH) +: DATA_WIDTH] = act_mem[addr01];
        samples[(2*DATA_WIDTH) +: DATA_WIDTH] = act_mem[addr10];
        samples[(3*DATA_WIDTH) +: DATA_WIDTH] = act_mem[addr11];
    end

    avg_pool2x2_int8 #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_avg_pool2x2_int8 (
        .samples_i (samples),
        .average_o (average)
    );

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            cfg_in_w_q     <= '0;
            cfg_in_h_q     <= '0;
            cfg_in_ch_q    <= '0;
            cfg_out_w_q    <= '0;
            cfg_out_h_q    <= '0;
            channel_q      <= '0;
            out_y_q        <= '0;
            out_x_q        <= '0;
            out_valid_o    <= 1'b0;
            out_data_o     <= '0;
            out_channel_o  <= '0;
            out_y_o        <= '0;
            out_x_o        <= '0;
            busy_o         <= 1'b0;
            done_o         <= 1'b0;
            config_error_o <= 1'b0;
        end else begin
            done_o <= 1'b0;

            if (!busy_o && start_i) begin
                config_error_o <= 1'b0;
                out_valid_o    <= 1'b0;

                if ((cfg_in_w_i == 0) || (cfg_in_h_i == 0)
                    || (cfg_in_w_i[0] != 1'b0) || (cfg_in_h_i[0] != 1'b0)
                    || (cfg_in_w_i > MAX_IN_W) || (cfg_in_h_i > MAX_IN_H)
                    || (cfg_in_ch_i == 0) || (cfg_in_ch_i > MAX_IN_CH)) begin
                    config_error_o <= 1'b1;
                    busy_o         <= 1'b0;
                end else begin
                    cfg_in_w_q  <= cfg_in_w_i;
                    cfg_in_h_q  <= cfg_in_h_i;
                    cfg_in_ch_q <= cfg_in_ch_i;
                    cfg_out_w_q <= cfg_in_w_i >> 1;
                    cfg_out_h_q <= cfg_in_h_i >> 1;
                    channel_q   <= '0;
                    out_y_q     <= '0;
                    out_x_q     <= '0;
                    busy_o      <= 1'b1;
                end
            end else if (busy_o) begin
                if (out_valid_o) begin
                    if (out_ready_i) begin
                        out_valid_o <= 1'b0;

                        if ((channel_q == (cfg_in_ch_q - 1))
                            && (out_y_q == (cfg_out_h_q - 1))
                            && (out_x_q == (cfg_out_w_q - 1))) begin
                            busy_o <= 1'b0;
                            done_o <= 1'b1;
                        end else if (out_x_q != (cfg_out_w_q - 1)) begin
                            out_x_q <= out_x_q + 1'b1;
                        end else if (out_y_q != (cfg_out_h_q - 1)) begin
                            out_x_q <= '0;
                            out_y_q <= out_y_q + 1'b1;
                        end else begin
                            out_x_q   <= '0;
                            out_y_q   <= '0;
                            channel_q <= channel_q + 1'b1;
                        end
                    end
                end else begin
                    out_data_o    <= average;
                    out_channel_o <= channel_q;
                    out_y_o       <= out_y_q;
                    out_x_o       <= out_x_q;
                    out_valid_o   <= 1'b1;
                end
            end
        end
    end
endmodule
