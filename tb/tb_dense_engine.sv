`timescale 1ns/1ps
`include "vectors/config.svh"

// Verifies dense_engine in an F6-shaped configuration.
module tb_dense_engine;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;
    localparam integer ACT_DEPTH  = `TV_F6_ACT_COUNT;
    localparam integer WGT_DEPTH  = `TV_F6_WGT_COUNT;
    localparam integer BIAS_DEPTH = `TV_F6_BIAS_COUNT;
    localparam integer ACT_AW  = (ACT_DEPTH  <= 1) ? 1 : $clog2(ACT_DEPTH);
    localparam integer WGT_AW  = (WGT_DEPTH  <= 1) ? 1 : $clog2(WGT_DEPTH);
    localparam integer BIAS_AW = (BIAS_DEPTH <= 1) ? 1 : $clog2(BIAS_DEPTH);
    localparam logic [7:0] CFG_IN_LEN  = `TV_F6_IN_LEN;
    localparam logic [7:0] CFG_OUT_LEN = `TV_F6_OUT_LEN;
    localparam logic [5:0] CFG_SHIFT   = `TV_F6_SHIFT;
    localparam logic       CFG_RELU    = `TV_F6_RELU;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic config_error;
    logic out_valid;
    logic out_ready;
    logic signed [7:0]  out_data;
    logic signed [31:0] out_acc;
    logic [7:0] out_index;

    logic load_act_we;
    logic [ACT_AW-1:0] load_act_addr;
    logic signed [7:0] load_act_data;
    logic load_wgt_we;
    logic [WGT_AW-1:0] load_wgt_addr;
    logic signed [7:0] load_wgt_data;
    logic load_bias_we;
    logic [BIAS_AW-1:0] load_bias_addr;
    logic signed [31:0] load_bias_data;

    logic [7:0]  act_vector [0:`TV_F6_ACT_COUNT-1];
    logic [7:0]  wgt_vector [0:`TV_F6_WGT_COUNT-1];
    logic [31:0] bias_vector [0:`TV_F6_BIAS_COUNT-1];
    logic [7:0]  expected_vector [0:`TV_F6_OUT_COUNT-1];

    integer cycle_count;
    integer received_count;

    dense_engine #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .MAX_IN_LEN (`TV_F6_IN_LEN),
        .MAX_OUT_LEN(`TV_F6_OUT_LEN),
        .ACT_DEPTH  (ACT_DEPTH),
        .WGT_DEPTH  (WGT_DEPTH),
        .BIAS_DEPTH (BIAS_DEPTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .cfg_in_len_i(CFG_IN_LEN),
        .cfg_out_len_i(CFG_OUT_LEN),
        .cfg_shift_i(CFG_SHIFT),
        .cfg_relu_en_i(CFG_RELU),
        .busy_o(busy),
        .done_o(done),
        .config_error_o(config_error),
        .load_act_we_i(load_act_we),
        .load_act_addr_i(load_act_addr),
        .load_act_data_i(load_act_data),
        .load_wgt_we_i(load_wgt_we),
        .load_wgt_addr_i(load_wgt_addr),
        .load_wgt_data_i(load_wgt_data),
        .load_bias_we_i(load_bias_we),
        .load_bias_addr_i(load_bias_addr),
        .load_bias_data_i(load_bias_data),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data),
        .out_acc_o(out_acc),
        .out_index_o(out_index)
    );

    always #5 clk = ~clk;

    // Deterministic backpressure proves that valid/data/index are held.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            out_ready   <= 1'b0;
        end else begin
            cycle_count <= cycle_count + 1;
            out_ready   <= ((cycle_count % 7) != 3);
        end
    end

    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            if (received_count >= `TV_F6_OUT_COUNT) begin
                $fatal(1, "Received more outputs than expected");
            end
            if ($signed(out_data) !== $signed(expected_vector[received_count])) begin
                $fatal(
                    1,
                    "Data mismatch at index %0d: RTL=%0d golden=%0d",
                    received_count,
                    $signed(out_data),
                    $signed(expected_vector[received_count])
                );
            end
            if (out_index !== received_count) begin
                $fatal(
                    1,
                    "Index mismatch: got %0d, expected %0d",
                    out_index,
                    received_count
                );
            end
            received_count <= received_count + 1;
        end
    end

    task load_activations;
        integer index;
        begin
            for (index = 0; index < `TV_F6_ACT_COUNT; index = index + 1) begin
                @(negedge clk);
                load_act_we   = 1'b1;
                load_act_addr = index[ACT_AW-1:0];
                load_act_data = act_vector[index];
            end
            @(negedge clk);
            load_act_we = 1'b0;
        end
    endtask

    task load_weights;
        integer index;
        begin
            for (index = 0; index < `TV_F6_WGT_COUNT; index = index + 1) begin
                @(negedge clk);
                load_wgt_we   = 1'b1;
                load_wgt_addr = index[WGT_AW-1:0];
                load_wgt_data = wgt_vector[index];
            end
            @(negedge clk);
            load_wgt_we = 1'b0;
        end
    endtask

    task load_biases;
        integer index;
        begin
            for (index = 0; index < `TV_F6_BIAS_COUNT; index = index + 1) begin
                @(negedge clk);
                load_bias_we   = 1'b1;
                load_bias_addr = index[BIAS_AW-1:0];
                load_bias_data = bias_vector[index];
            end
            @(negedge clk);
            load_bias_we = 1'b0;
        end
    endtask

    initial begin
        $readmemh("vectors/f6/act.hex", act_vector);
        $readmemh("vectors/f6/wgt.hex", wgt_vector);
        $readmemh("vectors/f6/bias.hex", bias_vector);
        $readmemh("vectors/f6/expected.hex", expected_vector);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        load_act_we = 1'b0;
        load_wgt_we = 1'b0;
        load_bias_we = 1'b0;
        load_act_addr = '0;
        load_wgt_addr = '0;
        load_bias_addr = '0;
        load_act_data = '0;
        load_wgt_data = '0;
        load_bias_data = '0;
        received_count = 0;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        load_activations();
        load_weights();
        load_biases();

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (busy === 1'b1);
        wait (done === 1'b1);
        @(negedge clk);

        if (config_error) begin
            $fatal(1, "DUT rejected a valid configuration");
        end
        if (received_count != `TV_F6_OUT_COUNT) begin
            $fatal(
                1,
                "Output count mismatch: got %0d, expected %0d",
                received_count,
                `TV_F6_OUT_COUNT
            );
        end
        $display(
            "PASS tb_dense_engine: %0d outputs matched the Python golden model",
            received_count
        );
        $finish;
    end

    initial begin
        repeat (200000) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
