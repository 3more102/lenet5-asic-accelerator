`timescale 1ns/1ps
`include "vectors/config.svh"

module tb_classifier_argmax;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;
    localparam integer ACT_DEPTH  = `TV_CLS_ACT_COUNT;
    localparam integer WGT_DEPTH  = `TV_CLS_WGT_COUNT;
    localparam integer BIAS_DEPTH = `TV_CLS_BIAS_COUNT;
    localparam integer ACT_AW  = (ACT_DEPTH  <= 1) ? 1 : $clog2(ACT_DEPTH);
    localparam integer WGT_AW  = (WGT_DEPTH  <= 1) ? 1 : $clog2(WGT_DEPTH);
    localparam integer BIAS_AW = (BIAS_DEPTH <= 1) ? 1 : $clog2(BIAS_DEPTH);
    localparam logic [7:0] CFG_IN_LEN = `TV_CLS_IN_LEN;

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic config_error;
    logic [3:0] class_val;
    logic       valid;

    logic load_act_we;
    logic [ACT_AW-1:0] load_act_addr;
    logic signed [7:0] load_act_data;
    logic load_wgt_we;
    logic [WGT_AW-1:0] load_wgt_addr;
    logic signed [7:0] load_wgt_data;
    logic load_bias_we;
    logic [BIAS_AW-1:0] load_bias_addr;
    logic signed [31:0] load_bias_data;

    logic [7:0]  act_vector [0:`TV_CLS_ACT_COUNT-1];
    logic [7:0]  wgt_vector [0:`TV_CLS_WGT_COUNT-1];
    logic [31:0] bias_vector [0:`TV_CLS_BIAS_COUNT-1];

    classifier_argmax #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH),
        .MAX_IN_LEN (`TV_CLS_IN_LEN),
        .NUM_CLASSES(`TV_CLS_NUM_CLASSES),
        .ACT_DEPTH  (ACT_DEPTH),
        .WGT_DEPTH  (WGT_DEPTH),
        .BIAS_DEPTH (BIAS_DEPTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .cfg_in_len_i(CFG_IN_LEN),
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
        .class_o(class_val),
        .valid_o(valid)
    );

    always #5 clk = ~clk;

    task load_activations;
        integer index;
        begin
            for (index = 0; index < `TV_CLS_ACT_COUNT; index = index + 1) begin
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
            for (index = 0; index < `TV_CLS_WGT_COUNT; index = index + 1) begin
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
            for (index = 0; index < `TV_CLS_BIAS_COUNT; index = index + 1) begin
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
        $readmemh("vectors/classifier/act.hex", act_vector);
        $readmemh("vectors/classifier/wgt.hex", wgt_vector);
        $readmemh("vectors/classifier/bias.hex", bias_vector);

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
        wait (valid === 1'b1);
        @(negedge clk);

        if (config_error) begin
            $fatal(1, "DUT rejected a valid configuration");
        end
        if (class_val !== `TV_CLS_EXPECTED_CLASS) begin
            $fatal(
                1,
                "Class mismatch: got %0d, expected %0d",
                class_val,
                `TV_CLS_EXPECTED_CLASS
            );
        end
        $display("PASS tb_classifier_argmax: predicted class %0d matched the golden model", class_val);
        $finish;
    end

    initial begin
        repeat (200000) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
