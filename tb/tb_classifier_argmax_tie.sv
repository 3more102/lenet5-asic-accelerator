`timescale 1ns/1ps

// Hand-computed tie-break check for classifier_argmax: three classes with
// accumulators [5, 1, 5] (a single-input dot product against weights
// [5, 1, 5], bias 0). Classes 0 and 2 tie for the highest accumulator;
// the lowest index (0) must win, matching np.argmax's tie-break in
// golden/quantized_conv.py:argmax_classifier (see
// golden/test_golden.py:test_argmax_classifier_tie_breaks_to_lowest_index
// for the equivalent golden-model case).
module tb_classifier_argmax_tie;
    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic config_error;
    logic [3:0] class_val;
    logic       valid;

    logic load_act_we;
    logic [0:0] load_act_addr;
    logic signed [7:0] load_act_data;
    logic load_wgt_we;
    logic [1:0] load_wgt_addr;
    logic signed [7:0] load_wgt_data;
    logic load_bias_we;
    logic [1:0] load_bias_addr;
    logic signed [31:0] load_bias_data;

    integer index;

    classifier_argmax #(
        .DATA_WIDTH (8),
        .ACC_WIDTH  (32),
        .MAX_IN_LEN (1),
        .NUM_CLASSES(3)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .cfg_in_len_i(8'd1),
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

    initial begin
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

        @(negedge clk);
        load_act_we   = 1'b1;
        load_act_addr = 1'd0;
        load_act_data = 8'sd1;
        @(negedge clk);
        load_act_we = 1'b0;

        for (index = 0; index < 3; index = index + 1) begin
            @(negedge clk);
            load_wgt_we   = 1'b1;
            load_wgt_addr = index[1:0];
            load_wgt_data = (index == 1) ? 8'sd1 : 8'sd5; // weights [5, 1, 5]
            load_bias_we   = 1'b1;
            load_bias_addr = index[1:0];
            load_bias_data = 32'sd0;
        end
        @(negedge clk);
        load_wgt_we  = 1'b0;
        load_bias_we = 1'b0;

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
        if (class_val !== 4'd0) begin
            $fatal(1, "Tie-break mismatch: got class %0d, expected lowest index 0", class_val);
        end
        $display("PASS tb_classifier_argmax_tie: tied max score resolved to lowest index");
        $finish;
    end

    initial begin
        repeat (5000) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
