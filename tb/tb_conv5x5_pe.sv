`timescale 1ns/1ps

module tb_conv5x5_pe;
    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic first;
    logic last;
    logic signed [39:0] act_row;
    logic signed [39:0] wgt_row;
    logic signed [31:0] bias;
    logic [5:0] shift;
    logic relu_en;
    logic out_valid;
    logic out_ready;
    logic signed [7:0] out_data;
    integer row;
    integer lane;

    conv5x5_pe dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .in_valid_i(in_valid),
        .in_ready_o(in_ready),
        .first_i(first),
        .last_i(last),
        .act_row_i(act_row),
        .wgt_row_i(wgt_row),
        .bias_i(bias),
        .shift_i(shift),
        .relu_en_i(relu_en),
        .out_valid_o(out_valid),
        .out_ready_i(out_ready),
        .out_data_o(out_data)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        in_valid = 0;
        first = 0;
        last = 0;
        act_row = '0;
        wgt_row = '0;
        bias = 32'sd5;
        shift = 0;
        relu_en = 0;
        out_ready = 0;

        repeat (3) @(negedge clk);
        rst_n = 1;

        // Five rows: sum(row r)=5*(r+1). Total=75, plus bias=5 -> 80.
        for (row = 0; row < 5; row = row + 1) begin
            @(negedge clk);
            for (lane = 0; lane < 5; lane = lane + 1) begin
                act_row[(lane*8) +: 8] = 8'sd1;
                wgt_row[(lane*8) +: 8] = row + 1;
            end
            first = (row == 0);
            last = (row == 4);
            in_valid = 1;
            wait (in_ready);
        end
        @(negedge clk);
        in_valid = 0;
        first = 0;
        last = 0;

        wait (out_valid);
        repeat (3) begin
            @(posedge clk);
            if (!out_valid || ($signed(out_data) != 80)) begin
                $fatal(1, "Output was not held correctly under backpressure");
            end
        end
        @(negedge clk);
        out_ready = 1;
        @(posedge clk);
        if ($signed(out_data) != 80) begin
            $fatal(1, "PE result mismatch: got %0d, expected 80", $signed(out_data));
        end
        @(negedge clk);
        $display("PASS tb_conv5x5_pe: accumulation, bias, and backpressure");
        $finish;
    end
endmodule

