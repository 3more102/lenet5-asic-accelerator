`timescale 1ns/1ps
`include "vectors/config.svh"

// End-to-end check of lenet5_top against golden/deploy.py:deploy_forward_int8
// on the full canonical 32x32x1 shape (the only shape the C1->S4->C5 chain
// collapses correctly at). Unlike the per-module testbenches, the
// scoreboard here is a single terminal assertion of the predicted class,
// since lenet5_top exposes only the final decision, not per-stage taps.
module tb_lenet5_top;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;

    // Mirrors lenet5_top's own ROM_ADDR_WIDTH derivation so this testbench
    // cannot silently drift out of sync with the DUT.
    localparam integer ROM_ADDR_WIDTH = $clog2(120*16*25);

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic [3:0] class_val;

    logic load_we;
    logic [3:0] rom_sel;
    logic [ROM_ADDR_WIDTH-1:0] load_addr;
    logic signed [ACC_WIDTH-1:0] load_data;

    lenet5_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .busy_o(busy),
        .done_o(done),
        .class_o(class_val),
        .load_we_i(load_we),
        .cfg_rom_sel_i(rom_sel),
        .load_addr_i(load_addr),
        .load_data_i(load_data)
    );

    logic signed [7:0]  image_vector   [0:32*32-1];
    logic signed [7:0]  c1_wgt_vector  [0:6*1*25-1];
    logic signed [31:0] c1_bias_vector [0:6-1];
    logic signed [7:0]  c3_wgt_vector  [0:16*6*25-1];
    logic signed [31:0] c3_bias_vector [0:16-1];
    logic signed [7:0]  c5_wgt_vector  [0:120*16*25-1];
    logic signed [31:0] c5_bias_vector [0:120-1];
    logic signed [7:0]  f6_wgt_vector  [0:84*120-1];
    logic signed [31:0] f6_bias_vector [0:84-1];
    logic signed [7:0]  cls_wgt_vector [0:10*84-1];
    logic signed [31:0] cls_bias_vector [0:10-1];

    always #5 clk = ~clk;

    task automatic load_narrow(input integer sel, input integer count);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                load_we   = 1'b1;
                rom_sel   = sel[3:0];
                load_addr = index[ROM_ADDR_WIDTH-1:0];
                case (sel)
                    0:  load_data = image_vector[index];
                    1:  load_data = c1_wgt_vector[index];
                    3:  load_data = c3_wgt_vector[index];
                    5:  load_data = c5_wgt_vector[index];
                    7:  load_data = f6_wgt_vector[index];
                    9:  load_data = cls_wgt_vector[index];
                    default: load_data = '0;
                endcase
            end
            @(negedge clk);
            load_we = 1'b0;
        end
    endtask

    task automatic load_wide(input integer sel, input integer count);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                load_we   = 1'b1;
                rom_sel   = sel[3:0];
                load_addr = index[ROM_ADDR_WIDTH-1:0];
                case (sel)
                    2:  load_data = c1_bias_vector[index];
                    4:  load_data = c3_bias_vector[index];
                    6:  load_data = c5_bias_vector[index];
                    8:  load_data = f6_bias_vector[index];
                    10: load_data = cls_bias_vector[index];
                    default: load_data = '0;
                endcase
            end
            @(negedge clk);
            load_we = 1'b0;
        end
    endtask

    initial begin
        $readmemh("vectors/top/image.hex", image_vector);
        $readmemh("vectors/top/c1_wgt.hex", c1_wgt_vector);
        $readmemh("vectors/top/c1_bias.hex", c1_bias_vector);
        $readmemh("vectors/top/c3_wgt.hex", c3_wgt_vector);
        $readmemh("vectors/top/c3_bias.hex", c3_bias_vector);
        $readmemh("vectors/top/c5_wgt.hex", c5_wgt_vector);
        $readmemh("vectors/top/c5_bias.hex", c5_bias_vector);
        $readmemh("vectors/top/f6_wgt.hex", f6_wgt_vector);
        $readmemh("vectors/top/f6_bias.hex", f6_bias_vector);
        $readmemh("vectors/top/cls_wgt.hex", cls_wgt_vector);
        $readmemh("vectors/top/cls_bias.hex", cls_bias_vector);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        load_we = 1'b0;
        rom_sel = '0;
        load_addr = '0;
        load_data = '0;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        // ROM select values (0..10) mirror lenet5_top.sv's ROM_SEL_* order.
        load_narrow(0, 32*32);   // image
        load_narrow(1, 6*1*25);  // c1 weights
        load_wide(2, 6);         // c1 bias
        load_narrow(3, 16*6*25); // c3 weights
        load_wide(4, 16);        // c3 bias
        load_narrow(5, 120*16*25); // c5 weights
        load_wide(6, 120);         // c5 bias
        load_narrow(7, 84*120); // f6 weights
        load_wide(8, 84);       // f6 bias
        load_narrow(9, 10*84);  // classifier weights
        load_wide(10, 10);      // classifier bias

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        wait (busy === 1'b1);
        wait (done === 1'b1);
        @(negedge clk);

        if (class_val !== `TV_TOP_EXPECTED_CLASS) begin
            $fatal(
                1,
                "Predicted class mismatch: got %0d, expected %0d",
                class_val,
                `TV_TOP_EXPECTED_CLASS
            );
        end
        $display(
            "PASS tb_lenet5_top: predicted class %0d matched golden/deploy.py end to end",
            class_val
        );
        $finish;
    end

    initial begin
        repeat (`TV_TOP_WATCHDOG_CYCLES) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
