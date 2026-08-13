`timescale 1ns/1ps

// Five signed multipliers evaluate one complete row of a 5x5 kernel.
// Packing order is lane 0 in bits [DATA_WIDTH-1:0], then lane 1, etc.
module conv5x5_row_mac #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32
) (
    input  logic signed [(5*DATA_WIDTH)-1:0] act_row_i,
    input  logic signed [(5*DATA_WIDTH)-1:0] wgt_row_i,
    output logic signed [ACC_WIDTH-1:0]       row_sum_o
);
    localparam integer PROD_WIDTH = 2 * DATA_WIDTH;

    logic signed [DATA_WIDTH-1:0] act0, act1, act2, act3, act4;
    logic signed [DATA_WIDTH-1:0] wgt0, wgt1, wgt2, wgt3, wgt4;
    logic signed [PROD_WIDTH-1:0] prod0, prod1, prod2, prod3, prod4;
    logic signed [ACC_WIDTH-1:0] ext0, ext1, ext2, ext3, ext4;

    // Signals are written explicitly so synthesis tools do not interpret a
    // five-entry unpacked array as a memory.
    always_comb begin
        act0 = $signed(act_row_i[(0*DATA_WIDTH) +: DATA_WIDTH]);
        act1 = $signed(act_row_i[(1*DATA_WIDTH) +: DATA_WIDTH]);
        act2 = $signed(act_row_i[(2*DATA_WIDTH) +: DATA_WIDTH]);
        act3 = $signed(act_row_i[(3*DATA_WIDTH) +: DATA_WIDTH]);
        act4 = $signed(act_row_i[(4*DATA_WIDTH) +: DATA_WIDTH]);

        wgt0 = $signed(wgt_row_i[(0*DATA_WIDTH) +: DATA_WIDTH]);
        wgt1 = $signed(wgt_row_i[(1*DATA_WIDTH) +: DATA_WIDTH]);
        wgt2 = $signed(wgt_row_i[(2*DATA_WIDTH) +: DATA_WIDTH]);
        wgt3 = $signed(wgt_row_i[(3*DATA_WIDTH) +: DATA_WIDTH]);
        wgt4 = $signed(wgt_row_i[(4*DATA_WIDTH) +: DATA_WIDTH]);

        prod0 = act0 * wgt0;
        prod1 = act1 * wgt1;
        prod2 = act2 * wgt2;
        prod3 = act3 * wgt3;
        prod4 = act4 * wgt4;

        ext0 = {{(ACC_WIDTH-PROD_WIDTH){prod0[PROD_WIDTH-1]}}, prod0};
        ext1 = {{(ACC_WIDTH-PROD_WIDTH){prod1[PROD_WIDTH-1]}}, prod1};
        ext2 = {{(ACC_WIDTH-PROD_WIDTH){prod2[PROD_WIDTH-1]}}, prod2};
        ext3 = {{(ACC_WIDTH-PROD_WIDTH){prod3[PROD_WIDTH-1]}}, prod3};
        ext4 = {{(ACC_WIDTH-PROD_WIDTH){prod4[PROD_WIDTH-1]}}, prod4};

        // The explicit two-level tree avoids a narrow, left-associated sum.
        row_sum_o = (ext0 + ext1) + (ext2 + ext3) + ext4;
    end

`ifndef SYNTHESIS
    initial begin
        if (ACC_WIDTH < (PROD_WIDTH + 3)) begin
            $error("ACC_WIDTH is too small for five products");
        end
    end
`endif
endmodule
