`timescale 1ns/1ps

// LANES signed multipliers evaluate one partial dot product of a dense
// layer. Packing order is lane 0 in bits [DATA_WIDTH-1:0], then lane 1,
// etc., generalizing conv5x5_row_mac's fixed 5-lane pattern to a
// parameterizable lane count for dense_engine's fully-connected datapath.
// Products are combined with a pairwise reduction tree built at
// elaboration time via generate/genvar (a procedural while-loop version
// simulates correctly but is not synthesizable -- Yosys requires
// procedural for-loop bounds to be constant). An unpaired lane at any
// level carries forward unchanged, so non-power-of-two LANES values are
// handled without special-casing.
module dense_row_mac #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH  = 32,
    parameter integer LANES      = 8
) (
    input  logic signed [(LANES*DATA_WIDTH)-1:0] act_lane_i,
    input  logic signed [(LANES*DATA_WIDTH)-1:0] wgt_lane_i,
    output logic signed [ACC_WIDTH-1:0]          lane_sum_o
);
    localparam integer PROD_WIDTH  = 2 * DATA_WIDTH;
    localparam integer TREE_LEVELS = (LANES <= 1) ? 1 : $clog2(LANES);

    // level[0][0:LANES-1] holds the LANES sign-extended leaf products;
    // level[l+1] holds ceil(count(level[l])/2) partial sums, until
    // level[TREE_LEVELS][0] holds the total. count(level[l]) =
    // ceil(LANES/2^l), inlined below as (LANES+(2^l-1))>>l since generate
    // bounds must be elaboration-time constants, not function calls.
    logic signed [ACC_WIDTH-1:0] level [0:TREE_LEVELS][0:LANES-1];

    genvar lane_idx, lvl, node_idx;
    generate
        for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin : g_leaf
            logic signed [DATA_WIDTH-1:0] act_l, wgt_l;
            logic signed [PROD_WIDTH-1:0] prod_l;
            assign act_l  = $signed(act_lane_i[(lane_idx*DATA_WIDTH) +: DATA_WIDTH]);
            assign wgt_l  = $signed(wgt_lane_i[(lane_idx*DATA_WIDTH) +: DATA_WIDTH]);
            assign prod_l = act_l * wgt_l;
            assign level[0][lane_idx] =
                {{(ACC_WIDTH-PROD_WIDTH){prod_l[PROD_WIDTH-1]}}, prod_l};
        end

        for (lvl = 0; lvl < TREE_LEVELS; lvl = lvl + 1) begin : g_level
            for (node_idx = 0;
                 node_idx < ((LANES + ((1 << (lvl+1)) - 1)) >> (lvl+1));
                 node_idx = node_idx + 1) begin : g_node
                if ((2*node_idx + 1) < ((LANES + ((1 << lvl) - 1)) >> lvl)) begin : g_pair
                    assign level[lvl+1][node_idx] =
                        level[lvl][2*node_idx] + level[lvl][2*node_idx+1];
                end else begin : g_carry
                    assign level[lvl+1][node_idx] = level[lvl][2*node_idx];
                end
            end
        end
    endgenerate

    assign lane_sum_o = level[TREE_LEVELS][0];

`ifndef SYNTHESIS
    initial begin
        if (ACC_WIDTH < (PROD_WIDTH + $clog2((LANES < 2) ? 2 : LANES) + 1)) begin
            $error("ACC_WIDTH is too small for LANES products");
        end
    end
`endif
endmodule
