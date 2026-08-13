`timescale 1ns/1ps

// S2/S4 pooling-layer input dimensions, sized for avg_pool2x2_stream.
module lenet5_pool_config (
    input  logic       layer_i, // 0=S2, 1=S4
    output logic [7:0] in_w_o,
    output logic [7:0] in_h_o,
    output logic [7:0] in_ch_o,
    output logic       valid_o
);
    always_comb begin
        in_w_o  = '0;
        in_h_o  = '0;
        in_ch_o = '0;
        valid_o = 1'b1;

        case (layer_i)
            1'b0: begin // S2: 6x28x28 -> 6x14x14
                in_w_o  = 8'd28;
                in_h_o  = 8'd28;
                in_ch_o = 8'd6;
            end
            1'b1: begin // S4: 16x10x10 -> 16x5x5
                in_w_o  = 8'd10;
                in_h_o  = 8'd10;
                in_ch_o = 8'd16;
            end
            default: valid_o = 1'b0;
        endcase
    end
endmodule
