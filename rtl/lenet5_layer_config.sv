`timescale 1ns/1ps

// Canonical convolution-layer dimensions from LeNet-5.
module lenet5_layer_config (
    input  logic [1:0] layer_i, // 0=C1, 1=C3, 2=C5
    output logic [7:0] in_w_o,
    output logic [7:0] in_h_o,
    output logic [7:0] in_ch_o,
    output logic [7:0] out_ch_o,
    output logic       valid_o
);
    always_comb begin
        in_w_o   = '0;
        in_h_o   = '0;
        in_ch_o  = '0;
        out_ch_o = '0;
        valid_o  = 1'b1;

        case (layer_i)
            2'd0: begin // C1: 1x32x32 -> 6x28x28
                in_w_o   = 8'd32;
                in_h_o   = 8'd32;
                in_ch_o  = 8'd1;
                out_ch_o = 8'd6;
            end
            2'd1: begin // C3: 6x14x14 -> 16x10x10
                in_w_o   = 8'd14;
                in_h_o   = 8'd14;
                in_ch_o  = 8'd6;
                out_ch_o = 8'd16;
            end
            2'd2: begin // C5: 16x5x5 -> 120x1x1
                in_w_o   = 8'd5;
                in_h_o   = 8'd5;
                in_ch_o  = 8'd16;
                out_ch_o = 8'd120;
            end
            default: valid_o = 1'b0;
        endcase
    end
endmodule

