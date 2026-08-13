`timescale 1ns/1ps

// F6/classifier dense-layer dimensions, sized for dense_engine /
// classifier_argmax.
module lenet5_dense_config (
    input  logic       layer_i, // 0=F6, 1=classifier
    output logic [7:0] in_len_o,
    output logic [7:0] out_len_o,
    output logic       valid_o
);
    always_comb begin
        in_len_o  = '0;
        out_len_o = '0;
        valid_o   = 1'b1;

        case (layer_i)
            1'b0: begin // F6: 120 -> 84
                in_len_o  = 8'd120;
                out_len_o = 8'd84;
            end
            1'b1: begin // classifier: 84 -> 10
                in_len_o  = 8'd84;
                out_len_o = 8'd10;
            end
            default: valid_o = 1'b0;
        endcase
    end
endmodule
