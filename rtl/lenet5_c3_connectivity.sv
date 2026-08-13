`timescale 1ns/1ps

// Exact sparse S2-to-C3 connection table from LeCun et al. (1998), Table I.
module lenet5_c3_connectivity (
    input  logic [3:0] output_map_i,
    input  logic [2:0] input_map_i,
    output logic       connected_o
);
    logic [5:0] input_mask;

    always_comb begin
        case (output_map_i)
            4'd0:  input_mask = 6'b000111; // 0,1,2
            4'd1:  input_mask = 6'b001110; // 1,2,3
            4'd2:  input_mask = 6'b011100; // 2,3,4
            4'd3:  input_mask = 6'b111000; // 3,4,5
            4'd4:  input_mask = 6'b110001; // 4,5,0
            4'd5:  input_mask = 6'b100011; // 5,0,1
            4'd6:  input_mask = 6'b001111; // 0,1,2,3
            4'd7:  input_mask = 6'b011110; // 1,2,3,4
            4'd8:  input_mask = 6'b111100; // 2,3,4,5
            4'd9:  input_mask = 6'b111001; // 3,4,5,0
            4'd10: input_mask = 6'b110011; // 4,5,0,1
            4'd11: input_mask = 6'b100111; // 5,0,1,2
            4'd12: input_mask = 6'b011011; // 0,1,3,4
            4'd13: input_mask = 6'b110110; // 1,2,4,5
            4'd14: input_mask = 6'b101101; // 0,2,3,5
            4'd15: input_mask = 6'b111111; // all maps
            default: input_mask = 6'b000000;
        endcase

        connected_o = (input_map_i < 6) ? input_mask[input_map_i] : 1'b0;
    end
endmodule

