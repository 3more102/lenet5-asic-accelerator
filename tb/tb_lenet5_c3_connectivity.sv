`timescale 1ns/1ps

module tb_lenet5_c3_connectivity;
    logic [3:0] output_map;
    logic [2:0] input_map;
    logic connected;
    logic [5:0] expected_mask;
    integer out_idx;
    integer in_idx;
    integer connection_count;

    lenet5_c3_connectivity dut (
        .output_map_i(output_map),
        .input_map_i(input_map),
        .connected_o(connected)
    );

    function automatic [5:0] mask_for_output(input integer index);
        begin
            case (index)
                0: mask_for_output=6'b000111; 1: mask_for_output=6'b001110;
                2: mask_for_output=6'b011100; 3: mask_for_output=6'b111000;
                4: mask_for_output=6'b110001; 5: mask_for_output=6'b100011;
                6: mask_for_output=6'b001111; 7: mask_for_output=6'b011110;
                8: mask_for_output=6'b111100; 9: mask_for_output=6'b111001;
                10: mask_for_output=6'b110011; 11: mask_for_output=6'b100111;
                12: mask_for_output=6'b011011; 13: mask_for_output=6'b110110;
                14: mask_for_output=6'b101101; 15: mask_for_output=6'b111111;
                default: mask_for_output=6'b000000;
            endcase
        end
    endfunction

    initial begin
        connection_count = 0;
        for (out_idx = 0; out_idx < 16; out_idx = out_idx + 1) begin
            expected_mask = mask_for_output(out_idx);
            for (in_idx = 0; in_idx < 6; in_idx = in_idx + 1) begin
                output_map = out_idx;
                input_map = in_idx;
                #1;
                if (connected !== expected_mask[in_idx]) begin
                    $fatal(
                        1,
                        "C3 mismatch: output=%0d input=%0d got=%0b expected=%0b",
                        out_idx, in_idx, connected, expected_mask[in_idx]
                    );
                end
                if (connected) connection_count = connection_count + 1;
            end
        end
        if (connection_count != 60) begin
            $fatal(1, "C3 connection count is %0d, expected 60", connection_count);
        end
        $display("PASS tb_lenet5_c3_connectivity: exact 60-connection table");
        $finish;
    end
endmodule

