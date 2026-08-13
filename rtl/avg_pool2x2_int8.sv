`timescale 1ns/1ps

// Hardware-friendly 2x2 average pooling for signed int8 activations.
// Division by four is rounded to nearest, ties away from zero.
module avg_pool2x2_int8 #(
    parameter integer DATA_WIDTH = 8
) (
    input  logic signed [(4*DATA_WIDTH)-1:0] samples_i,
    output logic signed [DATA_WIDTH-1:0]     average_o
);
    localparam integer SUM_WIDTH = DATA_WIDTH + 2;

    logic signed [DATA_WIDTH-1:0] sample [0:3];
    logic signed [SUM_WIDTH-1:0]  sample_ext [0:3];
    logic signed [SUM_WIDTH-1:0]  sum;
    logic signed [SUM_WIDTH-1:0]  magnitude;
    logic signed [SUM_WIDTH-1:0]  rounded;

    integer idx;
    always_comb begin
        for (idx = 0; idx < 4; idx = idx + 1) begin
            sample[idx] = $signed(samples_i[(idx*DATA_WIDTH) +: DATA_WIDTH]);
            sample_ext[idx] = {{2{sample[idx][DATA_WIDTH-1]}}, sample[idx]};
        end

        sum = sample_ext[0] + sample_ext[1]
            + sample_ext[2] + sample_ext[3];

        magnitude = '0;
        if (sum >= 0) begin
            rounded = (sum + 2) >>> 2;
        end else begin
            magnitude = -sum;
            rounded   = -((magnitude + 2) >>> 2);
        end
        average_o = rounded[DATA_WIDTH-1:0];
    end
endmodule

