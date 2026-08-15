`timescale 1ns/1ps

// Gate-level-simulation shim for `avg_pool2x2_int8`. See
// tb/gls_shim_requantize.sv for why these shims exist.
//
// This block has no testbench of its own -- it is a combinational primitive
// reached through `avg_pool2x2_stream`, whose activation store is still a
// behavioral array and therefore not synthesizable to sky130 cells. So
// `make gls` runs tb_avg_pool2x2_stream as a *mixed* RTL/gate simulation:
// the streaming controller stays RTL, the pooling arithmetic underneath it
// is the mapped netlist.
module avg_pool2x2_int8 #(
    parameter integer DATA_WIDTH = 8
) (
    input  logic signed [(4*DATA_WIDTH)-1:0] samples_i,
    output logic signed [DATA_WIDTH-1:0]     average_o
);
    initial begin
        if (DATA_WIDTH != 8) begin
            $display("FAIL gls_shim_avg_pool2x2_int8: netlist is DATA_WIDTH=8, caller asked for %0d",
                     DATA_WIDTH);
            $fatal(1);
        end
    end

    avg_pool2x2_int8_gls u_netlist (
        .samples_i (samples_i),
        .average_o (average_o)
    );
endmodule
