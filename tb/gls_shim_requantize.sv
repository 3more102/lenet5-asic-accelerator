`timescale 1ns/1ps

// Gate-level-simulation shim for `requantize`.
//
// `scripts/run_gls.sh` re-runs the existing testbenches against the
// sky130hd-mapped netlist instead of the RTL. A gate-level netlist has no
// parameters -- synthesis resolves them -- but every testbench and every
// wrapper instantiates these blocks *with* parameter overrides. This shim
// closes that gap: it presents the RTL's parameterized interface, forwards
// the ports to the netlist, and fails loudly if a caller ever asks for a
// configuration the netlist was not synthesized at.
//
// The check is the point. Without it, a testbench that started overriding
// ACC_WIDTH would keep passing here while silently exercising a 32-bit
// netlist through a differently-shaped port list.
//
// This file is compiled only by `make gls`; nothing in `make regression`
// sees it, so the RTL simulations still bind straight to rtl/requantize.sv.
module requantize #(
    parameter integer ACC_WIDTH = 32,
    parameter integer OUT_WIDTH = 8
) (
    input  logic signed [ACC_WIDTH-1:0] acc_i,
    input  logic        [5:0]           shift_i,
    input  logic                        relu_en_i,
    output logic signed [OUT_WIDTH-1:0] data_o
);
    initial begin
        if (ACC_WIDTH != 32 || OUT_WIDTH != 8) begin
            $display("FAIL gls_shim_requantize: netlist is ACC_WIDTH=32 OUT_WIDTH=8, caller asked for %0d/%0d",
                     ACC_WIDTH, OUT_WIDTH);
            $fatal(1);
        end
    end

    requantize_gls u_netlist (
        .acc_i     (acc_i),
        .shift_i   (shift_i),
        .relu_en_i (relu_en_i),
        .data_o    (data_o)
    );
endmodule
