`timescale 1ns/1ps

// Miter used by scripts/prove_pe_output_hold.sh. Analysis artifact, not part of
// any regression: it exists to settle one question about one mutation, and the
// answer is written up in docs/VERIFICATION_PLAN.md.
//
// The mutation under study removes the `rq_valid_q` guard on conv5x5_pe's
// output register, so `out_data_o` is rewritten on every cycle the requantize
// stage is ready rather than only when a result is pending. It survives both
// PE testbenches, and `equiv_make` reports `out_data_o` as unproven -- correctly,
// because the two designs really do drive different values on that wire.
//
// The question that matters is narrower than signal equality: can a consumer
// that honours valid/ready ever see the difference? `bad_o` is the negation of
// that property. A handshake difference counts always; a payload difference
// counts only while the producer is announcing a beat, because that is the only
// time the payload means anything.
module pe_output_hold_miter (
    input  logic                clk_i,
    input  logic                rst_ni,
    input  logic                in_valid_i,
    input  logic                first_i,
    input  logic                last_i,
    input  logic signed [39:0]  act_row_i,
    input  logic signed [39:0]  wgt_row_i,
    input  logic signed [31:0]  bias_i,
    input  logic        [5:0]   shift_i,
    input  logic                relu_en_i,
    input  logic                out_ready_i,
    output logic                bad_o
);
    logic gold_in_ready, gold_out_valid;
    logic gate_in_ready, gate_out_valid;
    logic signed [7:0] gold_out_data, gate_out_data;

    conv5x5_pe u_gold (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .in_valid_i(in_valid_i), .in_ready_o(gold_in_ready),
        .first_i(first_i), .last_i(last_i),
        .act_row_i(act_row_i), .wgt_row_i(wgt_row_i),
        .bias_i(bias_i), .shift_i(shift_i), .relu_en_i(relu_en_i),
        .out_valid_o(gold_out_valid), .out_ready_i(out_ready_i),
        .out_data_o(gold_out_data)
    );

    conv5x5_pe_mut u_gate (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .in_valid_i(in_valid_i), .in_ready_o(gate_in_ready),
        .first_i(first_i), .last_i(last_i),
        .act_row_i(act_row_i), .wgt_row_i(wgt_row_i),
        .bias_i(bias_i), .shift_i(shift_i), .relu_en_i(relu_en_i),
        .out_valid_o(gate_out_valid), .out_ready_i(out_ready_i),
        .out_data_o(gate_out_data)
    );

    assign bad_o = (gold_in_ready  !== gate_in_ready)
                 | (gold_out_valid !== gate_out_valid)
                 | (gold_out_valid & (gold_out_data !== gate_out_data));
endmodule
