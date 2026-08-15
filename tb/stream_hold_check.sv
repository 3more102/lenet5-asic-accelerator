`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// Continuous valid/ready stream protocol checker.
//
// Every engine in this project produces its results on a valid/ready stream,
// which obliges the producer to honour two rules that no testbench here
// checked before this file existed:
//
//   1. valid, once raised, is never withdrawn until the consumer accepts the
//      beat. A producer that drops valid because the consumer was slow has
//      silently discarded an output;
//   2. the payload is held stable for as long as valid is high and ready is
//      low. A producer that advances its data mid-stall hands the consumer a
//      different beat than the one it announced.
//
// Both violations are structurally invisible to a testbench that keeps ready
// tied high -- the stall state is simply never entered -- and equally
// invisible to one that keeps ready *periodic*, if the period happens to
// avoid the offending beat. That is why this is a continuous checker
// instantiated alongside the DUT rather than a check written into any one
// stimulus sequence: it costs nothing on runs that never stall, and fires on
// the exact cycle of a violation on runs that do.
//
// The payload is taken as one packed vector so a single checker covers
// streams with different side-band widths (conv/pool carry channel/y/x,
// dense carries index/acc).
// ---------------------------------------------------------------------------
module stream_hold_check #(
    parameter integer PAYLOAD_WIDTH = 8
) (
    input wire                     clk_i,
    input wire                     rst_ni,
    input wire                     valid_i,
    input wire                     ready_i,
    input wire [PAYLOAD_WIDTH-1:0] payload_i
);
    // Fixed rather than parameterised, for the same reason as fsm_cov's
    // NAME_CHARS: Verilog truncates an over-long string literal silently, and
    // a checker that misreports which stream it is policing is worse than no
    // checker at all.
    localparam integer NAME_CHARS = 24;

    reg [NAME_CHARS*8-1:0] label;

    reg                     prev_valid;
    reg                     prev_ready;
    reg [PAYLOAD_WIDTH-1:0] prev_payload;
    reg                     armed;

    // Free-running totals. Deliberately never cleared here: the testbench
    // snapshots them around each run and works in deltas, so nothing outside
    // this module ever writes them and the two-process assignment hazard
    // cannot arise.
    integer beats;
    integer stall_cycles;

    initial begin
        label        = "stream";
        prev_valid   = 1'b0;
        prev_ready   = 1'b0;
        prev_payload = {PAYLOAD_WIDTH{1'b0}};
        armed        = 1'b0;
        beats        = 0;
        stall_cycles = 0;
    end

    task set_label(input reg [NAME_CHARS*8-1:0] nm);
        begin
            label = nm;
        end
    endtask

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            // Reset is entitled to drop valid mid-beat -- that is an abort,
            // not a protocol violation -- so the cycle-to-cycle comparison is
            // disarmed here and re-armed on the first clock after release.
            armed        <= 1'b0;
            prev_valid   <= 1'b0;
            prev_ready   <= 1'b0;
        end else begin
            if (armed && prev_valid && !prev_ready) begin
                if (valid_i !== 1'b1) begin
                    $fatal(1,
                        "%0s: valid was withdrawn at time %0t before the consumer accepted the beat -- an output was discarded",
                        label, $time);
                end
                if (payload_i !== prev_payload) begin
                    $fatal(1,
                        "%0s: payload changed while stalled at time %0t (%0h -> %0h) -- the consumer would latch a different beat than the one announced",
                        label, $time, prev_payload, payload_i);
                end
                stall_cycles <= stall_cycles + 1;
            end

            if (valid_i && ready_i) begin
                beats <= beats + 1;
            end

            prev_valid   <= valid_i;
            prev_ready   <= ready_i;
            prev_payload <= payload_i;
            armed        <= 1'b1;
        end
    end
endmodule
