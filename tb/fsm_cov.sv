`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// Portable FSM state / transition coverage collector and illegal-transition
// checker.
//
// Why this is not a SystemVerilog covergroup
// ------------------------------------------
// Icarus Verilog -- which runs this project's CI and is the only simulator a
// reader can install with apt-get -- does not implement covergroups. A
// coverage tier expressed in covergroups would therefore run on exactly one
// commercial simulator, and would quietly stop running the moment CI is the
// only thing anyone executes. Coverage that silently stops collecting is
// worse than no coverage, because the absence of a failure still reads as a
// pass.
//
// Everything below is plain always/integer-array Verilog, so Icarus,
// ModelSim, and Questa all produce the same numbers, and CI enforces them on
// every push.
//
// What it enforces at report time
// -------------------------------
//   1. every declared state was entered at least once;
//   2. every declared-legal transition was exercised at least once
//      (an unexercised legal edge is a coverage hole in the stimulus);
//   3. no transition outside the declared-legal set ever occurred
//      (an illegal edge is a design bug, and is reported the cycle it
//       happens with both state names and the simulation time).
//
// Usage: instantiate, then in an initial block declare the state names and
// the legal edge set before the DUT leaves reset, and call report_and_check
// once the run is over.
// ---------------------------------------------------------------------------
module fsm_cov #(
    parameter integer NSTATES     = 2,
    parameter integer STATE_WIDTH = 8
) (
    input wire                   clk_i,
    input wire                   rst_ni,
    input wire [STATE_WIDTH-1:0] state_i
);
    localparam integer NPAIRS = NSTATES * NSTATES;

    // Name width is fixed rather than a parameter on purpose. Verilog
    // truncates a string literal silently when it is passed into a narrower
    // reg argument, so a caller who picks NAME_CHARS one character too small
    // gets a report that quietly lies about which state it is describing --
    // exactly the failure this collector exists to prevent. 24 characters
    // covers any realistic state name; longer names lose their leading
    // characters in the printed report only, and never affect pass/fail.
    localparam integer NAME_CHARS = 24;

    // cycles[s]      : clock edges observed with the FSM sitting in state s.
    // taken[f*N+t]   : times the edge f->t was observed (self-edges included).
    // allowed[f*N+t] : 1 if the design is permitted to make that edge.
    integer cycles  [0:NSTATES-1];
    integer taken   [0:NPAIRS-1];
    integer allowed [0:NPAIRS-1];

    reg [NAME_CHARS*8-1:0] sname [0:NSTATES-1];

    integer prev_state;
    integer illegal_seen;
    integer i;

    initial begin
        for (i = 0; i < NSTATES; i = i + 1) begin
            cycles[i] = 0;
            // Default name, overwritten by set_name. Keeps the report
            // readable even for a partially-named FSM.
            sname[i]  = "?";
        end
        for (i = 0; i < NPAIRS; i = i + 1) begin
            taken[i]   = 0;
            allowed[i] = 0;
        end
        prev_state   = -1;
        illegal_seen = 0;
    end

    task set_name(input integer idx, input reg [NAME_CHARS*8-1:0] nm);
        begin
            sname[idx] = nm;
        end
    endtask

    // Declare one legal edge. Self-edges (from == to) mean "this state is
    // allowed to stall"; omitting a self-edge asserts the state lasts
    // exactly one cycle, which is what makes single-cycle start pulses
    // checkable rather than merely intended.
    task allow(input integer from_state, input integer to_state);
        begin
            allowed[from_state * NSTATES + to_state] = 1;
        end
    endtask

    // Convenience: declare a linear stage chain 0->1->2->...->N-1->0.
    task allow_chain;
        integer s;
        begin
            for (s = 0; s < NSTATES; s = s + 1) begin
                allow(s, (s + 1) % NSTATES);
            end
        end
    endtask

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            // Reset breaks the edge sequence: the jump back to the reset
            // state is not a transition the FSM logic chose to make, so it
            // must not be scored as one.
            prev_state = -1;
        end else begin
            if (state_i >= NSTATES) begin
                $fatal(1,
                    "fsm_cov: state %0d is outside the declared 0..%0d range at time %0t",
                    state_i, NSTATES - 1, $time);
            end

            cycles[state_i] = cycles[state_i] + 1;

            if (prev_state >= 0) begin
                taken[prev_state * NSTATES + state_i] =
                    taken[prev_state * NSTATES + state_i] + 1;

                if (allowed[prev_state * NSTATES + state_i] === 0) begin
                    illegal_seen = illegal_seen + 1;
                    $display("  ILLEGAL TRANSITION %0s -> %0s at time %0t",
                             sname[prev_state], sname[state_i], $time);
                end
            end

            prev_state = state_i;
        end
    end

    // Print the coverage table and fail on any unvisited state, any
    // unexercised legal edge, or any illegal edge seen during the run.
    task report_and_check(input reg [NAME_CHARS*8-1:0] label);
        integer s, t, idx;
        integer legal_total, legal_hit, unvisited;
        begin
            legal_total = 0;
            legal_hit   = 0;
            unvisited   = 0;

            $display("  %0s FSM coverage:", label);
            for (s = 0; s < NSTATES; s = s + 1) begin
                if (cycles[s] == 0) begin
                    unvisited = unvisited + 1;
                    $display("    UNVISITED STATE %0s", sname[s]);
                end
            end

            for (s = 0; s < NSTATES; s = s + 1) begin
                for (t = 0; t < NSTATES; t = t + 1) begin
                    idx = s * NSTATES + t;
                    if (allowed[idx] !== 0) begin
                        legal_total = legal_total + 1;
                        if (taken[idx] > 0) begin
                            legal_hit = legal_hit + 1;
                        end else begin
                            $display("    UNEXERCISED EDGE %0s -> %0s",
                                     sname[s], sname[t]);
                        end
                    end
                end
            end

            $display("    states %0d/%0d entered, legal edges %0d/%0d exercised",
                     NSTATES - unvisited, NSTATES, legal_hit, legal_total);

            if (unvisited != 0) begin
                $fatal(1, "%0s: %0d state(s) never entered", label, unvisited);
            end
            if (legal_hit != legal_total) begin
                $fatal(1, "%0s: %0d legal transition(s) never exercised",
                       label, legal_total - legal_hit);
            end
            if (illegal_seen != 0) begin
                $fatal(1, "%0s: %0d illegal transition(s) observed",
                       label, illegal_seen);
            end
        end
    endtask
endmodule
