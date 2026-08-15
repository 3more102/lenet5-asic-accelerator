`timescale 1ns/1ps
`include "vectors/config.svh"

// End-to-end check of lenet5_top against golden/deploy.py:deploy_forward_int8
// on the full canonical 32x32x1 shape (the only shape the C1->S4->C5 chain
// collapses correctly at). Unlike the per-module testbenches, the
// scoreboard here is a single terminal assertion of the predicted class,
// since lenet5_top exposes only the final decision, not per-stage taps.
module tb_lenet5_top;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;

    // Mirrors lenet5_top's own ROM_ADDR_WIDTH derivation so this testbench
    // cannot silently drift out of sync with the DUT.
    localparam integer ROM_ADDR_WIDTH = $clog2(120*16*25);

    logic clk;
    logic rst_n;
    logic start;
    logic busy;
    logic done;
    logic [3:0] class_val;

    logic load_we;
    logic [3:0] rom_sel;
    logic [ROM_ADDR_WIDTH-1:0] load_addr;
    logic signed [ACC_WIDTH-1:0] load_data;

    lenet5_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH)
    ) dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .start_i(start),
        .busy_o(busy),
        .done_o(done),
        .class_o(class_val),
        .load_we_i(load_we),
        .cfg_rom_sel_i(rom_sel),
        .load_addr_i(load_addr),
        .load_data_i(load_data)
    );

    // ------------------------------------------------------------------
    // Control-FSM coverage.
    //
    // Hung off this existing run rather than given its own testbench: the
    // end-to-end pass is already the only stimulus that walks the whole
    // C1->S2->C3->S4->C5->F6->classifier sequence, and duplicating it would
    // double a 2 ms simulation to observe the same states.
    //
    // The declared edge set is the full linear stage chain plus a stall
    // self-edge on exactly those states that are allowed to wait. The seven
    // S_KICK_* states are deliberately given no self-edge: each drives a
    // one-cycle start pulse into an engine, so a KICK lasting two cycles
    // would start that engine twice. Leaving the self-edge out turns that
    // from a comment into a checked property.
    // ------------------------------------------------------------------
    localparam integer NSTAGES = 20;

    fsm_cov #(
        .NSTATES    (NSTAGES),
        .STATE_WIDTH(5)
    ) u_top_fsm_cov (
        .clk_i  (clk),
        .rst_ni (rst_n),
        .state_i(dut.top_stage_q)
    );

    // Start pulses must be exactly one cycle wide and must never be
    // asserted while their engine is still busy.
    integer conv_start_cycles, pool_start_cycles;
    integer f6_start_cycles,   cls_start_cycles;

    always @(posedge clk) begin
        if (rst_n) begin
            if (dut.conv_start) begin
                conv_start_cycles = conv_start_cycles + 1;
                if (dut.conv_busy) begin
                    $fatal(1, "conv_start asserted while conv2d_engine busy at %0t", $time);
                end
            end
            if (dut.pool_start) begin
                pool_start_cycles = pool_start_cycles + 1;
                if (dut.pool_busy) begin
                    $fatal(1, "pool_start asserted while avg_pool2x2_stream busy at %0t", $time);
                end
            end
            if (dut.f6_start) begin
                f6_start_cycles = f6_start_cycles + 1;
                if (dut.f6_busy) begin
                    $fatal(1, "f6_start asserted while dense_engine busy at %0t", $time);
                end
            end
            if (dut.cls_start) begin
                cls_start_cycles = cls_start_cycles + 1;
            end
        end
    end

    // Free-running cycle counter so the per-inference cost is reported
    // directly rather than left to the reader to divide $finish's timestamp
    // by the clock period. That indirect reading stopped working once this
    // testbench ran two images, and the headline number in README.md and
    // docs/ARCHITECTURE.md is a per-inference figure, not a whole-simulation
    // one.
    integer cycle_count;
    integer run1_start_cycle, run1_done_cycle;
    integer run2_start_cycle, run2_done_cycle;

    // Steady-state cost of one inference with the weight ROMs already
    // resident, measured start_i -> done_o. Pinned so engine scheduling
    // cannot silently move the performance number this project publishes.
    //
    // README.md and docs/ARCHITECTURE.md quote 209,290 cycles; that is the
    // cold path -- the host writing all 62,730 ROM words, then one inference.
    // This is the steady-state figure underneath it, and it is the one that
    // matters for a device streaming images. If a deliberate optimisation
    // moves this, update this constant and both documents together.
    localparam integer EXPECTED_INFERENCE_CYCLES = 146544;

    always @(posedge clk) begin
        if (!rst_n) cycle_count <= 0;
        else        cycle_count <= cycle_count + 1;
    end

    logic signed [7:0]  image_vector   [0:32*32-1];
    logic signed [7:0]  c1_wgt_vector  [0:6*1*25-1];
    logic signed [31:0] c1_bias_vector [0:6-1];
    logic signed [7:0]  c3_wgt_vector  [0:16*6*25-1];
    logic signed [31:0] c3_bias_vector [0:16-1];
    logic signed [7:0]  c5_wgt_vector  [0:120*16*25-1];
    logic signed [31:0] c5_bias_vector [0:120-1];
    logic signed [7:0]  f6_wgt_vector  [0:84*120-1];
    logic signed [31:0] f6_bias_vector [0:84-1];
    logic signed [7:0]  cls_wgt_vector [0:10*84-1];
    logic signed [31:0] cls_bias_vector [0:10-1];

    always #5 clk = ~clk;

    task automatic load_narrow(input integer sel, input integer count);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                load_we   = 1'b1;
                rom_sel   = sel[3:0];
                load_addr = index[ROM_ADDR_WIDTH-1:0];
                case (sel)
                    0:  load_data = image_vector[index];
                    1:  load_data = c1_wgt_vector[index];
                    3:  load_data = c3_wgt_vector[index];
                    5:  load_data = c5_wgt_vector[index];
                    7:  load_data = f6_wgt_vector[index];
                    9:  load_data = cls_wgt_vector[index];
                    default: load_data = '0;
                endcase
            end
            @(negedge clk);
            load_we = 1'b0;
        end
    endtask

    task automatic load_wide(input integer sel, input integer count);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                load_we   = 1'b1;
                rom_sel   = sel[3:0];
                load_addr = index[ROM_ADDR_WIDTH-1:0];
                case (sel)
                    2:  load_data = c1_bias_vector[index];
                    4:  load_data = c3_bias_vector[index];
                    6:  load_data = c5_bias_vector[index];
                    8:  load_data = f6_bias_vector[index];
                    10: load_data = cls_bias_vector[index];
                    default: load_data = '0;
                endcase
            end
            @(negedge clk);
            load_we = 1'b0;
        end
    endtask

    initial begin
        $readmemh("vectors/top/image.hex", image_vector);
        $readmemh("vectors/top/c1_wgt.hex", c1_wgt_vector);
        $readmemh("vectors/top/c1_bias.hex", c1_bias_vector);
        $readmemh("vectors/top/c3_wgt.hex", c3_wgt_vector);
        $readmemh("vectors/top/c3_bias.hex", c3_bias_vector);
        $readmemh("vectors/top/c5_wgt.hex", c5_wgt_vector);
        $readmemh("vectors/top/c5_bias.hex", c5_bias_vector);
        $readmemh("vectors/top/f6_wgt.hex", f6_wgt_vector);
        $readmemh("vectors/top/f6_bias.hex", f6_bias_vector);
        $readmemh("vectors/top/cls_wgt.hex", cls_wgt_vector);
        $readmemh("vectors/top/cls_bias.hex", cls_bias_vector);

        // Stage names and the legal edge set, declared before reset is
        // released so the collector is armed for the very first cycle.
        // Indices mirror lenet5_top.sv's S_* localparams.
        u_top_fsm_cov.set_name(0,  "S_IDLE");
        u_top_fsm_cov.set_name(1,  "S_LOAD_C1");
        u_top_fsm_cov.set_name(2,  "S_KICK_C1");
        u_top_fsm_cov.set_name(3,  "S_RUN_C1");
        u_top_fsm_cov.set_name(4,  "S_KICK_S2");
        u_top_fsm_cov.set_name(5,  "S_RUN_S2");
        u_top_fsm_cov.set_name(6,  "S_LOAD_C3");
        u_top_fsm_cov.set_name(7,  "S_KICK_C3");
        u_top_fsm_cov.set_name(8,  "S_RUN_C3");
        u_top_fsm_cov.set_name(9,  "S_KICK_S4");
        u_top_fsm_cov.set_name(10, "S_RUN_S4");
        u_top_fsm_cov.set_name(11, "S_LOAD_C5");
        u_top_fsm_cov.set_name(12, "S_KICK_C5");
        u_top_fsm_cov.set_name(13, "S_RUN_C5");
        u_top_fsm_cov.set_name(14, "S_LOAD_F6");
        u_top_fsm_cov.set_name(15, "S_KICK_F6");
        u_top_fsm_cov.set_name(16, "S_RUN_F6");
        u_top_fsm_cov.set_name(17, "S_LOAD_CLS");
        u_top_fsm_cov.set_name(18, "S_KICK_CLS");
        u_top_fsm_cov.set_name(19, "S_RUN_CLS");

        // The stage sequence is a flat ring: each stage advances to the
        // next, and S_RUN_CLS wraps back to S_IDLE for the next image.
        u_top_fsm_cov.allow_chain();

        // Stall self-edges, allowed only where the stage genuinely waits:
        // S_IDLE waits for start_i, every S_LOAD_* counts through its ROM,
        // and every S_RUN_* waits for its engine's done. The S_KICK_*
        // stages are absent by design -- see the note at the instance.
        u_top_fsm_cov.allow(0,  0);   // S_IDLE     waits for start_i
        u_top_fsm_cov.allow(1,  1);   // S_LOAD_C1  counts ROM words
        u_top_fsm_cov.allow(3,  3);   // S_RUN_C1   waits for conv_done
        u_top_fsm_cov.allow(5,  5);   // S_RUN_S2   waits for pool_done
        u_top_fsm_cov.allow(6,  6);   // S_LOAD_C3
        u_top_fsm_cov.allow(8,  8);   // S_RUN_C3
        u_top_fsm_cov.allow(10, 10);  // S_RUN_S4
        u_top_fsm_cov.allow(11, 11);  // S_LOAD_C5
        u_top_fsm_cov.allow(13, 13);  // S_RUN_C5
        u_top_fsm_cov.allow(14, 14);  // S_LOAD_F6
        u_top_fsm_cov.allow(16, 16);  // S_RUN_F6
        u_top_fsm_cov.allow(17, 17);  // S_LOAD_CLS
        u_top_fsm_cov.allow(19, 19);  // S_RUN_CLS  waits for cls_valid

        conv_start_cycles = 0;
        pool_start_cycles = 0;
        f6_start_cycles   = 0;
        cls_start_cycles  = 0;

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        load_we = 1'b0;
        rom_sel = '0;
        load_addr = '0;
        load_data = '0;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        // ROM select values (0..10) mirror lenet5_top.sv's ROM_SEL_* order.
        load_narrow(0, 32*32);   // image
        load_narrow(1, 6*1*25);  // c1 weights
        load_wide(2, 6);         // c1 bias
        load_narrow(3, 16*6*25); // c3 weights
        load_wide(4, 16);        // c3 bias
        load_narrow(5, 120*16*25); // c5 weights
        load_wide(6, 120);         // c5 bias
        load_narrow(7, 84*120); // f6 weights
        load_wide(8, 84);       // f6 bias
        load_narrow(9, 10*84);  // classifier weights
        load_wide(10, 10);      // classifier bias

        @(negedge clk);
        start = 1'b1;
        run1_start_cycle = cycle_count;
        @(negedge clk);
        start = 1'b0;

        wait (busy === 1'b1);
        wait (done === 1'b1);
        run1_done_cycle = cycle_count;
        @(negedge clk);

        if (class_val !== `TV_TOP_EXPECTED_CLASS) begin
            $fatal(
                1,
                "Predicted class mismatch: got %0d, expected %0d",
                class_val,
                `TV_TOP_EXPECTED_CLASS
            );
        end
        // conv2d_engine is resource-shared across C1/C3/C5 and
        // avg_pool2x2_stream across S2/S4, so the expected pulse counts are
        // a direct statement of the sharing scheme.
        if (conv_start_cycles != 3) begin
            $fatal(1, "conv_start pulsed %0d times, expected 3 (C1, C3, C5)",
                   conv_start_cycles);
        end
        if (pool_start_cycles != 2) begin
            $fatal(1, "pool_start pulsed %0d times, expected 2 (S2, S4)",
                   pool_start_cycles);
        end
        if (f6_start_cycles != 1) begin
            $fatal(1, "f6_start pulsed %0d times, expected 1 (F6)", f6_start_cycles);
        end
        if (cls_start_cycles != 1) begin
            $fatal(1, "cls_start pulsed %0d times, expected 1", cls_start_cycles);
        end

        $display(
            "PASS tb_lenet5_top: predicted class %0d matched golden/deploy.py end to end",
            class_val
        );

        // ------------------------------------------------------------------
        // Second inference: no reset, no weight reload.
        //
        // A real inference accelerator processes a stream of images, so every
        // counter, accumulator and pipeline register inside the five engines
        // has to return to its idle value on its own after done_o. Nothing
        // above tests that -- the run has only ever been observed from a
        // freshly reset design. image_rom is written only during the load
        // phase and each engine owns its activation RAM, so re-pulsing start_i
        // re-runs the identical image and must reproduce the identical class.
        //
        // This is also the stimulus that exercises S_RUN_CLS -> S_IDLE. The
        // FSM coverage collector flagged that edge as the single unexercised
        // legal transition in the single-shot run, which is what prompted
        // this check.
        // ------------------------------------------------------------------
        wait (busy === 1'b0);
        @(negedge clk);
        start = 1'b1;
        run2_start_cycle = cycle_count;
        @(negedge clk);
        start = 1'b0;

        wait (busy === 1'b1);
        wait (done === 1'b1);
        run2_done_cycle = cycle_count;
        @(negedge clk);

        if (class_val !== `TV_TOP_EXPECTED_CLASS) begin
            $fatal(
                1,
                "Second inference predicted %0d, expected %0d: accelerator does not return to a clean state after done_o",
                class_val,
                `TV_TOP_EXPECTED_CLASS
            );
        end

        // Exactly double the single-run counts: the second pass must issue the
        // same kicks as the first, no more and no fewer. A stuck or skipped
        // stage that still happened to land on the right class would be caught
        // here rather than passing silently.
        if (conv_start_cycles != 6) begin
            $fatal(1, "conv_start pulsed %0d times over two runs, expected 6",
                   conv_start_cycles);
        end
        if (pool_start_cycles != 4) begin
            $fatal(1, "pool_start pulsed %0d times over two runs, expected 4",
                   pool_start_cycles);
        end
        if (f6_start_cycles != 2) begin
            $fatal(1, "f6_start pulsed %0d times over two runs, expected 2",
                   f6_start_cycles);
        end
        if (cls_start_cycles != 2) begin
            $fatal(1, "cls_start pulsed %0d times over two runs, expected 2",
                   cls_start_cycles);
        end

        $display(
            "PASS tb_lenet5_top: back-to-back inference reproduced class %0d with no reset and no weight reload",
            class_val
        );

        $display("  inference 1: %0d cycles start_i -> done_o",
                 run1_done_cycle - run1_start_cycle);
        $display("  inference 2: %0d cycles start_i -> done_o",
                 run2_done_cycle - run2_start_cycle);

        if ((run1_done_cycle - run1_start_cycle) != EXPECTED_INFERENCE_CYCLES) begin
            $fatal(1,
                "inference took %0d cycles, expected %0d -- if this change was deliberate, update EXPECTED_INFERENCE_CYCLES here and the published cycle counts in README.md and docs/ARCHITECTURE.md",
                run1_done_cycle - run1_start_cycle, EXPECTED_INFERENCE_CYCLES);
        end

        // Stated separately from the absolute check because it fails with a
        // far more useful diagnostic: a stale counter or uncleared pipeline
        // register shows up as a second run that costs a different number of
        // cycles than the first, whatever the absolute figure happens to be.
        if ((run2_done_cycle - run2_start_cycle)
            != (run1_done_cycle - run1_start_cycle)) begin
            $fatal(1,
                "inference 2 took %0d cycles against inference 1's %0d: the accelerator's second run is not cycle-identical, so some state did not return to idle",
                run2_done_cycle - run2_start_cycle,
                run1_done_cycle - run1_start_cycle);
        end
        $display("PASS tb_lenet5_top: both inferences cost exactly %0d cycles start_i -> done_o",
                 EXPECTED_INFERENCE_CYCLES);

        u_top_fsm_cov.report_and_check("tb_lenet5_top");
        $display("PASS tb_lenet5_top: control FSM state and transition coverage complete");
        $finish;
    end

    initial begin
        repeat (`TV_TOP_WATCHDOG_CYCLES) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
