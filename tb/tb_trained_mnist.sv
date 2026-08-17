`timescale 1ns/1ps
`include "vectors/config.svh"

// ---------------------------------------------------------------------------
// lenet5_top running a *trained* network on *real MNIST digits*.
//
// Every other tier in this project checks the RTL against
// golden/deploy.py:deploy_forward_int8 driven by random_deploy_parameters --
// deterministic, shape-correct weights that the golden model itself documents
// as "not trained and not expected to classify MNIST". That is the right
// stimulus for finding arithmetic bugs, and it is why those tiers reach
// accumulator ranges a trained network never visits. What it cannot do is
// answer the only question a person outside this repository will ask: does the
// accelerator actually recognise a digit?
//
// This testbench answers it. The weights are trained (golden/train_lenet5.py),
// quantized to the exact fixed-point scheme rtl/requantize.sv implements, with
// a per-layer calibrated shift rather than the flat shift=7 the random tiers
// use. The images are unmodified MNIST test digits in dataset order -- no
// cherry-picking, which matters, because a set chosen for being classified
// correctly would make the accuracy check tautological.
//
// Two independent assertions per image, which are NOT the same claim:
//
//   1. class_o == the golden INT8 model's prediction. This is the hardware
//      fidelity check and must hold for every image regardless of whether the
//      network is any good. A wrong answer here is an RTL bug.
//   2. class_o == the true MNIST label, aggregated. This is a check on the
//      *network*, not the hardware, and it is asserted against the count the
//      golden model achieves on these same images -- so it encodes a measured
//      fact rather than a hope, and it still fails loudly if the RTL breaks.
//
// A third check has no equivalent anywhere else in the regression: every
// inference must cost an identical number of cycles. tb_lenet5_top reruns the
// *same* image twice, which proves state returns to idle but says nothing
// about data dependence. Here ten *different* images run back to back with the
// weight ROMs never rewritten, so a cycle count that moves with the data would
// mean some engine's sequencing depends on activation values.
// ---------------------------------------------------------------------------
module tb_trained_mnist;
    localparam integer DATA_WIDTH = 8;
    localparam integer ACC_WIDTH  = 32;

    localparam integer ROM_ADDR_WIDTH = $clog2(120*16*25);

    localparam integer NUM_IMAGES  = `TV_TR_NUM_IMAGES;
    localparam integer IMG_PIXELS  = 32*32;
    localparam integer EXPECTED_LABEL_MATCHES = `TV_TR_LABEL_MATCHES;

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

    // The calibrated shifts are passed in, not defaulted. Every other tier in
    // this project runs lenet5_top at its default shift=7 on all four layers,
    // so this instantiation is also the only place the SHIFT_* parameters are
    // exercised at non-default values -- a per-layer calibrated network simply
    // does not decode correctly at a flat 7, which is what makes this a real
    // check of the parameterisation rather than a restatement of it.
    lenet5_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .SHIFT_C1  (`TV_TR_SHIFT_C1),
        .SHIFT_C3  (`TV_TR_SHIFT_C3),
        .SHIFT_C5  (`TV_TR_SHIFT_C5),
        .SHIFT_F6  (`TV_TR_SHIFT_F6)
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

    integer cycle_count;
    always @(posedge clk) begin
        if (!rst_n) cycle_count <= 0;
        else        cycle_count <= cycle_count + 1;
    end

    // All NUM_IMAGES images concatenated, 1024 int8 pixels each.
    logic signed [7:0]  image_vector   [0:`TV_TR_NUM_IMAGES*32*32-1];
    logic signed [7:0]  expected_class [0:`TV_TR_NUM_IMAGES-1];
    logic signed [7:0]  true_label     [0:`TV_TR_NUM_IMAGES-1];

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

    integer img;
    integer model_matches;
    integer label_matches;
    integer run_start_cycle, run_done_cycle;
    integer first_run_cycles;
    integer this_run_cycles;

    always #5 clk = ~clk;

    // Weight ROMs, written once. Image ROM is deliberately excluded: it is
    // rewritten per image below, which is the point of this testbench.
    task automatic load_narrow(input integer sel, input integer count);
        integer index;
        begin
            for (index = 0; index < count; index = index + 1) begin
                @(negedge clk);
                load_we   = 1'b1;
                rom_sel   = sel[3:0];
                load_addr = index[ROM_ADDR_WIDTH-1:0];
                case (sel)
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

    // Rewrite only ROM_SEL_IMAGE, leaving all ten weight ROMs resident.
    task automatic load_image(input integer which);
        integer index;
        begin
            for (index = 0; index < IMG_PIXELS; index = index + 1) begin
                @(negedge clk);
                load_we   = 1'b1;
                rom_sel   = 4'd0;
                load_addr = index[ROM_ADDR_WIDTH-1:0];
                load_data = image_vector[which*IMG_PIXELS + index];
            end
            @(negedge clk);
            load_we = 1'b0;
        end
    endtask

    initial begin
        $readmemh("vectors/trained/images.hex",   image_vector);
        $readmemh("vectors/trained/expected.hex", expected_class);
        $readmemh("vectors/trained/labels.hex",   true_label);
        $readmemh("vectors/trained/c1_wgt.hex",   c1_wgt_vector);
        $readmemh("vectors/trained/c1_bias.hex",  c1_bias_vector);
        $readmemh("vectors/trained/c3_wgt.hex",   c3_wgt_vector);
        $readmemh("vectors/trained/c3_bias.hex",  c3_bias_vector);
        $readmemh("vectors/trained/c5_wgt.hex",   c5_wgt_vector);
        $readmemh("vectors/trained/c5_bias.hex",  c5_bias_vector);
        $readmemh("vectors/trained/f6_wgt.hex",   f6_wgt_vector);
        $readmemh("vectors/trained/f6_bias.hex",  f6_bias_vector);
        $readmemh("vectors/trained/cls_wgt.hex",  cls_wgt_vector);
        $readmemh("vectors/trained/cls_bias.hex", cls_bias_vector);

        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        load_we = 1'b0;
        rom_sel = '0;
        load_addr = '0;
        load_data = '0;
        model_matches = 0;
        label_matches = 0;
        first_run_cycles = -1;

        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        load_narrow(1, 6*1*25);
        load_wide(2, 6);
        load_narrow(3, 16*6*25);
        load_wide(4, 16);
        load_narrow(5, 120*16*25);
        load_wide(6, 120);
        load_narrow(7, 84*120);
        load_wide(8, 84);
        load_narrow(9, 10*84);
        load_wide(10, 10);

        for (img = 0; img < NUM_IMAGES; img = img + 1) begin
            load_image(img);

            @(negedge clk);
            start = 1'b1;
            run_start_cycle = cycle_count;
            @(negedge clk);
            start = 1'b0;

            wait (busy === 1'b1);
            wait (done === 1'b1);
            run_done_cycle = cycle_count;
            @(negedge clk);

            this_run_cycles = run_done_cycle - run_start_cycle;

            if (class_val !== expected_class[img][3:0]) begin
                $fatal(1,
                    "image %0d: RTL predicted %0d, golden INT8 model predicted %0d -- the hardware and the model disagree, which is an RTL bug independent of whether the network is accurate",
                    img, class_val, expected_class[img]);
            end
            model_matches = model_matches + 1;

            if (class_val === true_label[img][3:0]) begin
                label_matches = label_matches + 1;
            end

            if (first_run_cycles < 0) begin
                first_run_cycles = this_run_cycles;
            end else if (this_run_cycles != first_run_cycles) begin
                $fatal(1,
                    "image %0d took %0d cycles against image 0's %0d: inference cost must not depend on the image data",
                    img, this_run_cycles, first_run_cycles);
            end

            if (class_val === true_label[img][3:0]) begin
                $display("  image %0d: predicted %0d, label %0d, %0d cycles",
                         img, class_val, true_label[img], this_run_cycles);
            end else begin
                // Not a failure: the RTL agreed with the model above, so this
                // is the network being wrong, not the hardware. Printed so the
                // aggregate count below can be read against something.
                $display("  image %0d: predicted %0d, label %0d, %0d cycles   <- model error, hardware agreed with the model",
                         img, class_val, true_label[img], this_run_cycles);
            end

            wait (busy === 1'b0);
        end

        if (model_matches != NUM_IMAGES) begin
            $fatal(1, "only %0d/%0d images matched the golden model",
                   model_matches, NUM_IMAGES);
        end
        $display("PASS tb_trained_mnist: %0d/%0d real MNIST digits matched golden/deploy.py exactly, weights resident throughout",
                 model_matches, NUM_IMAGES);

        // Asserted against the golden model's own score on these same images,
        // so this fails if the RTL breaks, and does NOT silently pass by
        // redefining success downward if the network is retrained.
        if (label_matches != EXPECTED_LABEL_MATCHES) begin
            $fatal(1,
                "%0d/%0d digits classified correctly, expected %0d -- if the trained network changed, regenerate vectors so this constant tracks it",
                label_matches, NUM_IMAGES, EXPECTED_LABEL_MATCHES);
        end
        $display("PASS tb_trained_mnist: %0d/%0d digits classified correctly against their true MNIST labels",
                 label_matches, NUM_IMAGES);

        $display("PASS tb_trained_mnist: every inference cost exactly %0d cycles, independent of image data",
                 first_run_cycles);
        $finish;
    end

    initial begin
        repeat (`TV_TR_WATCHDOG_CYCLES) @(posedge clk);
        $fatal(1, "Simulation timeout");
    end
endmodule
