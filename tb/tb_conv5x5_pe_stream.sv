`timescale 1ns/1ps
`include "vectors/config.svh"

// Streaming differential test for rtl/conv5x5_pe.sv against the Python oracle.
//
// `tb_conv5x5_pe.sv` remains as the fast smoke test, and this is what actually
// exercises the block. That one produces a single output pixel -- five rows of
// all-ones activations, weights 1..5, bias 5, **shift 0, ReLU off** -- which
// leaves most of the PE's control logic unexecuted: the requantizer inside the
// PE never sees a non-zero shift, never saturates and never clamps; the bias is
// never negative or large; a pixel never follows another, so the accumulator's
// restart at `first_i` is never tested; and `first_i` and `last_i` are never
// asserted on the same beat. A gate-level mutation survives in exactly that
// wrapper logic (results/gls_20260815.log), which is what an untested control
// path looks like from the outside.
//
// This drives 848 pixels of 1 to 8 rows across every shift 0..31 with ReLU off
// and on, both saturation rails, the exact rounding ties, and bias from zero to
// +/-2^28 -- with randomized gaps on the input stream and randomized
// backpressure on the output, so the stall paths are entered rather than
// assumed. Vectors come from golden/generate_vectors.py: generate_pe_vectors().
//
// One contract detail drives the structure below. `shift_i` and `relu_en_i`
// feed the requantizer combinationally, and the output register samples it at
// the cycle the beat is captured -- not at `last_i` -- so they must stay stable
// while a pixel is in flight. `conv2d_engine` holds them constant for a whole
// layer, so the testbench streams one pixel straight into the next only when
// both match and drains first otherwise. The count of back-to-back pixels is
// asserted, because those are the only pixels that exercise the accumulator
// restart.
module tb_conv5x5_pe_stream;
    localparam integer PIXELS     = `TV_PE_PIXELS;
    localparam integer ROWS_TOTAL = `TV_PE_ROWS_TOTAL;

    logic clk;
    logic rst_n;
    logic in_valid;
    logic in_ready;
    logic first;
    logic last;
    logic signed [39:0] act_row;
    logic signed [39:0] wgt_row;
    logic signed [31:0] bias;
    logic [5:0] shift;
    logic relu_en;
    logic out_valid;
    logic out_ready;
    logic signed [7:0] out_data;

    logic [39:0] act_vector [0:ROWS_TOTAL-1];
    logic [39:0] wgt_vector [0:ROWS_TOTAL-1];
    logic [7:0]  rows_vector     [0:PIXELS-1];
    logic [31:0] bias_vector     [0:PIXELS-1];
    logic [7:0]  shift_vector    [0:PIXELS-1];
    logic [7:0]  relu_vector     [0:PIXELS-1];
    logic [7:0]  expected_vector [0:PIXELS-1];

    integer pixel;
    integer row;
    integer row_base;
    integer gap;
    integer outputs_seen;
    integer mismatches;
    integer back_to_back;
    integer sat_hi_seen;
    integer sat_lo_seen;
    integer relu_clamped_seen;
    integer one_row_seen;
    integer shifts_used;
    integer stalls;
    logic [31:0] shift_seen_mask;

    conv5x5_pe dut (
        .clk_i       (clk),
        .rst_ni      (rst_n),
        .in_valid_i  (in_valid),
        .in_ready_o  (in_ready),
        .first_i     (first),
        .last_i      (last),
        .act_row_i   (act_row),
        .wgt_row_i   (wgt_row),
        .bias_i      (bias),
        .shift_i     (shift),
        .relu_en_i   (relu_en),
        .out_valid_o (out_valid),
        .out_ready_i (out_ready),
        .out_data_o  (out_data)
    );

    // The same continuous protocol checker the engine testbenches use: valid is
    // never withdrawn before acceptance, and the payload never moves while
    // stalled. Both are invisible to a testbench that holds ready high.
    stream_hold_check #(.PAYLOAD_WIDTH(8)) u_hold (
        .clk_i     (clk),
        .rst_ni    (rst_n),
        .valid_i   (out_valid),
        .ready_i   (out_ready),
        .payload_i (out_data)
    );

    always #5 clk = ~clk;

    // Backpressure and input gaps come from one free-running LFSR, seeded and
    // taps identical to tb_robustness.sv's, so a failure is reproducible from
    // the log rather than from a wall-clock seed.
    logic [15:0] lfsr;
    always @(negedge clk) begin
        if (!rst_n) begin
            lfsr <= 16'hACE1;
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
        end
    end
    assign out_ready = lfsr[0] | lfsr[3];

    // ------------------------------------------------------------------
    // Scoreboard. Every accepted output beat is compared in order against the
    // golden value for that pixel; a beat that never arrives is caught by the
    // final count rather than by a timeout that could be tuned away.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            if (outputs_seen >= PIXELS) begin
                $fatal(1, "tb_conv5x5_pe_stream: PE produced more outputs than pixels sent (%0d)",
                       outputs_seen + 1);
            end
            if ($signed(out_data) !== $signed(expected_vector[outputs_seen])) begin
                mismatches = mismatches + 1;
                if (mismatches <= 10) begin
                    $display("MISMATCH pixel %0d: rows=%0d shift=%0d relu=%0b bias=%0d -> got %0d, expected %0d",
                             outputs_seen, rows_vector[outputs_seen],
                             shift_vector[outputs_seen], relu_vector[outputs_seen][0],
                             $signed(bias_vector[outputs_seen]), $signed(out_data),
                             $signed(expected_vector[outputs_seen]));
                end
            end
            outputs_seen = outputs_seen + 1;
        end
    end

    // Present one kernel row and hold it until the PE accepts the beat. Values
    // are driven on the falling edge and sampled on the rising one; leaving
    // in_valid high across calls is what produces genuinely back-to-back beats
    // when `gap` is zero.
    task automatic send_row(input [39:0] a, input [39:0] w,
                            input logic f, input logic l, input integer idle);
        integer g;
        begin
            for (g = 0; g < idle; g = g + 1) begin
                @(negedge clk);
                in_valid = 1'b0;
            end
            @(negedge clk);
            in_valid = 1'b1;
            act_row  = a;
            wgt_row  = w;
            first    = f;
            last     = l;
            @(posedge clk);
            while (!in_ready) @(posedge clk);
        end
    endtask

    // Wait until every pixel sent so far has produced its output, so shift_i
    // and relu_en_i can be changed without disturbing a pixel still in flight.
    task automatic drain(input integer sent);
        begin
            @(negedge clk);
            in_valid = 1'b0;
            first    = 1'b0;
            last     = 1'b0;
            while (outputs_seen < sent) @(negedge clk);
        end
    endtask

    initial begin
        $readmemh("vectors/pe/act.hex", act_vector);
        $readmemh("vectors/pe/wgt.hex", wgt_vector);
        $readmemh("vectors/pe/rows.hex", rows_vector);
        $readmemh("vectors/pe/bias.hex", bias_vector);
        $readmemh("vectors/pe/shift.hex", shift_vector);
        $readmemh("vectors/pe/relu.hex", relu_vector);
        $readmemh("vectors/pe/expected.hex", expected_vector);

        clk               = 1'b0;
        rst_n             = 1'b0;
        in_valid          = 1'b0;
        first             = 1'b0;
        last              = 1'b0;
        act_row           = '0;
        wgt_row           = '0;
        bias              = '0;
        shift             = 6'd0;
        relu_en           = 1'b0;
        outputs_seen      = 0;
        mismatches        = 0;
        back_to_back      = 0;
        one_row_seen      = 0;
        shift_seen_mask   = 32'b0;
        row_base          = 0;

        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        for (pixel = 0; pixel < PIXELS; pixel = pixel + 1) begin
            if (pixel > 0) begin
                if (shift_vector[pixel] !== shift_vector[pixel-1] ||
                    relu_vector[pixel] !== relu_vector[pixel-1]) begin
                    drain(pixel);
                end else begin
                    // Streamed straight in: the PE starts this pixel while the
                    // previous one is still in its requantize stage, which is
                    // the only way the accumulator restart at first_i is
                    // exercised at all.
                    back_to_back = back_to_back + 1;
                end
            end

            bias    = $signed(bias_vector[pixel]);
            shift   = shift_vector[pixel][5:0];
            relu_en = relu_vector[pixel][0];
            shift_seen_mask[shift_vector[pixel][4:0]] = 1'b1;
            if (rows_vector[pixel] == 8'd1) one_row_seen = one_row_seen + 1;

            for (row = 0; row < rows_vector[pixel]; row = row + 1) begin
                // Idle beats between rows, from the same LFSR that drives the
                // output backpressure, so the feeder stalls mid-pixel too.
                gap = (lfsr[5] & lfsr[9]) ? 1 : 0;
                send_row(act_vector[row_base + row], wgt_vector[row_base + row],
                         (row == 0), (row == rows_vector[pixel] - 1), gap);
            end
            row_base = row_base + rows_vector[pixel];
        end

        drain(PIXELS);
        repeat (4) @(negedge clk);

        // --------------------------------------------------------------
        // Verdict. Mismatches first, then the coverage the run must have
        // reached -- a pass over stimulus that never saturated, never clamped
        // and never streamed two pixels together would be a false pass.
        // --------------------------------------------------------------
        if (mismatches != 0) begin
            $fatal(1, "tb_conv5x5_pe_stream: %0d/%0d pixels mismatched the golden model",
                   mismatches, PIXELS);
        end
        if (outputs_seen != PIXELS) begin
            $fatal(1, "tb_conv5x5_pe_stream: expected %0d output pixels, saw %0d",
                   PIXELS, outputs_seen);
        end
        if (row_base != ROWS_TOTAL) begin
            $fatal(1, "tb_conv5x5_pe_stream: consumed %0d rows of %0d -- vectors and config.svh disagree",
                   row_base, ROWS_TOTAL);
        end

        sat_hi_seen       = 0;
        sat_lo_seen       = 0;
        relu_clamped_seen = 0;
        for (pixel = 0; pixel < PIXELS; pixel = pixel + 1) begin
            if ($signed(expected_vector[pixel]) == 8'sd127)  sat_hi_seen = sat_hi_seen + 1;
            if ($signed(expected_vector[pixel]) == -8'sd128) sat_lo_seen = sat_lo_seen + 1;
            if (relu_vector[pixel][0] && expected_vector[pixel] == 8'd0)
                relu_clamped_seen = relu_clamped_seen + 1;
        end
        shifts_used = $countones(shift_seen_mask);
        stalls      = u_hold.stall_cycles;

        if (sat_hi_seen != `TV_PE_SAT_HI || sat_lo_seen != `TV_PE_SAT_LO) begin
            $fatal(1, "tb_conv5x5_pe_stream: saturation coverage %0d/%0d disagrees with config.svh %0d/%0d",
                   sat_hi_seen, sat_lo_seen, `TV_PE_SAT_HI, `TV_PE_SAT_LO);
        end
        if (relu_clamped_seen < `TV_PE_RELU_CLAMPED) begin
            $fatal(1, "tb_conv5x5_pe_stream: only %0d ReLU-clamped pixels, expected %0d",
                   relu_clamped_seen, `TV_PE_RELU_CLAMPED);
        end
        if (one_row_seen != `TV_PE_ONE_ROW || one_row_seen == 0) begin
            $fatal(1, "tb_conv5x5_pe_stream: %0d single-beat pixels, expected %0d -- first_i and last_i together is untested",
                   one_row_seen, `TV_PE_ONE_ROW);
        end
        if (shifts_used != `TV_PE_SHIFTS_USED || shifts_used != 32) begin
            $fatal(1, "tb_conv5x5_pe_stream: %0d distinct shifts driven, expected %0d",
                   shifts_used, `TV_PE_SHIFTS_USED);
        end
        if (back_to_back != `TV_PE_BACK_TO_BACK || back_to_back == 0) begin
            $fatal(1, "tb_conv5x5_pe_stream: %0d back-to-back pixels, expected %0d -- the accumulator restart is what these cover",
                   back_to_back, `TV_PE_BACK_TO_BACK);
        end
        if (stalls == 0) begin
            $fatal(1, "tb_conv5x5_pe_stream: the output stream never stalled, so the hold path was not exercised");
        end

        $display("PASS tb_conv5x5_pe_stream: %0d/%0d pixels matched the Python golden model", PIXELS, PIXELS);
        $display("  %0d rows driven, %0d back-to-back pixels, %0d single-beat pixels",
                 ROWS_TOTAL, back_to_back, one_row_seen);
        $display("  %0d/32 shifts, %0d saturate-high, %0d saturate-low, %0d ReLU-clamped, %0d output stall cycles",
                 shifts_used, sat_hi_seen, sat_lo_seen, relu_clamped_seen, stalls);
        $finish;
    end
endmodule
