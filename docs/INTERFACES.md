# Interfaces and Memory Layout

## `conv5x5_pe`

The PE uses independent ready/valid input and output handshakes.

- An input row transfers when `in_valid_i && in_ready_o`.
- `first_i` means this is the first connected row for an output pixel.
- `last_i` means this is the final connected row for that pixel.
- `bias_i`, `shift_i`, and `relu_en_i` must be valid with every row; only
  `bias_i` on the first row and quantization controls on the last row affect the
  result.
- An output transfers when `out_valid_o && out_ready_i`.
- While stalled, `out_valid_o` and `out_data_o` remain stable.

Each packed 40-bit row places lane zero in `[7:0]`, lane one in `[15:8]`, and
so on.

## `conv2d_engine`

The engine implements valid 5x5 convolution with stride 1 and padding 0.
Output dimensions are computed as `in_h-4` and `in_w-4`.

### Configuration

Configuration is sampled on `start_i` while `busy_o=0`.

| Signal | Meaning |
|---|---|
| `cfg_in_w_i`, `cfg_in_h_i` | input spatial size |
| `cfg_in_ch_i` | active input-channel count |
| `cfg_out_ch_i` | active output-channel count |
| `cfg_shift_i` | unsigned requantization right shift |
| `cfg_relu_en_i` | output ReLU enable |

Invalid dimensions pulse `config_error_o` and do not start. `done_o` pulses
after the final output handshake.

### Tensor memory order

All addresses are zero-based and tightly packed using the active runtime
dimensions.

Activation CHW:

`act_addr = ((input_channel * in_h) + y) * in_w + x`

Weight OIHW:

`wgt_addr = (((output_channel * in_ch) + input_channel) * 25) + ky*5 + kx`

Bias:

`bias_addr = output_channel`

Connection mask:

`conn_addr = output_channel * in_ch + input_channel`

A `1` connection bit enables that input/output channel pair. Dense layers load
all ones. C3 loads the exact matrix from `golden/lenet5.py` or
`rtl/lenet5_c3_connectivity.sv`.

### Output order

The stream is output-channel-major, then row, then column:

`flat_index = (output_channel * out_h + y) * out_w + x`

`out_channel_o`, `out_y_o`, and `out_x_o` travel with `out_data_o` and are held
under backpressure.

## `avg_pool2x2_stream`

Streaming controller for 2x2 average pooling (S2/S4), reused at runtime for
both via `cfg_in_w_i`/`cfg_in_h_i`/`cfg_in_ch_i` (one instance, not two).
Output size is `in_h/2` x `in_w/2`. Unlike `conv2d_engine`, there is no
`cfg_shift_i`/`cfg_relu_en_i`: the wrapped `avg_pool2x2_int8` primitive has
no such ports (shift=2, no ReLU is fixed inside it), so this controller
exposes none either.

### Configuration

Sampled on `start_i` while `busy_o=0`, same contract as `conv2d_engine`.
`cfg_in_w_i`/`cfg_in_h_i` must be even and within `MAX_IN_W`/`MAX_IN_H`;
`cfg_in_ch_i` must be within `MAX_IN_CH`. Invalid dimensions pulse
`config_error_o` and do not start.

### Tensor memory order

Only one preload port exists (`load_act_we_i`/`addr_i`/`data_i`), CHW
addressed identically to `conv2d_engine`: `act_addr = ((channel * in_h) + y)
* in_w + x`. Output order is channel-major, then row, then column, exactly
like `conv2d_engine`'s output stream, and `out_channel_o`/`out_y_o`/`out_x_o`
travel with `out_data_o` under the same backpressure contract.

## `dense_engine`

Memory-backed fully-connected layer engine, used for F6 and (a second,
internal instance) the classifier's dense stage. Same `start_i`/`busy_o`/
`done_o`/`config_error_o` and `load_*_we_i/addr_i/data_i` control idiom as
`conv2d_engine`, generalized to 1-D tensors: `cfg_in_len_i`/`cfg_out_len_i`
replace the spatial/channel config fields, and weights are addressed
row-major `wgt_addr = (out_index * in_len) + in_index` rather than OIHW.

In addition to the usual requantized `out_data_o` (int8) and `out_index_o`
streamed one neuron per beat, `dense_engine` exposes `out_acc_o`: the raw
accumulator for that neuron *before* `requantize`'s shift/round/ReLU/
saturate. `classifier_argmax` compares this port rather than `out_data_o` --
see the Classifier section below.

## Classifier (`classifier_argmax`)

Wraps one internal `dense_engine` (`cfg_out_len_i` tied to `NUM_CLASSES`,
`cfg_relu_en_i` tied low, its own `out_ready_i` tied high so every class
score is consumed the cycle it is produced) with a streaming running-max
comparator. All `load_*` ports forward straight through to the internal
engine; there is no separate weight/bias layout for the wrapper itself.

**Normative decision rule:** the comparator maxes over `dense_engine`'s raw
`out_acc_o`, not its saturated `out_data_o` int8 score.
`requantize`'s saturation clamp to `[-128,127]` is not injective -- with no
per-layer calibration step defined yet (see `VERIFICATION_PLAN.md`), two
classes can both exceed +127 and collapse to the same clamped score, which
would make an int8-score argmax pick an arbitrary winner among exactly the
classes most likely to be correct. Comparing the pre-saturation accumulator
avoids this failure mode entirely.

**Tie-break rule (normative):** ties resolve to the **lowest class index**,
matching NumPy's `np.argmax` default and `golden/quantized_conv.py:
argmax_classifier`. The RTL comparator enforces this with a **strict `>`**
(never `>=`) when updating the running best score, so a later class with an
equal score never overwrites an earlier one. Any classifier-shaped RTL added
later must preserve this rule to stay bit-exact with the golden model.

## `lenet5_top`

Top-level sequencer for the full C1->S2->C3->S4->C5->F6->classifier chain.
Resource-shared: one `conv2d_engine` runs C1/C3/C5 in turn, one
`avg_pool2x2_stream` runs S2/S4, one `dense_engine` runs F6, and one
`classifier_argmax` produces the final class. Per-stage weights/bias live in
persistent ROMs owned by `lenet5_top` and are copied into the active
engine's scratch memory immediately before that engine's `start_i`; every
stage's activation input is bridged directly from the previous stage's
output stream while that stage runs (both engines' load ports accept writes
independent of `busy_o`, and outputs hold stable until accepted, so no extra
buffering stage or shared memory is needed).

Weights/biases/the input image are preloaded through one demuxed ROM bus
before `start_i`: `load_we_i` + `cfg_rom_sel_i` (selects which of the 11 ROM
regions -- image, then weight+bias pairs for C1/C3/C5/F6/classifier) +
`load_addr_i`/`load_data_i`. C1 and C5 use full connectivity (a constant-1
fill, no ROM). C3's sparse connectivity is generated live by instantiating
the existing `lenet5_c3_connectivity` inside `lenet5_top` and walking it
with a counter, rather than storing the 96-bit table in a ROM.

`lenet5_top` is a behavioral/verification integration, not a synthesis
target -- like `conv2d_engine`, its ROM arrays stand in for future SRAM
macros (see `SEMICUSTOM_FLOW.md`).

## SRAM replacement notes

The reference engine uses behavioral arrays with combinational multi-read
access to keep verification simple. Do not sign off those arrays as physical
memories. A macro implementation should:

1. bank or widen activation/weight storage to deliver five lanes per cycle;
2. account for synchronous SRAM read latency in the feeder;
3. add clock gating only after handshake equivalence is preserved;
4. use ping-pong activation banks between convolution and pooling;
5. protect CDCs if the DMA and compute clocks differ.

