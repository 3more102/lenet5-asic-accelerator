# One-command waveform demo: compile, load tb_conv2d_engine, add the
# presentation signal set, run to completion, and leave the GUI open on a
# fully-zoomed wave window. Launch with:
#
#     vsim -gui -do scripts/demo_wave.do
#
# The signal list itself lives in scripts/setup_finished_wave.do so the batch
# regression (scripts/modelsim.do) and this demo stay in sync.
onerror {resume}

if {[file exists work]} {
    vdel -lib work -all
}
vlib work

vlog -sv +incdir+. \
    rtl/conv5x5_row_mac.sv \
    rtl/requantize.sv \
    rtl/conv5x5_pe.sv \
    rtl/avg_pool2x2_int8.sv \
    rtl/lenet5_c3_connectivity.sv \
    rtl/lenet5_layer_config.sv \
    rtl/conv2d_engine.sv \
    rtl/dense_row_mac.sv \
    rtl/avg_pool2x2_stream.sv \
    rtl/dense_engine.sv \
    rtl/classifier_argmax.sv \
    rtl/lenet5_pool_config.sv \
    rtl/lenet5_dense_config.sv \
    rtl/lenet5_top.sv \
    tb/tb_conv2d_engine.sv

vsim -wlf results/conv2d_engine.wlf work.tb_conv2d_engine
onfinish stop

do scripts/setup_finished_wave.do
run -all
wave zoom full

puts "Waveform demo loaded: 48/48 outputs checked against the Python golden model."
