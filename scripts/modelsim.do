onerror {quit -code 1}

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
    tb/tb_conv5x5_pe.sv \
    tb/tb_requantize.sv \
    tb/tb_lenet5_c3_connectivity.sv \
    tb/tb_conv2d_engine.sv \
    tb/tb_avg_pool2x2_stream.sv \
    tb/tb_dense_engine.sv \
    tb/tb_classifier_argmax.sv \
    tb/tb_classifier_argmax_tie.sv \
    tb/tb_config_guard.sv \
    tb/stream_hold_check.sv \
    tb/tb_robustness.sv \
    tb/tb_extremes.sv \
    tb/fsm_cov.sv \
    tb/tb_lenet5_top.sv

vsim -c work.tb_conv5x5_pe
onfinish stop
run -all
quit -sim

vsim -c work.tb_requantize
onfinish stop
run -all
quit -sim

vsim -c work.tb_lenet5_c3_connectivity
onfinish stop
run -all
quit -sim

vsim -c -wlf results/conv2d_engine.wlf work.tb_conv2d_engine
onfinish stop
run -all
quit -sim

vsim -c work.tb_avg_pool2x2_stream
onfinish stop
run -all
quit -sim

vsim -c work.tb_dense_engine
onfinish stop
run -all
quit -sim

vsim -c work.tb_classifier_argmax
onfinish stop
run -all
quit -sim

vsim -c work.tb_classifier_argmax_tie
onfinish stop
run -all
quit -sim

vsim -c work.tb_config_guard
onfinish stop
run -all
quit -sim

vsim -c work.tb_robustness
onfinish stop
run -all
quit -sim

vsim -c work.tb_extremes
onfinish stop
run -all
quit -sim

vsim -c -wlf results/lenet5_top.wlf work.tb_lenet5_top
onfinish stop
run -all
quit -sim

puts "PASS: all ModelSim/Questa regressions completed"
quit -code 0
