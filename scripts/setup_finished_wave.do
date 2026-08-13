# Final clean waveform setup for LeNet-5 ASIC CNN convolution project
onerror {resume}

view wave
view structure
view signals

# Remove any previous wave entries.
delete wave *

add wave -divider {CONTROL AND STATUS}
add wave -radix binary   sim:/tb_conv2d_engine/clk
add wave -radix binary   sim:/tb_conv2d_engine/rst_n
add wave -radix binary   sim:/tb_conv2d_engine/start
add wave -radix binary   sim:/tb_conv2d_engine/busy
add wave -radix binary   sim:/tb_conv2d_engine/done
add wave -radix binary   sim:/tb_conv2d_engine/config_error

add wave -divider {OUTPUT STREAM}
add wave -radix binary   sim:/tb_conv2d_engine/out_valid
add wave -radix binary   sim:/tb_conv2d_engine/out_ready
add wave -radix decimal  sim:/tb_conv2d_engine/out_data
add wave -radix unsigned sim:/tb_conv2d_engine/out_channel
add wave -radix unsigned sim:/tb_conv2d_engine/out_y
add wave -radix unsigned sim:/tb_conv2d_engine/out_x

add wave -divider {ACTIVATION LOAD}
add wave -radix binary   sim:/tb_conv2d_engine/load_act_we
add wave -radix unsigned sim:/tb_conv2d_engine/load_act_addr
add wave -radix decimal  sim:/tb_conv2d_engine/load_act_data

add wave -divider {WEIGHT AND BIAS LOAD}
add wave -radix binary   sim:/tb_conv2d_engine/load_wgt_we
add wave -radix unsigned sim:/tb_conv2d_engine/load_wgt_addr
add wave -radix decimal  sim:/tb_conv2d_engine/load_wgt_data
add wave -radix binary   sim:/tb_conv2d_engine/load_bias_we
add wave -radix unsigned sim:/tb_conv2d_engine/load_bias_addr
add wave -radix decimal  sim:/tb_conv2d_engine/load_bias_data

add wave -divider {CONNECTIVITY LOAD}
add wave -radix binary   sim:/tb_conv2d_engine/load_conn_we
add wave -radix unsigned sim:/tb_conv2d_engine/load_conn_addr
add wave -radix unsigned sim:/tb_conv2d_engine/load_conn_data

add wave -divider {TESTBENCH CHECKS}
add wave -radix decimal sim:/tb_conv2d_engine/cycle_count
add wave -radix decimal sim:/tb_conv2d_engine/received_count
add wave -radix decimal sim:/tb_conv2d_engine/expected_oc
add wave -radix decimal sim:/tb_conv2d_engine/expected_y
add wave -radix decimal sim:/tb_conv2d_engine/expected_x

configure wave -namecolwidth 245
configure wave -valuecolwidth 110
configure wave -timelineunits ns
configure wave -signalnamewidth 1
configure wave -gridperiod 100
configure wave -griddelta 40
configure wave -rowmargin 4
configure wave -childrowmargin 2

wave zoom full
