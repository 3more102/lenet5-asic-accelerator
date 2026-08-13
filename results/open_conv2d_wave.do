onerror {resume}
view wave
add wave -divider {CONTROL}
add wave -radix unsigned /tb_conv2d_engine/clk
add wave -radix unsigned /tb_conv2d_engine/rst_n
add wave -radix unsigned /tb_conv2d_engine/start
add wave -radix unsigned /tb_conv2d_engine/busy
add wave -radix unsigned /tb_conv2d_engine/done
add wave -radix unsigned /tb_conv2d_engine/config_error

add wave -divider {OUTPUT STREAM}
add wave -radix unsigned /tb_conv2d_engine/out_valid
add wave -radix unsigned /tb_conv2d_engine/out_ready
add wave -radix decimal  /tb_conv2d_engine/out_data
add wave -radix unsigned /tb_conv2d_engine/out_channel
add wave -radix unsigned /tb_conv2d_engine/out_y
add wave -radix unsigned /tb_conv2d_engine/out_x

add wave -divider {LOAD INTERFACES}
add wave -radix unsigned /tb_conv2d_engine/load_act_we
add wave -radix unsigned /tb_conv2d_engine/load_act_addr
add wave -radix decimal  /tb_conv2d_engine/load_act_data
add wave -radix unsigned /tb_conv2d_engine/load_wgt_we
add wave -radix unsigned /tb_conv2d_engine/load_wgt_addr
add wave -radix decimal  /tb_conv2d_engine/load_wgt_data
add wave -radix unsigned /tb_conv2d_engine/load_bias_we
add wave -radix unsigned /tb_conv2d_engine/load_bias_addr
add wave -radix decimal  /tb_conv2d_engine/load_bias_data
add wave -radix unsigned /tb_conv2d_engine/load_conn_we
add wave -radix unsigned /tb_conv2d_engine/load_conn_addr
add wave -radix unsigned /tb_conv2d_engine/load_conn_data

add wave -divider {TESTBENCH STATUS}
add wave -radix decimal /tb_conv2d_engine/cycle_count
add wave -radix decimal /tb_conv2d_engine/received_count
add wave -radix decimal /tb_conv2d_engine/expected_oc
add wave -radix decimal /tb_conv2d_engine/expected_y
add wave -radix decimal /tb_conv2d_engine/expected_x

wave zoom full
configure wave -timelineunits ns
