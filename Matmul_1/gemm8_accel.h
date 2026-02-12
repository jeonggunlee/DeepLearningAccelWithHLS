#pragma once
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef float dtype;

// AXI4-Stream 32-bit data (float) + TLAST
typedef ap_axiu<32,0,0,0> axis32_t;

extern "C" {
void gemm8_accel(hls::stream<axis32_t>& s_in,
                 hls::stream<axis32_t>& s_out);
}
