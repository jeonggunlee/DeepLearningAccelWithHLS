// gemm16_tiled_accel.h
#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>
#include <ap_axi_sdata.h>

typedef float dtype;

typedef ap_axiu<32,0,0,0> axis32_t;

extern "C" void gemm16_accel(
    hls::stream<axis32_t>& s_in,
    hls::stream<axis32_t>& s_out);
