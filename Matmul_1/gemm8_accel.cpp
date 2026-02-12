#include "gemm8_accel.h"
#include <stdint.h>

// float <-> ap_uint<32> 변환 함수
static inline dtype u32_to_f(ap_uint<32> u) {
#pragma HLS INLINE
    union { uint32_t ui; float f; } v;
    v.ui = (uint32_t)u;
    return v.f;
}

static inline ap_uint<32> f_to_u32(dtype f) {
#pragma HLS INLINE
    union { uint32_t ui; float f; } v;
    v.f = f;
    return (ap_uint<32>)v.ui;
}

extern "C" void gemm8_accel(hls::stream<axis32_t>& s_in,
                           hls::stream<axis32_t>& s_out)
{
#pragma HLS INTERFACE axis port=s_in
#pragma HLS INTERFACE axis port=s_out
#pragma HLS INTERFACE ap_ctrl_none port=return

    dtype A[8][8];
    dtype B[8][8];
    dtype C[8][8];
    
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2

READ_A:
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II=1
            axis32_t word = s_in.read();
            A[i][j] = u32_to_f(word.data);
        }
    }

READ_B:
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II=1
            axis32_t word = s_in.read();
            B[i][j] = u32_to_f(word.data);
        }
    }

READ_C:
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II=1
            axis32_t word = s_in.read();
            C[i][j] = u32_to_f(word.data);
        }
    }

    GEMM_OUTER_I:
        for (int i = 0; i < 8; i++) {
    GEMM_OUTER_J:
            for (int j = 0; j < 8; j++) {
    #pragma HLS PIPELINE II=2  // II를 2로 완화
    #pragma HLS DEPENDENCE variable=C inter false
                dtype accumulator = C[i][j];

    GEMM_INNER_K:
                for (int k = 0; k < 8; k++) {
    #pragma HLS UNROLL factor=4
    #pragma HLS LATENCY min=0 max=2  // 레이턴시 제약 완화
                    dtype prod = A[i][k] * B[k][j];
    #pragma HLS BIND_OP variable=prod op=fmul impl=maxdsp
                    accumulator += prod;
    #pragma HLS BIND_OP variable=accumulator op=fadd impl=fabric
                }

                C[i][j] = accumulator;
            }
        }

WRITE_C:
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE II=1
            axis32_t word;
            word.data = f_to_u32(C[i][j]);
            word.keep = -1;
            word.strb = -1;
            word.user = 0;
            word.id   = 0;
            word.dest = 0;
            word.last = (i == 7 && j == 7) ? 1 : 0;
            s_out.write(word);
        }
    }
}
