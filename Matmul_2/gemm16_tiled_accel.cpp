#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>
#include <ap_axi_sdata.h>

typedef float dtype;

typedef ap_axiu<32,0,0,0> axis32_t;

static inline dtype u32_to_f(ap_uint<32> u){
#pragma HLS INLINE
    union { uint32_t ui; float f; } v;
    v.ui = (uint32_t)u;
    return v.f;
}

static inline ap_uint<32> f_to_u32(dtype f){
#pragma HLS INLINE
    union { uint32_t ui; float f; } v;
    v.f = f;
    return (ap_uint<32>)v.ui;
}

extern "C" void gemm16_accel(hls::stream<axis32_t>& s_in,
                            hls::stream<axis32_t>& s_out)
{
#pragma HLS INTERFACE axis port=s_in
#pragma HLS INTERFACE axis port=s_out
#pragma HLS INTERFACE ap_ctrl_none port=return   // ⭐ start 신호 불필요

    const int N = 16;

    dtype A[N][N];
    dtype B[N][N];
    dtype C[N][N];

#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2

// ================= READ A =================
READ_A:
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
        	axis32_t w = s_in.read();
        	A[i][j] = u32_to_f(w.data);
        }

// ================= READ B =================
READ_B:
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
        	axis32_t w = s_in.read();
        	B[i][j] = u32_to_f(w.data);
        }

// ================= C 초기화 =================
INIT_C:
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
            C[i][j]=0;
        }

// ================= GEMM =================
GEMM_I:
    for(int i=0;i<N;i++){
GEMM_J:
        for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
            float acc=0;

GEMM_K:
            for(int k=0;k<N;k++){
#pragma HLS UNROLL factor=4
                acc+=A[i][k]*B[k][j];
            }
            C[i][j]=acc;
        }
    }

// ================= WRITE =================
WRITE:
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
#pragma HLS PIPELINE II=1
        	axis32_t w;
        	w.data = f_to_u32(C[i][j]);
        	w.keep = 0xF;
        	w.strb = 0xF;
        	w.last = (i==15 && j==15);
        	s_out.write(w);
        }
}

