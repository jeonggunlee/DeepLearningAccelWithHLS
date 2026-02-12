// ================================================================
// gemm16_accum_axis.cpp  (timing-safe + fast + CSIM-safe)
//  - Target: Zynq-7000 (xc7z020) @ 100MHz class
//  - AXI4-Stream in/out (32-bit float packed in TDATA)
//  - AXI-Lite control: Ktiles
//
//  - Key optimization: MANUAL ADER TREE reduction for k dimension
//    * Avoids long ripple reduction chain from naive "sum += a*b" w/ UNROLL
//    * Uses 8-way MAC chunk: 8 muls + 3-level add tree per chunk
//    * Accumulates chunk sums into C[i][j]
//
//  - Protocol:
//      Input:  Ktiles frames, each frame = A16(256) + B16(256) = 512 words
//              TLAST recommended at end of each 512-word frame (not required by this code)
//      Output: C16(256) words, TLAST asserted on last output word
//
//  - CSIM-safe float<->u32 bitcast via memcpy (no union w/ ap_uint)
// ================================================================

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <cstring>
#include <stdint.h>

#define N 16
#define KCHUNK 8   // 8-way reduction chunk (must divide N=16)

typedef ap_axiu<32, 0, 0, 0> axis_t;

// ------------------------------
// CSIM-safe bit reinterpretation
// ------------------------------
static inline float u32_to_f(ap_uint<32> u) {
#pragma HLS INLINE
    float f;
    uint32_t tmp = (uint32_t)u.to_uint();
    std::memcpy(&f, &tmp, sizeof(float));
    return f;
}
static inline ap_uint<32> f_to_u32(float f) {
#pragma HLS INLINE
    uint32_t tmp;
    std::memcpy(&tmp, &f, sizeof(uint32_t));
    return ap_uint<32>(tmp);
}

// ------------------------------
// 8-way adder-tree reduction of 8 products
// (CSA-like effect via balanced tree; HLS-friendly)
// ------------------------------
static inline float reduce8_tree(float p0, float p1, float p2, float p3,
                                 float p4, float p5, float p6, float p7) {
#pragma HLS INLINE
    float s0 = p0 + p1;
    float s1 = p2 + p3;
    float s2 = p4 + p5;
    float s3 = p6 + p7;

    float s4 = s0 + s1;
    float s5 = s2 + s3;

    return s4 + s5;
}

// ------------------------------
// Top
// ------------------------------
void gemm16_accum_axis(
    hls::stream<axis_t>& s_in,
    hls::stream<axis_t>& s_out,
    int Ktiles
){
#pragma HLS INTERFACE axis register_mode=both port=s_in
#pragma HLS INTERFACE axis register_mode=both port=s_out
#pragma HLS INTERFACE s_axilite port=Ktiles bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    float A[N][N];
    float B[N][N];
    float C[N][N];

    // Partition for fast access in inner MAC:
    // A[i][k] needs k-parallel access (dim=2), B[k][j] needs k-parallel access (dim=1)
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2

    if (Ktiles <= 0) return;

    // ------------------------------
    // Clear accumulator
    // ------------------------------
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = 0.0f;
        }
    }

    // ------------------------------
    // Process Ktiles frames: recv A,B then accumulate
    // ------------------------------
    for (int kt=0; kt<Ktiles; kt++) {

        // ---- recv A ----
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
#pragma HLS PIPELINE II=1
                axis_t w = s_in.read();
                A[i][j] = u32_to_f(w.data);
            }
        }

        // ---- recv B ----
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
#pragma HLS PIPELINE II=1
                axis_t w = s_in.read();
                B[i][j] = u32_to_f(w.data);
            }
        }

        // ---- compute: C += A*B with 8-way tree reduction ----
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
#pragma HLS PIPELINE II=1

                float sum = 0.0f;

                // N=16, KCHUNK=8 => 2 chunks: kb=0 and kb=8
                for (int kb = 0; kb < N; kb += KCHUNK) {
#pragma HLS UNROLL

                    // 8 parallel products
                    float p0 = A[i][kb + 0] * B[kb + 0][j];
                    float p1 = A[i][kb + 1] * B[kb + 1][j];
                    float p2 = A[i][kb + 2] * B[kb + 2][j];
                    float p3 = A[i][kb + 3] * B[kb + 3][j];
                    float p4 = A[i][kb + 4] * B[kb + 4][j];
                    float p5 = A[i][kb + 5] * B[kb + 5][j];
                    float p6 = A[i][kb + 6] * B[kb + 6][j];
                    float p7 = A[i][kb + 7] * B[kb + 7][j];

                    // balanced add tree (depth=3)
                    float part = reduce8_tree(p0,p1,p2,p3,p4,p5,p6,p7);

                    // accumulate 2 partial sums (depth small)
                    sum += part;
                }

                C[i][j] += sum;
            }
        }
    }

    // ------------------------------
    // Send C (256 words), TLAST on last word
    // ------------------------------
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
#pragma HLS PIPELINE II=1
            axis_t o;
            o.data = f_to_u32(C[i][j]);
            o.keep = (ap_uint<4>)0xF;
            o.strb = (ap_uint<4>)0xF;
            o.user = 0;
            o.id   = 0;
            o.dest = 0;
            o.last = ((i == N-1) && (j == N-1)) ? 1 : 0;
            s_out.write(o);
        }
    }
}
