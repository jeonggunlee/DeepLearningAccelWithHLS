// ================================================================
// gemm16_accum_axis_db.cpp  (Double-Buffered version)
//  - Target: Zynq-7000 (xc7z020) @ 100MHz class
//  - AXI4-Stream in/out (32-bit float packed in TDATA)
//  - AXI-Lite control: Ktiles
//
//  - Key optimizations:
//    1) DOUBLE BUFFERING: overlap recv of next A/B tile with
//       compute of current tile via ping-pong buffers + DATAFLOW
//    2) MANUAL ADDER TREE: 8-way MAC chunk with balanced tree
//
//  - Protocol:
//      Input:  Ktiles frames, each frame = A16(256) + B16(256) = 512 words
//      Output: C16(256) words, TLAST asserted on last output word
//
//  - Pipeline structure (per Ktile iteration):
//      [recv A/B into buf[ping]] || [compute C += A*B from buf[pong]]
//      (first iteration: recv only, last iteration: compute only)
//      Total latency ≈ (Ktiles+1) * max(recv_time, compute_time)
//      vs. original: Ktiles * (recv_time + compute_time)
//
//  - CSIM-safe float<->u32 bitcast via memcpy
// ================================================================

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <cstring>
#include <stdint.h>

#define N 16
#define KCHUNK 8

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
// 8-way adder-tree reduction
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

// ==============================================================
// Sub-functions for DATAFLOW-friendly double buffering
// ==============================================================

// ---- Receive one A+B tile into flat arrays via FIFO streams ----
static void recv_tile(
    hls::stream<axis_t>& s_in,
    hls::stream<float>&  fifo_A,
    hls::stream<float>&  fifo_B)
{
    // recv A (256 floats)
    for (int idx = 0; idx < N*N; idx++) {
#pragma HLS PIPELINE II=1
        axis_t w = s_in.read();
        fifo_A.write(u32_to_f(w.data));
    }
    // recv B (256 floats)
    for (int idx = 0; idx < N*N; idx++) {
#pragma HLS PIPELINE II=1
        axis_t w = s_in.read();
        fifo_B.write(u32_to_f(w.data));
    }
}

// ---- Load A/B from FIFOs into local BRAM arrays ----
static void load_tile(
    hls::stream<float>& fifo_A,
    hls::stream<float>& fifo_B,
    float A[N][N],
    float B[N][N])
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            A[i][j] = fifo_A.read();
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            B[i][j] = fifo_B.read();
        }
    }
}

// ---- MAC: C += A * B with 8-way tree ----
static void mac_tile(
    float A[N][N],
    float B[N][N],
    float C[N][N])
{
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1

            float sum = 0.0f;

            for (int kb = 0; kb < N; kb += KCHUNK) {
#pragma HLS UNROLL
                float p0 = A[i][kb+0] * B[kb+0][j];
                float p1 = A[i][kb+1] * B[kb+1][j];
                float p2 = A[i][kb+2] * B[kb+2][j];
                float p3 = A[i][kb+3] * B[kb+3][j];
                float p4 = A[i][kb+4] * B[kb+4][j];
                float p5 = A[i][kb+5] * B[kb+5][j];
                float p6 = A[i][kb+6] * B[kb+6][j];
                float p7 = A[i][kb+7] * B[kb+7][j];

                float part = reduce8_tree(p0,p1,p2,p3,p4,p5,p6,p7);
                sum += part;
            }

            C[i][j] += sum;
        }
    }
}

// ---- Send C (256 words) with TLAST ----
static void send_result(
    float C[N][N],
    hls::stream<axis_t>& s_out)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
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

// ==============================================================
// Top: Double-Buffered GEMM16 accumulate
// ==============================================================
void gemm16_accum_axis_db(
    hls::stream<axis_t>& s_in,
    hls::stream<axis_t>& s_out,
    int Ktiles
){
#pragma HLS INTERFACE axis register_mode=both port=s_in
#pragma HLS INTERFACE axis register_mode=both port=s_out
#pragma HLS INTERFACE s_axilite port=Ktiles bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    if (Ktiles <= 0) return;

    // ---- Ping-pong buffers for A and B ----
    float A_buf[2][N][N];
    float B_buf[2][N][N];
    float C[N][N];

#pragma HLS ARRAY_PARTITION variable=A_buf complete dim=3
#pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2
#pragma HLS ARRAY_PARTITION variable=C     complete dim=2

    // Clear accumulator
    CLEAR_C:
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = 0.0f;
        }
    }

    // ================================================================
    // Double-buffering loop:
    //
    //  Iteration 0           : recv -> buf[0]
    //  Iteration 1           : recv -> buf[1]  ||  compute buf[0]
    //  Iteration 2           : recv -> buf[0]  ||  compute buf[1]
    //  ...
    //  Iteration Ktiles      :                     compute buf[last]
    //
    //  Total iterations = Ktiles + 1
    //  - iteration 0:       recv only  (prolog)
    //  - iteration 1..K-1:  recv || compute  (steady state)
    //  - iteration Ktiles:  compute only (epilog)
    // ================================================================
    for (int phase = 0; phase < Ktiles + 1; phase++) {

        int recv_buf = phase & 1;         // buffer index for receiving
        int comp_buf = (phase - 1) & 1;   // buffer index for computing (previous tile)

        bool do_recv    = (phase < Ktiles);
        bool do_compute = (phase > 0);

        // --- FIFOs to decouple stream read from BRAM write ---
        hls::stream<float> fifo_A("fifo_A");
        hls::stream<float> fifo_B("fifo_B");
#pragma HLS STREAM variable=fifo_A depth=256
#pragma HLS STREAM variable=fifo_B depth=256

        // --- DATAFLOW region: recv and compute run concurrently ---
#pragma HLS DATAFLOW

        // Stage 1: Receive next tile from AXI-Stream into FIFOs
        if (do_recv) {
            recv_tile(s_in, fifo_A, fifo_B);
        }

        // Stage 2: Load FIFOs into ping-pong BRAM
        if (do_recv) {
            load_tile(fifo_A, fifo_B, A_buf[recv_buf], B_buf[recv_buf]);
        }

        // Stage 3: MAC accumulate using previous tile's buffer
        if (do_compute) {
            mac_tile(A_buf[comp_buf], B_buf[comp_buf], C);
        }
    }

    // ---- Send result ----
    send_result(C, s_out);
}
