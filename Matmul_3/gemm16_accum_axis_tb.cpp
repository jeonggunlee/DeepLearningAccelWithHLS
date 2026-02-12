#include <iostream>
#include <cmath>
#include <cstring>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

#define N 16
#define EPS 0.005

// ⭐ 매크로 대신 const 사용 (CSIM 안전)
const int Ktiles_tb = 3;

typedef ap_axiu<32,0,0,0> axis_t;

// DUT prototype
void gemm16_accum_axis(
    hls::stream<axis_t>& s_in,
    hls::stream<axis_t>& s_out,
    int Ktiles
);

// =====================================================
// bit cast helpers (CSIM-safe)
// =====================================================
static inline ap_uint<32> f2u(float f){
    uint32_t tmp;
    std::memcpy(&tmp, &f, sizeof(float));
    return ap_uint<32>(tmp);
}

static inline float u2f(ap_uint<32> u){
    uint32_t tmp = u.to_uint();
    float f;
    std::memcpy(&f, &tmp, sizeof(float));
    return f;
}

// =====================================================
// SW GEMM (reference)
// =====================================================
void gemm16_sw(float A[N][N], float B[N][N], float C[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++)
                s += A[i][k]*B[k][j];
            C[i][j]=s;
        }
}

// =====================================================
// Main Testbench
// =====================================================
int main()
{
    std::cout << "\n===== GEMM16_ACCUM_AXIS CSIM TEST =====\n";

    hls::stream<axis_t> s_in;
    hls::stream<axis_t> s_out;

    float A[Ktiles_tb][N][N];
    float B[Ktiles_tb][N][N];

    float Cref[N][N] = {0};
    float Ctmp[N][N];
    float Chw [N][N];

    // -------------------------------------------------
    // Generate input matrices
    // -------------------------------------------------
    for(int kt=0; kt<Ktiles_tb; kt++)
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                A[kt][i][j] = i + j*0.1f + kt*0.5f;
                B[kt][i][j] = j + i*0.2f + kt*0.3f;
            }

    // -------------------------------------------------
    // SW reference accumulate
    // -------------------------------------------------
    for(int kt=0; kt<Ktiles_tb; kt++){
        gemm16_sw(A[kt],B[kt],Ctmp);
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++)
                Cref[i][j] += Ctmp[i][j];
    }

    // -------------------------------------------------
    // Pack AXIS input stream
    // frame = 512 words (A then B)
    // TLAST on each frame end
    // -------------------------------------------------
    int words_in = 0;

    for(int kt=0; kt<Ktiles_tb; kt++)
    {
        // ---- A ----
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                axis_t w;
                w.data = f2u(A[kt][i][j]);
                w.keep = 0xF;
                w.strb = 0xF;
                w.user = 0;
                w.id   = 0;
                w.dest = 0;
                w.last = 0;

                s_in.write(w);
                words_in++;
            }

        // ---- B ----
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++){
                axis_t w;
                w.data = f2u(B[kt][i][j]);
                w.keep = 0xF;
                w.strb = 0xF;
                w.user = 0;
                w.id   = 0;
                w.dest = 0;

                // ⭐ TLAST at frame end
                w.last = (i==N-1 && j==N-1) ? 1 : 0;

                s_in.write(w);
                words_in++;
            }
    }

    std::cout << "Input words  : " << words_in
              << "  (expected " << Ktiles_tb*512 << ")\n";

    // -------------------------------------------------
    // Run DUT
    // -------------------------------------------------
    gemm16_accum_axis(s_in, s_out, Ktiles_tb);

    // -------------------------------------------------
    // Read output
    // -------------------------------------------------
    int words_out = 0;
    bool last_seen = false;

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            axis_t w = s_out.read();
            Chw[i][j] = u2f(w.data);

            if(w.last){
                last_seen = true;
                std::cout << "TLAST at output index = "
                          << words_out << std::endl;
            }
            words_out++;
        }

    std::cout << "Output words : " << words_out
              << "  (expected 256)\n";

    // -------------------------------------------------
    // Error check
    // -------------------------------------------------
    float max_err = 0;

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float e = fabs(Cref[i][j]-Chw[i][j]);
            if(e > max_err) max_err = e;
        }

    std::cout << "Max error = " << max_err << std::endl;

    // -------------------------------------------------
    // Result
    // -------------------------------------------------
    if(max_err < EPS && last_seen && words_out==256)
        std::cout << "\nPASS ✅\n";
    else
        std::cout << "\nFAIL ❌\n";

    return 0;
}
