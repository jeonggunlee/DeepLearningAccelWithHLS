#include <iostream>
#include <hls_stream.h>
#include <cmath>
#include <cstdlib>
#include <stdint.h>

#include "gemm16_tiled_accel.h"

#define N 16
#define EPS 1e-5

// ================= SW Reference =================
void gemm_sw(float A[N][N], float B[N][N], float C[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++)
                s+=A[i][k]*B[k][j];
            C[i][j]=s;
        }
}

// float ↔ u32 변환
static inline ap_uint<32> f2u(float f){
    union{float f; uint32_t u;} v;
    v.f=f;
    return v.u;
}
static inline float u2f(ap_uint<32> u){
    union{float f; uint32_t u;} v;
    v.u=(uint32_t)u;
    return v.f;
}

// ================= MAIN TB =================
int main()
{
    std::cout<<"===== GEMM16 HLS Testbench ====="<<std::endl;

    float A[N][N];
    float B[N][N];
    float C_ref[N][N];
    float C_hw[N][N];

    // ---------------- Init ----------------
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            A[i][j]= i + j*0.1f;
            B[i][j]= j + i*0.2f;
        }

    // SW reference
    gemm_sw(A,B,C_ref);

    // AXIS streams
    hls::stream<axis32_t> s_in;
    hls::stream<axis32_t> s_out;

    // ---------------- Write input stream ----------------
    axis32_t pkt;

    // A
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            pkt.data=f2u(A[i][j]);
            pkt.keep=0xF;
            pkt.strb=0xF;
            pkt.user=0;
            pkt.id=0;
            pkt.dest=0;
            pkt.last=0;
            s_in.write(pkt);
        }

    // B
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            pkt.data=f2u(B[i][j]);
            pkt.keep=0xF;
            pkt.strb=0xF;
            pkt.user=0;
            pkt.id=0;
            pkt.dest=0;
            pkt.last=0;
            s_in.write(pkt);
        }

    // ---------------- Call DUT ----------------
    gemm16_accel(s_in,s_out);

    // ---------------- Read output ----------------
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            axis32_t r=s_out.read();
            C_hw[i][j]=u2f(r.data);
        }

    // ---------------- Compare ----------------
    float max_err=0;
    float rmse=0;

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float e=fabs(C_hw[i][j]-C_ref[i][j]);
            if(e>max_err) max_err=e;
            rmse+=e*e;
        }

    rmse=sqrt(rmse/(N*N));

    std::cout<<"max_err = "<<max_err<<std::endl;
    std::cout<<"rmse    = "<<rmse<<std::endl;

    if(max_err < EPS){
        std::cout<<"PASS ✅"<<std::endl;
        return 0;
    }
    else{
        std::cout<<"FAIL ❌"<<std::endl;
        return 1;
    }
}

