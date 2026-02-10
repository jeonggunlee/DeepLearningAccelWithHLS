/********************************************************************
 * 32x32 GEMM via 16x16 AXI-Stream Accelerator (Zynq-7000 / Zybo Z7-20)
 *  - HLS gemm16_accel: input 512 floats (A16 + B16), output 256 floats (C16)
 *  - SW reference (32x32) + HW tiled (32x32) + timing + GFLOPS + error stats
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xtime_l.h"

// ================= CONFIG =================
#define N16 16
#define N32 32
#define NB  (N32 / N16)     // number of 16x16 blocks per dimension = 2

#define DMA_DEV_ID XPAR_AXIDMA_0_DEVICE_ID

#define CACHELINE 32
#define DMA_TIMEOUT 100000000
#define EPS 1e-6f

// ================= GLOBAL =================
static XAxiDma AxiDma;

// ================= TIMER =================
// Zynq-7000 GlobalTimer = CPU/2 → 반드시 ×2
static inline double cycles_to_us(XTime cyc){
    double freq = (double)XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ;
    return ((double)cyc * 2.0 * 1e6) / freq;
}

// ================= CACHE =================
static void cache_flush(void* p, int sz){
    UINTPTR a=(UINTPTR)p;
    a&=~(CACHELINE-1);
    sz=(sz+CACHELINE-1)&~(CACHELINE-1);
    Xil_DCacheFlushRange(a,sz);
}
static void cache_inv(void* p, int sz){
    UINTPTR a=(UINTPTR)p;
    a&=~(CACHELINE-1);
    sz=(sz+CACHELINE-1)&~(CACHELINE-1);
    Xil_DCacheInvalidateRange(a,sz);
}

// ================= INDEX =================
static inline int idx32(int r,int c){ return r*N32 + c; }
static inline int idx16(int r,int c){ return r*N16 + c; }

// ================= SW GEMM (32x32) =================
static void gemm32_sw(const float* A, const float* B, float* C){
    for(int i=0;i<N32;i++){
        for(int j=0;j<N32;j++){
            float s=0.0f;
            for(int k=0;k<N32;k++){
                s += A[idx32(i,k)] * B[idx32(k,j)];
            }
            C[idx32(i,j)] = s;
        }
    }
}

// ================= DMA (one 16x16 GEMM call) =================
static int run_dma_16x16(float *in_buf, float *out_buf){

    const int in_bytes  = 512*sizeof(float); // A16(256) + B16(256)
    const int out_bytes = 256*sizeof(float); // C16

    cache_flush(in_buf,in_bytes);
    cache_inv(out_buf,out_bytes);

    // S2MM 먼저 걸어두고 MM2S 전송 시작 (일반적으로 이 순서 권장)
    XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)out_buf,out_bytes,XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)in_buf ,in_bytes ,XAXIDMA_DMA_TO_DEVICE);

    int t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma,XAXIDMA_DMA_TO_DEVICE) && t--);
    if(t<=0){ printf("MM2S timeout\n"); return -1;}

    t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma,XAXIDMA_DEVICE_TO_DMA) && t--);
    if(t<=0){ printf("S2MM timeout\n"); return -1;}

    cache_inv(out_buf,out_bytes);
    return 0;
}

// ================= BLOCK HELPERS =================
// Extract 16x16 block from 32x32 matrix: (block row=br, block col=bc)
static void extract_block16_from32(const float* M32, int br, int bc, float* M16){
    int r0 = br * N16;
    int c0 = bc * N16;
    for(int i=0;i<N16;i++){
        for(int j=0;j<N16;j++){
            M16[idx16(i,j)] = M32[idx32(r0+i, c0+j)];
        }
    }
}

// Accumulate 16x16 block into 32x32 matrix: C32(block br, bc) += C16
static void accum_block16_to32(float* C32, int br, int bc, const float* C16){
    int r0 = br * N16;
    int c0 = bc * N16;
    for(int i=0;i<N16;i++){
        for(int j=0;j<N16;j++){
            C32[idx32(r0+i, c0+j)] += C16[idx16(i,j)];
        }
    }
}

// ================= HW GEMM (32x32 via 16x16 accel) =================
static int gemm32_hw_tiled(const float* A32, const float* B32, float* C32,
                          float* in_buf, float* out_buf,
                          float* A16, float* B16)
{
    // C 초기화
    for(int i=0;i<N32*N32;i++) C32[i] = 0.0f;

    // 32x32 블록 곱: NB=2 → 총 가속기 호출 횟수 = NB^3 = 8
    for(int bi=0; bi<NB; bi++){
        for(int bj=0; bj<NB; bj++){

            for(int bk=0; bk<NB; bk++){
                // Ablock = A32(bi,bk), Bblock = B32(bk,bj)
                extract_block16_from32(A32, bi, bk, A16);
                extract_block16_from32(B32, bk, bj, B16);

                // pack: in_buf[0..255]=A16, in_buf[256..511]=B16
                memcpy(in_buf,      A16, 256*sizeof(float));
                memcpy(in_buf+256,  B16, 256*sizeof(float));

                if(run_dma_16x16(in_buf, out_buf) != 0){
                    printf("DMA/HW failed at (bi,bj,bk)=(%d,%d,%d)\n", bi,bj,bk);
                    return -1;
                }

                // Cblock_partial(out_buf) 누적: C32(bi,bj) += out_buf
                accum_block16_to32(C32, bi, bj, out_buf);
            }
        }
    }
    return 0;
}

// ================= ERROR =================
static void error_stats_32(const float* ref, const float* hw){

    float max_abs=0, max_rel=0;
    double err2=0, ref2=0;

    for(int i=0;i<N32*N32;i++){
        float e = hw[i]-ref[i];
        float a = fabsf(e);

        if(a>max_abs) max_abs=a;

        float rel = a/(fabsf(ref[i])+EPS);
        if(rel>max_rel) max_rel=rel;

        err2 += (double)e*(double)e;
        ref2 += (double)ref[i]*(double)ref[i];
    }

    printf("\nError Stats (32x32)\n");
    printf("max_abs   = %.8f\n", max_abs);
    printf("max_rel   = %.8f\n", max_rel);
    printf("rmse      = %.8f\n", (float)sqrt(err2/(N32*N32)));
    printf("rel_frob  = %.8f\n", (float)(sqrt(err2)/sqrt(ref2)));
}

// ================= MAIN =================
int main(){

    printf("\n===== GEMM32 via GEMM16 AXI-DMA Benchmark =====\n");

    // DMA init
    XAxiDma_Config* cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    if(!cfg){
        printf("No DMA config found.\n");
        return -1;
    }
    if(XAxiDma_CfgInitialize(&AxiDma,cfg) != XST_SUCCESS){
        printf("DMA init failed.\n");
        return -1;
    }
    if(XAxiDma_HasSg(&AxiDma)){
        printf("DMA is in SG mode, this code expects Simple mode.\n");
        return -1;
    }

    // 32x32 buffers
    static float A32[N32*N32]   __attribute__((aligned(64)));
    static float B32[N32*N32]   __attribute__((aligned(64)));
    static float Csw[N32*N32]   __attribute__((aligned(64)));
    static float Chw[N32*N32]   __attribute__((aligned(64)));

    // DMA buffers (per 16x16 call)
    static float in_buf[512]    __attribute__((aligned(64)));
    static float out_buf[256]   __attribute__((aligned(64)));

    // Local 16x16 staging
    static float A16[256]       __attribute__((aligned(64)));
    static float B16[256]       __attribute__((aligned(64)));

    // init (예시 패턴)
    for(int i=0;i<N32;i++){
        for(int j=0;j<N32;j++){
            A32[idx32(i,j)] = (float)i + (float)j*0.01f;
            B32[idx32(i,j)] = (float)j + (float)i*0.02f;
        }
    }

    //---------------- SW 32x32 ----------------
    XTime t0,t1;
    XTime_GetTime(&t0);
    gemm32_sw(A32,B32,Csw);
    XTime_GetTime(&t1);
    double sw_us = cycles_to_us(t1-t0);

    //---------------- HW tiled 32x32 ----------------
    XTime_GetTime(&t0);
    int rc = gemm32_hw_tiled(A32,B32,Chw, in_buf,out_buf, A16,B16);
    XTime_GetTime(&t1);
    if(rc != 0){
        printf("HW tiled GEMM failed.\n");
        return -1;
    }
    double hw_us = cycles_to_us(t1-t0);

    //---------------- Performance ----------------
    // 32x32 GEMM FLOPs = 2*N^3
    double flops32 = 2.0 * (double)N32 * (double)N32 * (double)N32;

    // 가속기 호출 횟수 NB^3 = 8 (참고용)
    int hw_calls = NB*NB*NB;

    printf("\nPerformance (32x32)\n");
    printf("SW time        : %.3f us\n", sw_us);
    printf("HW tiled time  : %.3f us  (calls=%d)\n", hw_us, hw_calls);
    printf("Speedup        : %.3f x\n", sw_us/hw_us);
    printf("SW GFLOPS      : %.6f\n", flops32/(sw_us*1e-6)/1e9);
    printf("HW tiled GFLOPS: %.6f\n", flops32/(hw_us*1e-6)/1e9);

    // 원하면 call당 평균 시간도 출력
    printf("Avg per 16x16 call (incl. PS accumulate): %.3f us\n", hw_us / (double)hw_calls);

    //---------------- Error ----------------
    error_stats_32(Csw,Chw);

    printf("\nDone.\n");
    return 0;
}
