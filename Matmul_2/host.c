/********************************************************************
 * Production Host Code
 *  - Zynq-7000 (Zybo Z7-20)
 *  - AXI DMA + HLS gemm16_accel (512-float input)
 *  - SW reference + HW + timing + GFLOPS + error analysis
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
#define N 16
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
static inline int idx(int r,int c){ return r*N+c; }

// ================= SW GEMM =================
static void gemm_sw(const float*A,const float*B,float*C){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++)
                s+=A[idx(i,k)]*B[idx(k,j)];
            C[idx(i,j)]=s;
        }
}

// ================= DMA =================
static int run_dma(float *in_buf, float *out_buf){

    const int in_bytes  = 512*sizeof(float);
    const int out_bytes = 256*sizeof(float);

    cache_flush(in_buf,in_bytes);
    cache_inv(out_buf,out_bytes);

    XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)out_buf,out_bytes,XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)in_buf ,in_bytes ,XAXIDMA_DMA_TO_DEVICE);

    int t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma,XAXIDMA_DMA_TO_DEVICE)&&t--);
    if(t<=0){ printf("MM2S timeout\n"); return -1;}

    t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma,XAXIDMA_DEVICE_TO_DMA)&&t--);
    if(t<=0){ printf("S2MM timeout\n"); return -1;}

    cache_inv(out_buf,out_bytes);
    return 0;
}

// ================= ERROR =================
static void error_stats(const float*ref,const float*hw){

    float max_abs=0, max_rel=0;
    double err2=0, ref2=0;

    for(int i=0;i<N*N;i++){
        float e=hw[i]-ref[i];
        float a=fabsf(e);

        if(a>max_abs) max_abs=a;

        float rel=a/(fabsf(ref[i])+EPS);
        if(rel>max_rel) max_rel=rel;

        err2+=e*e;
        ref2+=ref[i]*ref[i];
    }

    printf("\nError Stats\n");
    printf("max_abs   = %.8f\n",max_abs);
    printf("max_rel   = %.8f\n",max_rel);
    printf("rmse      = %.8f\n",sqrt(err2/(N*N)));
    printf("rel_frob  = %.8f\n",sqrt(err2)/sqrt(ref2));
}

// ================= MAIN =================
int main(){

    printf("\n===== GEMM16 Batched DMA Benchmark =====\n");

    // DMA init
    XAxiDma_Config* cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    XAxiDma_CfgInitialize(&AxiDma,cfg);

    // buffers
    static float A[256]  __attribute__((aligned(64)));
    static float B[256]  __attribute__((aligned(64)));
    static float Csw[256] __attribute__((aligned(64)));
    static float Chw[256] __attribute__((aligned(64)));

    static float in_buf[512]  __attribute__((aligned(64)));
    static float out_buf[256] __attribute__((aligned(64)));

    // init
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            A[idx(i,j)] = i + j*0.1f;
            B[idx(i,j)] = j + i*0.2f;
        }

    //---------------- SW ----------------
    XTime t0,t1;
    XTime_GetTime(&t0);
    gemm_sw(A,B,Csw);
    XTime_GetTime(&t1);

    double sw_us = cycles_to_us(t1-t0);

    //---------------- HW ----------------
    memcpy(in_buf, A, 256*sizeof(float));
    memcpy(in_buf+256, B, 256*sizeof(float));

    XTime_GetTime(&t0);
    run_dma(in_buf,out_buf);
    XTime_GetTime(&t1);

    double hw_us = cycles_to_us(t1-t0);

    memcpy(Chw,out_buf,256*sizeof(float));

    //---------------- Performance ----------------
    double flops = 2.0*N*N*N;

    printf("\nPerformance\n");
    printf("SW time  : %.3f us\n",sw_us);
    printf("HW time  : %.3f us\n",hw_us);
    printf("Speedup  : %.3f x\n",sw_us/hw_us);
    printf("SW GFLOPS: %.6f\n", flops/(sw_us*1e-6)/1e9);
    printf("HW GFLOPS: %.6f\n", flops/(hw_us*1e-6)/1e9);

    //---------------- Error ----------------
    error_stats(Csw,Chw);

    printf("\nDone.\n");
    return 0;
}
