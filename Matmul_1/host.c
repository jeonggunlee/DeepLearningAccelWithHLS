//============================================================
// production_gemm8_final.c
// Zynq-7000 + AXI DMA + HLS GEMM8 benchmark (final version)
//============================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xtime_l.h"

//==================== CONFIG ====================
#define N16 16
#define TS  8
#define DMA_DEV_ID XPAR_AXIDMA_0_DEVICE_ID

#define CACHELINE 32
#define DMA_TIMEOUT 100000000
#define EPS 1e-6f

//==================== GLOBAL ====================
static XAxiDma AxiDma;

//==================== INDEX ====================
static inline int idx(int r,int c){ return r*N16+c; }
static inline int idx8(int r,int c){ return r*TS+c; }

//==================== CACHE ====================
static void cache_flush(void* p, int sz){
    UINTPTR a=(UINTPTR)p;
    a&=~(CACHELINE-1);
    sz=(sz+CACHELINE-1)&~(CACHELINE-1);
    Xil_DCacheFlushRange(a,sz);
}
static void cache_inv(void* p,int sz){
    UINTPTR a=(UINTPTR)p;
    a&=~(CACHELINE-1);
    sz=(sz+CACHELINE-1)&~(CACHELINE-1);
    Xil_DCacheInvalidateRange(a,sz);
}

//==================== TIMER ====================
// Zynq-7000: XTime tick = CPU/2  → 반드시 ×2
static double cycles_to_us(XTime cyc){
    double freq = (double)XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ;
    return ((double)cyc * 2.0 * 1e6) / freq;
}

//==================== SW GEMM ====================
static void gemm_sw(const float*A,const float*B,float*C){
    for(int i=0;i<N16;i++)
        for(int j=0;j<N16;j++){
            float s=0;
            for(int k=0;k<N16;k++)
                s+=A[idx(i,k)]*B[idx(k,j)];
            C[idx(i,j)]=s;
        }
}

//==================== TILE ====================
static void extract8(const float*A,float*T,int r0,int c0){
    for(int i=0;i<TS;i++)
        for(int j=0;j<TS;j++)
            T[idx8(i,j)]=A[idx(r0+i,c0+j)];
}
static void store8(float*C,const float*T,int r0,int c0){
    for(int i=0;i<TS;i++)
        for(int j=0;j<TS;j++)
            C[idx(r0+i,c0+j)]=T[idx8(i,j)];
}

//==================== DMA CALL ====================
static int run_dma(float*in,float*out){
    int inb=192*sizeof(float);
    int outb=64*sizeof(float);

    cache_flush(in,inb);
    cache_inv(out,outb);

    XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)out,outb,XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR)in ,inb ,XAXIDMA_DMA_TO_DEVICE);

    int t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma,XAXIDMA_DMA_TO_DEVICE)&&t--);
    if(t<=0){ printf("MM2S timeout\n"); return -1;}

    t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma,XAXIDMA_DEVICE_TO_DMA)&&t--);
    if(t<=0){ printf("S2MM timeout\n"); return -1;}

    cache_inv(out,outb);
    return 0;
}

//==================== ERROR ====================
static void error_stats(const float*ref,const float*hw){
    float max_abs=0, max_rel=0;
    double err2=0, ref2=0;

    for(int i=0;i<N16*N16;i++){
        float e=hw[i]-ref[i];
        float a=fabsf(e);
        if(a>max_abs) max_abs=a;
        float rel=a/(fabsf(ref[i])+EPS);
        if(rel>max_rel) max_rel=rel;
        err2+=e*e;
        ref2+=ref[i]*ref[i];
    }

    float rmse=sqrt(err2/(N16*N16));
    float relf=sqrt(err2)/sqrt(ref2);

    printf("\nError Stats\n");
    printf("max_abs  = %.8f\n",max_abs);
    printf("max_rel  = %.8f\n",max_rel);
    printf("rmse     = %.8f\n",rmse);
    printf("rel_frob = %.8f\n",relf);
}

//==================== MAIN ====================
int main(){

    printf("\n==== GEMM8 Production Benchmark ====\n");

    // DMA init
    XAxiDma_Config*cfg=XAxiDma_LookupConfig(DMA_DEV_ID);
    XAxiDma_CfgInitialize(&AxiDma,cfg);

    // matrices
    static float A[256] __attribute__((aligned(64)));
    static float B[256] __attribute__((aligned(64)));
    static float Csw[256] __attribute__((aligned(64)));
    static float Chw[256] __attribute__((aligned(64)));

    for(int i=0;i<N16;i++)
        for(int j=0;j<N16;j++){
            A[idx(i,j)]=i+j*0.1f;
            B[idx(i,j)]=j+i*0.2f;
        }

    //---------------- SW ----------------
    XTime t0,t1;
    XTime_GetTime(&t0);
    gemm_sw(A,B,Csw);
    XTime_GetTime(&t1);

    double sw_us=cycles_to_us(t1-t0);

    //---------------- HW ----------------
    float A8[64],B8[64],Ct[64];
    static float in[192] __attribute__((aligned(64)));
    static float out[64] __attribute__((aligned(64)));

    XTime_GetTime(&t0);

    for(int ti=0;ti<2;ti++)
        for(int tj=0;tj<2;tj++){
            for(int x=0;x<64;x++) Ct[x]=0;
            for(int tk=0;tk<2;tk++){

                extract8(A,A8,ti*8,tk*8);
                extract8(B,B8,tk*8,tj*8);

                memcpy(in,A8,64*sizeof(float));
                memcpy(in+64,B8,64*sizeof(float));
                memcpy(in+128,Ct,64*sizeof(float));

                run_dma(in,out);
                memcpy(Ct,out,64*sizeof(float));
            }
            store8(Chw,Ct,ti*8,tj*8);
        }

    XTime_GetTime(&t1);
    double hw_us=cycles_to_us(t1-t0);

    //---------------- Performance ----------------
    double flops=2.0*N16*N16*N16;

    printf("\nPerformance\n");
    printf("SW  time  : %.3f us\n",sw_us);
    printf("HW  time  : %.3f us\n",hw_us);
    printf("Speedup   : %.3f x\n",sw_us/hw_us);

    printf("SW GFLOPS : %.6f\n", flops/(sw_us*1e-6)/1e9);
    printf("HW GFLOPS : %.6f\n", flops/(hw_us*1e-6)/1e9);

    //---------------- Error ----------------
    error_stats(Csw,Chw);

    printf("\nDone.\n");
    return 0;
}
