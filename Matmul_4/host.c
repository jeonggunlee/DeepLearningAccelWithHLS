/********************************************************************
 * SAFE Generic GEMM Host (Correct protocol for Ktiles-accum IP)
 *  - N = 16*k
 *  - Tile (bi,bj):
 *      S2MM (256 floats) ONCE
 *      MM2S (512 floats) Ktiles times
 ********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xtime_l.h"
#include "xil_io.h"

#define N 32              // 16의 배수
#define TILE 16           // 가속기 자체는 16*16 행렬 곱셈 & 누적
#define NB (N/TILE)       // Tile의 수
#define KTILES NB         // Tile의 수 (한 차원 측면에서)

#define MAXN 256*3        // 최대 행렬의 크기
#define DMA_DEV_ID XPAR_AXIDMA_0_DEVICE_ID
#define GEMM_CTRL_BASE XPAR_GEMM16_ACCUM_AXIS_0_S_AXI_CTRL_BASEADDR

#define REG_AP_CTRL  0x00
#define REG_KTILES   0x10    // Tile의 수를 가속기에 제공하여 가속기 내부에서 KTILES번 곱셈누적하도록 함.

#define DMA_TIMEOUT 100000000
#define EPS 1e-6f

static XAxiDma AxiDma;

static inline int idx(int r,int c){ return r*N+c; }         // 입력 행렬의 주소 index 반환
static inline int idx16(int r,int c){ return r*TILE+c; }    // tile 행렬의 주소 index 반환

static inline double cycles_to_us(XTime c){
    return (double)c * 2.0 * 1e6 / XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ;
}

static void flush(void* p,int sz){ Xil_DCacheFlushRange((UINTPTR)p,sz); }    // Cache Flush for READs
static void inval(void* p,int sz){ Xil_DCacheInvalidateRange((UINTPTR)p,sz); }    // Cache Invalidate for WRITEs

// ---------------- SW GEMM ----------------
void gemm_sw(float*A,float*B,float*C){
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<N;k++)
                s+=A[idx(i,k)]*B[idx(k,j)];
            C[idx(i,j)]=s;
        }
}

// ---------------- Tile helpers ----------------
void extract_block(float*src,int br,int bc,float*dst){
    int r0=br*TILE, c0=bc*TILE;
    for(int i=0;i<TILE;i++)
        for(int j=0;j<TILE;j++)
            dst[idx16(i,j)] = src[idx(r0+i,c0+j)];
}

void store_block(float*dst,int br,int bc,float*src){
    int r0=br*TILE, c0=bc*TILE;
    for(int i=0;i<TILE;i++)
        for(int j=0;j<TILE;j++)
            dst[idx(r0+i,c0+j)] = src[idx16(i,j)];
}

// ---------------- DMA helpers ----------------
// MM2S: send 512 floats (2KB)
static int dma_send_frame(float *in512){
    const int in_bytes = 512*sizeof(float);        // 16*16 행렬 2개: 16*16*2 = 512
    flush(in512, in_bytes);

    if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)in512, in_bytes, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
        return -1;

    int t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE) && t--);
    return (t<=0) ? -1 : 0;
}

// S2MM: receive 256 floats (1KB) - tile당 1번만!
static int dma_recv_tile(float *out256){
    const int out_bytes = 256*sizeof(float);        // 출력 행렬 1개: 16*16 = 256
    inval(out256, out_bytes);

    if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)out256, out_bytes, XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
        return -1;

    return 0;
}

static int dma_wait_recv_done(void){
    int t=DMA_TIMEOUT;
    while(XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA) && t--);
    return (t<=0) ? -1 : 0;
}

int main(){
    printf("\n===== GEMM (N=%d) correct Ktiles protocol =====\n", N);

    XAxiDma_Config* cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    XAxiDma_CfgInitialize(&AxiDma,cfg);

    static float A[MAXN*MAXN] __attribute__((aligned(64)));
    static float B[MAXN*MAXN] __attribute__((aligned(64)));
    static float Csw[MAXN*MAXN] __attribute__((aligned(64)));
    static float Chw[MAXN*MAXN] __attribute__((aligned(64)));

    static float frame_buf[512] __attribute__((aligned(64)));    // Two tile matrices (A16*16, B16*16)
    static float out_buf[256]   __attribute__((aligned(64)));    // One tile matrix (C16*16)

    static float A16[256], B16[256];

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){
            A[idx(i,j)] = i + j*0.1f;
            B[idx(i,j)] = j + i*0.2f;
        }

    // SW
    XTime t0,t1;
    XTime_GetTime(&t0);
    gemm_sw(A,B,Csw);
    XTime_GetTime(&t1);
    double sw_us=cycles_to_us(t1-t0);

    // HW
    Xil_Out32(GEMM_CTRL_BASE+REG_KTILES, KTILES);

    XTime_GetTime(&t0);

    for(int bi=0; bi<NB; bi++){                // NB: 한 축으로의 tile의 수
        for(int bj=0; bj<NB; bj++){            // NB: 한 축으로의 tile의 수

            // (1) 타일 출력 S2MM을 먼저 1회만 걸어둔다
            if(dma_recv_tile(out_buf)!=0){
                printf("S2MM submit fail\n");
                return -1;
            }

            // (2) IP start
            Xil_Out32(GEMM_CTRL_BASE+REG_AP_CTRL, 1);

            // (3) Ktiles 프레임을 MM2S로 연속 전송 (각 512 floats)
            for(int bk=0; bk<NB; bk++){
                extract_block(A, bi, bk, A16);
                extract_block(B, bk, bj, B16);

                memcpy(&frame_buf[0],   A16, 256*sizeof(float));
                memcpy(&frame_buf[256], B16, 256*sizeof(float));

                if(dma_send_frame(frame_buf)!=0){
                    printf("MM2S frame send fail\n");
                    return -1;
                }
            }

            // (4) S2MM 완료 대기 (여기서 out_buf 채워짐)
            if(dma_wait_recv_done()!=0){
                printf("S2MM wait timeout\n");
                return -1;
            }

            // (5) IP done도 확인(안전)
            while(!(Xil_In32(GEMM_CTRL_BASE+REG_AP_CTRL) & 0x2));

            // (6) 타일 저장
            inval(out_buf, 256*sizeof(float));
            store_block(Chw, bi, bj, out_buf);
        }
    }

    XTime_GetTime(&t1);
    double hw_us=cycles_to_us(t1-t0);

    double flops = 2.0 * (double)N * (double)N * (double)N;

    printf("SW %.3f us\n", sw_us);
    printf("HW %.3f us\n", hw_us);
    printf("Speedup %.2fx\n", sw_us/hw_us);
    printf("GFLOPS %.3f\n", flops/(hw_us*1e-6)/1e9);

    return 0;
}
