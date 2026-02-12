## Matmul_3:

<img width="1416" height="408" alt="image" src="https://github.com/user-attachments/assets/4573c514-2652-460c-b991-4c35a8d246ff" />

이전 가속기 설계와 달리 TLAST 생성을 위한 HDL code를 사용하여 block design 과정에서 통합.

## 성능평가:
===== GEMM32 via GEMM16 AXI-DMA Benchmark =====

### Performance (32x32)
- SW time        : 2824.869 us
- HW tiled time  : 861.894 us  (calls=8)
- Speedup        : 3.278 x
- SW GFLOPS      : 0.023200
- HW tiled GFLOPS: 0.076037
- Avg per 16x16 call (incl. PS accumulate): 107.737 us

### Error Stats (32x32)
- max_abs   = 0.00585938
- max_rel   = 0.00000028
- rmse      = 0.00109975
- rel_frob  = 0.00000010

