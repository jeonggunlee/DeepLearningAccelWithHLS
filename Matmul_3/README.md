## Matmul_3:

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

