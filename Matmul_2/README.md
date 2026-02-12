## Matmul_2:

16x16 행렬곱 가속기

문제점:
- tiling을 하는 경우 누적을 host code에서 수행해야함.
- 따라서 tiling된 C tile을 매번 DMA를 통해서 전송하고 받고 해야하는 담점이 있음.

## 성능 평가:
===== GEMM16 Batched DMA Benchmark =====

### Performance
- SW time  : 366.255 us
- HW time  : 42.018 us
- Speedup  : 8.717 x
- SW GFLOPS: 0.022367
- HW GFLOPS: 0.194964
