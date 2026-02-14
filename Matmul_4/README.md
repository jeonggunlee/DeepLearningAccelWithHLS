## Matmul_4:

<img width="1387" height="429" alt="image" src="https://github.com/user-attachments/assets/0462b0c2-7314-4a80-938f-f5e158abdfe1" />

## 성능평가:

===== GEMM (N=32) correct Ktiles protocol =====
- SW 2828.307 us
- HW 689.484 us
- Speedup 4.10x (이전 Speedup 3.85x)
- GFLOPS 0.095

===== GEMM (N=64) correct Ktiles protocol =====
- SW 22321.499 us
- HW 4584.267 us
- Speedup 4.87x (이전 Speedup 4.69x)
- GFLOPS 0.114

===== GEMM (N=128) correct Ktiles protocol =====
- SW 180529.701 us
- HW 33089.075 us
- Speedup 5.46x (이전 Speedup 5.35x)
- GFLOPS 0.127

===== GEMM (N=256) correct Ktiles protocol =====
- SW 1447190.782 us
- HW 250098.724 us
- Speedup 5.79x (이전 Speedup 5.72x)
- GFLOPS 0.134

===== GEMM (N=512) correct Ktiles protocol =====
- SW 17396005.281 us
- HW 1963059.060 us
- Speedup 8.86x (이전 Speedup 8.81x)
- GFLOPS 0.137

===== GEMM (N=768) correct Ktiles protocol =====
- SW 62876480.216 us
- HW 6956165.734 us
- Speedup 9.04x (이전 Speedup 9.01x)
- GFLOPS 0.130




# HLS 설계 (ChatGPT)
✅ 전체 아키텍처 한눈 요약
- stream pipeline 구조

