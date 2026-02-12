## Matmul_3:

<img width="1416" height="408" alt="image" src="https://github.com/user-attachments/assets/4573c514-2652-460c-b991-4c35a8d246ff" />

이전 가속기 설계와 달리 TLAST 생성을 위한 HDL code를 사용하여 block design 과정에서 통합.

## 성능평가:

===== GEMM (N=32) correct Ktiles protocol =====
- SW 2828.352 us
- HW 733.920 us
- Speedup 3.85x
- GFLOPS 0.089

===== GEMM (N=64) correct Ktiles protocol =====
- SW 22320.581 us
- HW 4755.468 us
- Speedup 4.69x
- GFLOPS 0.110

===== GEMM (N=128) correct Ktiles protocol =====
- SW 180532.791 us
- HW 33771.578 us
- Speedup 5.35x
- GFLOPS 0.124

===== GEMM (N=256) correct Ktiles protocol =====
- SW 1447176.529 us
- HW 252812.164 us
- Speedup 5.72x
- GFLOPS 0.133

===== GEMM (N=512) correct Ktiles protocol =====
- SW 17394293.061 us
- HW 1974236.946 us
- Speedup 8.81x
- GFLOPS 0.136

===== GEMM (N=768) correct Ktiles protocol =====
- SW 62862735.078 us
- HW 6980534.175 us
- Speedup 9.01x
- GFLOPS 0.130
