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



# HLS 설계 (ChatGPT)
✅ 전체 아키텍처 한눈 요약
```
동작 모델
(Ktiles 번 반복)
   A(16x16) 수신
   B(16x16) 수신
   C += A×B   ← 내부 누적

마지막에
   C(16x16) 한 번만 출력
```

즉:

⭐ 특징

✔ Block GEMM

✔ 내부 partial-sum 누적

✔ Output 1회만 전송 (DMA 효율 ↑)

## 🔷 핵심 특징 4가지

### ① AXI-Stream 기반 순수 스트리밍 커널

```
#pragma HLS INTERFACE axis port=s_in
#pragma HLS INTERFACE axis port=s_out
```

의미

- DMA와 직접 연결
- DDR → AXIS → Kernel → AXIS → DDR
- memcpy/BRAM copy 없음

⭐ Zynq에 최적
- PS–PL streaming 아키텍처에 딱 맞음

### ② Ktiles 내부 누적 구조 (가장 중요)
```
for (kt=0; kt<Ktiles; kt++)
    C[i][j] += A×B;
```

의미
- Host가 C partial sum 관리 안 함
- 커널 내부에서 누적
- 기존 방식(나쁜 구조)
- 👉 DMA 호출 폭증
```
for bk:
   DMA → compute → DMA → compute → DMA ...
```


현재 구조(좋은 구조)
- 마지막에 DMA(out) 1회
- 👉 DMA 호출 수 Ktiles배 감소

```
for bk:
   DMA(in) only
```

효과
- ⭐ DMA 횟수 감소
- 기존: tile × Ktiles × 2
- 현재: tile × 2
- ⭐ 성능 대폭 향상

### ③ ARRAY_PARTITION 완전 병렬화
```
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2
```
의미
```
A
A[i][k] → k 방향 병렬 읽기

B
B[k][j] → k 방향 병렬 읽기
```

효과
- ⭐ 한 cycle에 8~16개 곱셈 가능
- 👉 이 pragma가 성능의 핵심

### ④ 수동 Adder Tree (Timing-safe 핵심 설계)
일반 naive 방식
```
sum += A[i][k]*B[k][j];
```

문제
- sum → sum → sum → sum ...
- Ripple chain
- 결과
  - critical path 길어짐
  - Slack -2 ~ -5ns 발생
  - Timing violation

현재 코드
- reduce8_tree()
- 구조:
```
8 mul
 → 4 add
 → 2 add
 → 1 add

Depth
log2(8)=3
```
