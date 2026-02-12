# Deep Learning Acceleration with LLM generated HLS codes

HLS를 이용한 딥러닝 가속 - 행렬곱 가속기 설계 및 딥러닝에 적용

### Matmul1
8x8 행렬곱을 이용한 16x16 행렬곱 설계.
- 문제점: tiling되는 경우 누적이 host에서 진행되기 때문에 tiling된 C 행렬의 partial result가 지속적으로 DMA를 통해 PS<->PL간 이동해야함.
- "계산은 PL, 누적은 PS → DMA 병목 구조"
```
A_tile, B_tile → DMA → PL
PL 계산 → partial C → DMA → PS
PS에서 누적
```

### Matmul2
16x16 행렬곱을 이용한 32x32 행렬곱 설계.
- 문제점: tiling되는 경우 누적이 host에서 진행되기 때문에 tiling된 C 행렬의 partial result가 지속적으로 DMA를 통해 PS<->PL간 이동해야함.
- "계산은 PL, 누적은 PS → DMA 병목 구조"
```
A_tile, B_tile → DMA → PL
PL 계산 → partial C → DMA → PS
PS에서 누적
```

### Matmul3
16x16 행렬곱 가속기를 이용하여 임의의 NxN (N은 32의 배수) 행렬 곱을 수행.
- tiling된 C행렬의 부분값을 가속기 내부에 keep하면 누적 -> 이전 설계에서 발생했던 행렬곱 누적값의 PS <-> PL간 이동이 최소화됨.
- "계산 + 누적 모두 PL → 온칩 처리"
- partial C 전송 제거 -> 최종 1회만 DMA 

```
C_local = 0
for k:
    C_local += A_tile × B_tile   (PL 내부)
```

<img width="831" height="439" alt="image" src="https://github.com/user-attachments/assets/9d98b95d-54ab-43db-a864-8c9550d12c76" />
