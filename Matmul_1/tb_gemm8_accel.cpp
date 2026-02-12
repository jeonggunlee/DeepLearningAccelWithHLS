#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "gemm8_accel.h"

using namespace std;

// ==================== 유틸리티 함수 ====================

// float -> ap_uint<32> 변환
static inline ap_uint<32> float_to_u32(float f) {
    union { unsigned int ui; float flt; } v;
    v.flt = f;
    return (ap_uint<32>)v.ui;
}

// ap_uint<32> -> float 변환
static inline float u32_to_float(ap_uint<32> u) {
    union { unsigned int ui; float flt; } v;
    v.ui = (unsigned int)u;
    return v.flt;
}

// AXI Stream 패킷 생성
static inline axis32_t make_axis_packet(float data, bool last = false) {
    axis32_t pkt;
    pkt.data = float_to_u32(data);
    pkt.keep = -1;
    pkt.strb = -1;
    pkt.user = 0;
    pkt.id   = 0;
    pkt.dest = 0;
    pkt.last = last ? 1 : 0;
    return pkt;
}

// 행렬 출력 함수
void print_matrix(const char* name, float mat[8][8]) {
    cout << "\n" << name << ":\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            cout << setw(10) << fixed << setprecision(4) << mat[i][j] << " ";
        }
        cout << "\n";
    }
}

// CPU에서 8x8 GEMM 계산 (참조용)
void gemm8_cpu(float A[8][8], float B[8][8], float C_in[8][8], float C_out[8][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float sum = C_in[i][j];
            for (int k = 0; k < 8; k++) {
                sum += A[i][k] * B[k][j];
            }
            C_out[i][j] = sum;
        }
    }
}

// 결과 비교 함수
bool compare_matrices(float expected[8][8], float actual[8][8], float tolerance = 1e-3) {
    bool pass = true;
    int error_count = 0;
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float diff = fabs(expected[i][j] - actual[i][j]);
            if (diff > tolerance) {
                if (error_count < 10) {  // 처음 10개 에러만 출력
                    cout << "Mismatch at [" << i << "][" << j << "]: "
                         << "Expected=" << expected[i][j] 
                         << ", Got=" << actual[i][j]
                         << ", Diff=" << diff << "\n";
                }
                error_count++;
                pass = false;
            }
        }
    }
    
    if (error_count > 0) {
        cout << "Total mismatches: " << error_count << " out of 64\n";
    }
    
    return pass;
}

// ==================== 테스트 케이스 ====================

// Test 1: 단위 행렬 테스트
bool test_identity() {
    cout << "\n========================================\n";
    cout << "Test 1: Identity Matrix Test\n";
    cout << "========================================\n";
    
    float A[8][8], B[8][8], C_in[8][8];
    float C_expected[8][8], C_actual[8][8];
    
    // A = Identity, B = Random, C_in = 0
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A[i][j] = (i == j) ? 1.0f : 0.0f;
            B[i][j] = (float)(i + j);
            C_in[i][j] = 0.0f;
        }
    }
    
    // CPU 참조 계산
    gemm8_cpu(A, B, C_in, C_expected);
    
    // HLS 입력/출력 스트림
    hls::stream<axis32_t> s_in;
    hls::stream<axis32_t> s_out;
    
    // A 전송
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(A[i][j]));
        }
    }
    
    // B 전송
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(B[i][j]));
        }
    }
    
    // C_in 전송
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            bool is_last = (i == 7 && j == 7);
            s_in.write(make_axis_packet(C_in[i][j], is_last));
        }
    }
    
    // HLS 함수 실행
    gemm8_accel(s_in, s_out);
    
    // 결과 읽기
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            axis32_t pkt = s_out.read();
            C_actual[i][j] = u32_to_float(pkt.data);
        }
    }
    
    // 결과 비교
    bool pass = compare_matrices(C_expected, C_actual);
    
    if (pass) {
        cout << "Test 1: PASSED\n";
    } else {
        cout << "Test 1: FAILED\n";
        print_matrix("Expected", C_expected);
        print_matrix("Actual", C_actual);
    }
    
    return pass;
}

// Test 2: 누적 테스트
bool test_accumulation() {
    cout << "\n========================================\n";
    cout << "Test 2: Accumulation Test\n";
    cout << "========================================\n";
    
    float A[8][8], B[8][8], C_in[8][8];
    float C_expected[8][8], C_actual[8][8];
    
    // 간단한 값으로 초기화
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
            C_in[i][j] = 10.0f;  // 초기값
        }
    }
    
    // CPU 참조 계산: C_out = C_in + A*B
    gemm8_cpu(A, B, C_in, C_expected);
    
    // HLS 입력/출력 스트림
    hls::stream<axis32_t> s_in;
    hls::stream<axis32_t> s_out;
    
    // 데이터 전송
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(A[i][j]));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(B[i][j]));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            bool is_last = (i == 7 && j == 7);
            s_in.write(make_axis_packet(C_in[i][j], is_last));
        }
    }
    
    // HLS 실행
    gemm8_accel(s_in, s_out);
    
    // 결과 읽기
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            axis32_t pkt = s_out.read();
            C_actual[i][j] = u32_to_float(pkt.data);
        }
    }
    
    // 검증
    bool pass = compare_matrices(C_expected, C_actual);
    
    if (pass) {
        cout << "Test 2: PASSED (Expected: 26.0, Got: " 
             << C_actual[0][0] << ")\n";
    } else {
        cout << "Test 2: FAILED\n";
        print_matrix("Expected", C_expected);
        print_matrix("Actual", C_actual);
    }
    
    return pass;
}

// Test 3: 랜덤 행렬 테스트
bool test_random() {
    cout << "\n========================================\n";
    cout << "Test 3: Random Matrix Test\n";
    cout << "========================================\n";
    
    float A[8][8], B[8][8], C_in[8][8];
    float C_expected[8][8], C_actual[8][8];
    
    // 랜덤 초기화
    srand(12345);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A[i][j] = (float)(rand() % 100) / 10.0f;
            B[i][j] = (float)(rand() % 100) / 10.0f;
            C_in[i][j] = (float)(rand() % 50) / 10.0f;
        }
    }
    
    print_matrix("A (sample)", A);
    print_matrix("B (sample)", B);
    
    // CPU 참조 계산
    gemm8_cpu(A, B, C_in, C_expected);
    
    // HLS 입력/출력 스트림
    hls::stream<axis32_t> s_in;
    hls::stream<axis32_t> s_out;
    
    // 데이터 전송
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(A[i][j]));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(B[i][j]));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            bool is_last = (i == 7 && j == 7);
            s_in.write(make_axis_packet(C_in[i][j], is_last));
        }
    }
    
    // HLS 실행
    gemm8_accel(s_in, s_out);
    
    // 결과 읽기
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            axis32_t pkt = s_out.read();
            C_actual[i][j] = u32_to_float(pkt.data);
        }
    }
    
    // 검증
    bool pass = compare_matrices(C_expected, C_actual, 1e-2);
    
    if (pass) {
        cout << "Test 3: PASSED\n";
    } else {
        cout << "Test 3: FAILED\n";
        print_matrix("Expected", C_expected);
        print_matrix("Actual", C_actual);
    }
    
    return pass;
}

// Test 4: 제로 행렬 테스트
bool test_zero() {
    cout << "\n========================================\n";
    cout << "Test 4: Zero Matrix Test\n";
    cout << "========================================\n";
    
    float A[8][8], B[8][8], C_in[8][8];
    float C_expected[8][8], C_actual[8][8];
    
    // 모두 0으로 초기화
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A[i][j] = 0.0f;
            B[i][j] = 0.0f;
            C_in[i][j] = 0.0f;
        }
    }
    
    // CPU 참조 계산
    gemm8_cpu(A, B, C_in, C_expected);
    
    // HLS 입력/출력 스트림
    hls::stream<axis32_t> s_in;
    hls::stream<axis32_t> s_out;
    
    // 데이터 전송
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(A[i][j]));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            s_in.write(make_axis_packet(B[i][j]));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            bool is_last = (i == 7 && j == 7);
            s_in.write(make_axis_packet(C_in[i][j], is_last));
        }
    }
    
    // HLS 실행
    gemm8_accel(s_in, s_out);
    
    // 결과 읽기
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            axis32_t pkt = s_out.read();
            C_actual[i][j] = u32_to_float(pkt.data);
        }
    }
    
    // 검증
    bool pass = compare_matrices(C_expected, C_actual);
    
    if (pass) {
        cout << "Test 4: PASSED\n";
    } else {
        cout << "Test 4: FAILED\n";
    }
    
    return pass;
}

// Test 5: 16x16 타일링 시뮬레이션
bool test_16x16_tiling() {
    cout << "\n========================================\n";
    cout << "Test 5: 16x16 Tiling Simulation\n";
    cout << "========================================\n";
    
    const int N16 = 16;
    const int TS = 8;
    
    float A16[N16][N16], B16[N16][N16], C16[N16][N16];
    float C_expected[N16][N16];
    
    // 초기화
    for (int i = 0; i < N16; i++) {
        for (int j = 0; j < N16; j++) {
            A16[i][j] = (float)(i + j * 0.1f);
            B16[i][j] = (float)(j + i * 0.2f);
            C16[i][j] = 0.0f;
            C_expected[i][j] = 0.0f;
        }
    }
    
    // CPU로 16x16 계산 (참조)
    for (int i = 0; i < N16; i++) {
        for (int j = 0; j < N16; j++) {
            for (int k = 0; k < N16; k++) {
                C_expected[i][j] += A16[i][k] * B16[k][j];
            }
        }
    }
    
    // HLS 가속기로 타일 단위 계산
    for (int ti = 0; ti < 2; ti++) {
        for (int tj = 0; tj < 2; tj++) {
            float Ct[8][8] = {0};
            
            for (int tk = 0; tk < 2; tk++) {
                float A8[8][8], B8[8][8];
                
                // 타일 추출
                for (int i = 0; i < TS; i++) {
                    for (int j = 0; j < TS; j++) {
                        A8[i][j] = A16[ti*TS + i][tk*TS + j];
                        B8[i][j] = B16[tk*TS + i][tj*TS + j];
                    }
                }
                
                // HLS 실행
                hls::stream<axis32_t> s_in;
                hls::stream<axis32_t> s_out;
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        s_in.write(make_axis_packet(A8[i][j]));
                    }
                }
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        s_in.write(make_axis_packet(B8[i][j]));
                    }
                }
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        bool is_last = (i == 7 && j == 7);
                        s_in.write(make_axis_packet(Ct[i][j], is_last));
                    }
                }
                
                gemm8_accel(s_in, s_out);
                
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        axis32_t pkt = s_out.read();
                        Ct[i][j] = u32_to_float(pkt.data);
                    }
                }
            }
            
            // 결과 저장
            for (int i = 0; i < TS; i++) {
                for (int j = 0; j < TS; j++) {
                    C16[ti*TS + i][tj*TS + j] = Ct[i][j];
                }
            }
        }
    }
    
    // 검증 (일부만)
    bool pass = true;
    int errors = 0;
    for (int i = 0; i < N16; i++) {
        for (int j = 0; j < N16; j++) {
            float diff = fabs(C16[i][j] - C_expected[i][j]);
            if (diff > 1e-2) {
                if (errors < 5) {
                    cout << "Mismatch at [" << i << "][" << j << "]: "
                         << "Expected=" << C_expected[i][j]
                         << ", Got=" << C16[i][j] << "\n";
                }
                errors++;
                pass = false;
            }
        }
    }
    
    if (pass) {
        cout << "Test 5: PASSED\n";
        cout << "Sample C16[0][0] = " << C16[0][0] << "\n";
        cout << "Sample C16[7][7] = " << C16[7][7] << "\n";
        cout << "Sample C16[15][15] = " << C16[15][15] << "\n";
    } else {
        cout << "Test 5: FAILED (errors: " << errors << ")\n";
    }
    
    return pass;
}

// ==================== 메인 함수 ====================
int main() {
    cout << "========================================\n";
    cout << "  8x8 GEMM Accelerator Testbench\n";
    cout << "========================================\n";
    
    int passed = 0;
    int total = 5;
    
    if (test_identity())        passed++;
    if (test_accumulation())    passed++;
    if (test_random())          passed++;
    if (test_zero())            passed++;
    if (test_16x16_tiling())    passed++;
    
    cout << "\n========================================\n";
    cout << "  Test Summary\n";
    cout << "========================================\n";
    cout << "Passed: " << passed << " / " << total << "\n";
    
    if (passed == total) {
        cout << "All tests PASSED!\n";
        return 0;
    } else {
        cout << "Some tests FAILED!\n";
        return 1;
    }
}
