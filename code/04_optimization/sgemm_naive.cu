// =====================================================================
// 04_optimization / sgemm_naive.cu
// 矩阵乘 naive 版：每个线程算 1 个输出元素 C[row][col]。
//
// 编译 / 运行（远端 Win10，GTX 1080，CUDA 11.6）：
//   nvcc sgemm_naive.cu -o sgemm_naive -arch=sm_61
//   .\sgemm_naive.exe [M] [N] [K]    # 可选边长，默认 1024
//
// 对应文档：docs/04_performance.md §3.1
//
// 为什么慢（docs/04_performance.md §3.1）：
//   每个线程算 1 个输出，要读 A 的 K 个、B 的 K 个 = 2K 次全局内存。
//   A[i][k] 被 N 个线程重复读、B[k][j] 被 M 个线程重复读 —— 数据复用为零，
//   算术强度 ≈ 0.25 FLOP/byte，严重 memory-bound。
//   用 ncu 测：Achieved Occupancy 100%，但 DRAM Throughput ~95%、SM ~10%。
// =====================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err_));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                   \
    } while (0)

// 每个线程负责一个输出元素 C[row][col]
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 输出行
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 输出列
    if (row < M && col < N) {
        float sum = 0.f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];    // 每个 k 读两次全局内存
        C[row * N + col] = sum;
    }
}

// CPU 参考：C = A * B
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(bytesA);
    float* h_B = (float*)malloc(bytesB);
    float* h_C = (float*)malloc(bytesC);
    float* h_R = (float*)malloc(bytesC);
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)(rand() % 1000)) / 1000.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)(rand() % 1000)) / 1000.0f;

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, bytesA));
    CHECK(cudaMalloc(&d_B, bytesB));
    CHECK(cudaMalloc(&d_C, bytesC));
    CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    matmul_cpu(h_A, h_B, h_R, M, N, K);
    double max_err = 0.0;
    for (int i = 0; i < M * N; i++) {
        double e = fabs((double)h_C[i] - (double)h_R[i]);
        if (e > max_err) max_err = e;
    }

    printf("M=%d N=%d K=%d, grid(%d,%d), block(%d,%d)\n", M, N, K,
           grid.x, grid.y, block.x, block.y);
    printf("naive kernel 耗时 : %.3f ms\n", ms);
    printf("最大误差          : %g  =>  %s\n", max_err,
           max_err < 1e-3 ? "校验通过" : "校验失败");

    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_R);
    return 0;
}
