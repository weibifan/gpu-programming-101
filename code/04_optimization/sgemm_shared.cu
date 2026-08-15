// =====================================================================
// 04_optimization / sgemm_shared.cu
// 矩阵乘 shared 版：把 A/B 的 32x32 小块拷进共享内存，块内反复复用。
//
// 编译 / 运行（远端 Win10，GTX 1080，CUDA 11.6）：
//   nvcc sgemm_shared.cu -o sgemm_shared -arch=sm_61
//   .\sgemm_shared.exe [M] [N] [K]    # 可选边长，默认 1024
//
// 对应文档：docs/04_performance.md §3.2 + §4（bank conflict / __syncthreads）
//
// 思路（docs/03_memory.md §3.2）：
//   每个 block 负责输出 C 的一个 TILE x TILE 小块：
//   1. 循环 K/TILE 次：把 A 的 TILE 小块拷进 As、B 的拷进 Bs（合并访问）
//   2. __syncthreads()  ->  3. 块内线程从 As/Bs 算累加（读共享内存，快）
//   4. __syncthreads() 进入下一轮
//
// 行长用 TILE+1（padding）：行长 32 恰好是 bank 数，会撞 bank；33 天然错开
// （docs/04_performance.md §4.1）。
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

#define TILE 32

// 每个线程算 1 个输出；As/Bs 行长 TILE+1，防 bank conflict
__global__ void sgemm_shared(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;   // 本 block 负责的输出行
    int col = blockIdx.x * TILE + threadIdx.x;   // 本 block 负责的输出列
    float sum = 0.f;

    for (int kt = 0; kt < K; kt += TILE) {
        // ---- 1) 搬 A 的小块到共享内存（合并访问）----
        int a_row = row, a_col = kt + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.f;
        // ---- 搬 B 的小块到共享内存（合并访问）----
        int b_row = kt + threadIdx.y, b_col = col;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.f;

        // ---- 2) 等全 block 拷完再算 ----
        __syncthreads();

        // ---- 3) 从共享内存算累加（不再读全局内存）----
        for (int kk = 0; kk < TILE; kk++)
            sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];

        // ---- 4) 算完再进下一轮，防止下轮覆盖还没被读完的 tile ----
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
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

    dim3 block(TILE, TILE);                                  // 32x32 = 1024 线程
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    sgemm_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("shared kernel 耗时: %.3f ms\n", ms);
    printf("最大误差          : %g  =>  %s\n", max_err,
           max_err < 1e-3 ? "校验通过" : "校验失败");

    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_R);
    return 0;
}
