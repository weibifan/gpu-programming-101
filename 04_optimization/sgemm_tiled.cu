// =====================================================================
// 04_optimization / sgemm_tiled.cu
// 矩阵乘 tiling + register tiling 版：
//   每 block 覆盖 64x64 输出（BM x BN），每线程用寄存器算 4x4 个输出。
//
// 编译 / 运行（远端 Win10，GTX 1080，CUDA 11.6）：
//   nvcc sgemm_tiled.cu -o sgemm_tiled -arch=sm_61
//   .\sgemm_tiled.exe [M] [N] [K]    # 可选边长，默认 1024
//
// 对应文档：docs/03_cuda_advanced.md §10.3（register tiling）
//
// 思路：shared 版每线程只算 1 个输出，读写共享内存也有开销。
//       这里让每个线程算 4x4 = 16 个输出：从 As/Bs 读一个元素，可参与多次累加
//       （数据先留在寄存器里反复用），共享内存带宽压力再降一个数量级。
//
// 参数：
//   BM = BN = 64  每 block 负责的输出小块（行 x 列）
//   BK = 32       每轮搬进共享内存的 K 切片
//   blockDim = (16, 16) = 256 线程，每线程 4x4 输出 -> 16x16x16 = 64x64
//   As[64][33]  Bs[32][65]  行长 +1 padding，防 bank conflict
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

#define BM 64
#define BN 64
#define BK 32

__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x, ty = threadIdx.y;              // 16x16 线程
    int bx = blockIdx.x,  by = blockIdx.y;

    float cc[4][4] = {{0.f}};                            // 寄存器里的 16 个累加器

    for (int kt = 0; kt < K; kt += BK) {
        // ---- 1) 搬 As 小块：BM x BK，共 2048 个 float，256 线程均摊 8 个 ----
        for (int idx = ty * blockDim.x + tx; idx < BM * BK; idx += blockDim.x * blockDim.y) {
            int r = idx / BK, c = idx % BK;              // r:0..63  c:0..31
            int gr = by * BM + r, gc = kt + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // ---- 2) 搬 Bs 小块：BK x BN，同样 2048 个 float ----
        for (int idx = ty * blockDim.x + tx; idx < BK * BN; idx += blockDim.x * blockDim.y) {
            int r = idx / BN, c = idx % BN;              // r:0..31  c:0..63
            int gr = kt + r, gc = bx * BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();

        // ---- 3) 每线程算 4x4 输出：As/Bs 的元素读到寄存器后反复用 ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[4] = { As[ty*4+0][kk], As[ty*4+1][kk], As[ty*4+2][kk], As[ty*4+3][kk] };
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    cc[i][j] += a[i] * Bs[kk][tx*4+j];
        }
        __syncthreads();
    }

    // ---- 4) 写回 4x4 输出 ----
    int row = by * BM + ty * 4, col = bx * BN + tx * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            if (row + i < M && col + j < N)
                C[(row + i) * N + (col + j)] = cc[i][j];
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
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("tiled kernel 耗时: %.3f ms\n", ms);
    printf("最大误差        : %g  =>  %s\n", max_err,
           max_err < 1e-3 ? "校验通过" : "校验失败");

    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_R);
    return 0;
}
