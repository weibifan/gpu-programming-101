// =====================================================================
// 03_memory / matrix_transpose.cu
// 矩阵转置的"合并访问陷阱"：naive vs 共享内存中转
//
// 编译 / 运行（远端 Win10，GTX 1080，CUDA 11.6）：
//   nvcc matrix_transpose.cu -o matrix_transpose -arch=sm_61
//   .\matrix_transpose.exe [N]      # 可选矩阵边长，默认 2048
//
// 对应文档：docs/03_memory.md §9（矩阵转置的合并访问陷阱）。
//
// 结论：
//   写法 A(naive)  ：读 in 合并、写 out 不合并（out 下标跨行）-> 慢
//   写法 B(shared) ：经共享内存 tile 转一道，读、写两头都合并 -> 快
//
// 程序会打印两个版本的耗时与正确性校验，看"差多少倍"。
// =====================================================================
#include <stdio.h>
#include <stdlib.h>
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

// 写法 A：读合并、写不合并。
// 线程 (row, col) 读 in[row][col]（col 连续 -> 合并），
// 写 out[col][row]（row 连续，内存里隔一行 -> 不合并）。
__global__ void transpose_naive(const float* in, float* out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N)
        out[col * N + row] = in[row * N + col];
}

// 写法 B：共享内存中转，两头都合并。
// 1) 连续读 in 的一小块（合并）-> 存进共享内存 tile
// 2) __syncthreads()
// 3) 从 tile 按转置后的顺序连续写 out（合并）
// tile 行长 TILE+1：+1 是 padding，防 bank conflict（docs/03_memory.md §4.4）
__global__ void transpose_shared(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE + 1];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N)
        tile[threadIdx.y][threadIdx.x] = in[row * N + col];   // 合并读
    __syncthreads();

    // 转置写回：把"行/列"对调，让写也变成连续的
    int r2 = blockIdx.y * TILE + threadIdx.x;
    int c2 = blockIdx.x * TILE + threadIdx.y;
    if (r2 < N && c2 < N)
        out[r2 * N + c2] = tile[threadIdx.x][threadIdx.y];    // 合并写
}

// CPU 参考转置，用来校验 GPU 结果
void transpose_cpu(const float* in, float* out, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            out[j * N + i] = in[i * N + j];
}

// 模板按 kernel 类型实例化，从而能带着 <<<>>> 启动（__global__ 函数
// 不能用普通函数指针在 host 侧调用，必须走启动语法）
template <typename K>
static float kernel_ms(K kernel, dim3 grid, dim3 block,
                       const float* d_in, float* d_out, int N) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    kernel<<<grid, block>>>(d_in, d_out, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms;
}

int main(int argc, char** argv) {
    int N = 2048;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0 || N % TILE != 0) { fprintf(stderr, "N 需 >0 且能被 %d 整除\n", TILE); return 1; }

    size_t bytes = (size_t)N * N * sizeof(float);
    float* h_in  = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)(i % 1000) / 1000.0f;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // 各跑 3 次取平均，减少抖动
    float ms_naive = 0, ms_shared = 0;
    for (int r = 0; r < 3; r++) {
        ms_naive  += kernel_ms(transpose_naive,  grid, block, d_in, d_out, N) / 3.0f;
        ms_shared += kernel_ms(transpose_shared, grid, block, d_in, d_out, N) / 3.0f;
    }

    // 校验 shared 版本结果
    transpose_cpu(h_in, h_ref, N);
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (int i = 0; i < N * N; i++) {
        double e = fabs((double)h_out[i] - (double)h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("矩阵 %dx%d（float，%d 字节/矩阵）\n", N, N, (int)bytes);
    printf("写法 A naive  : %.3f ms   （读合并、写不合并）\n", ms_naive);
    printf("写法 B shared : %.3f ms   （经共享内存，两头都合并）\n", ms_shared);
    printf("加速比        : %.1fx\n", ms_naive / ms_shared);
    printf("正确性校验    : %s\n", max_err == 0 ? "通过" : "失败");

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_out);
    return 0;
}
