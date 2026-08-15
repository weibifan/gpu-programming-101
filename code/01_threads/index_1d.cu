// =====================================================================
// 01_threads / index_1d.cu
// 一维网格/线程块：理解全局索引计算与"防越界"边界。
//
// 核心公式（docs/02_cuda_basics.md §6.4）：
//     全局编号 = blockIdx.x * blockDim.x + threadIdx.x
//
// 本例子 N=510 故意取 128 的非整数倍（4 blocks×128 = 512 线程），
// 使最后 2 个线程（510、511）越界，靠 if (i<N) 拦下，观察"空闲线程"。
//
// 编译 / 运行：
//   nvcc index_1d.cu -o index_1d        # Windows 生成 index_1d.exe
//   ./index_1d
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

// 每个线程把自己的全局编号写进 out[i]（仅当 i 未越界）
__global__ void fill_index(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

int main() {
    const int n      = 510;          // 故意不是 blockDim 的整数倍
    const int threads = 128;
    const int blocks  = (n + threads - 1) / threads;

    int *h_out = (int *)malloc(n * sizeof(int));
    int *d_out;
    CHECK(cudaMalloc(&d_out, n * sizeof(int)));

    printf("n=%d, blockDim=%d, block数=%d\n", n, threads, blocks);
    printf("启动线程总数 = %d（比 n 多 %d 个，那些线程会因越界而空闲）\n",
           blocks * threads, blocks * threads - n);

    fill_index<<<blocks, threads>>>(d_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    // 校验每个元素：out[i] 应等于 i
    int ok = 1;
    for (int i = 0; i < n; i++) if (h_out[i] != i) { ok = 0; break; }
    printf("所有位置校验：%s\n", ok ? "全部正确" : "有误");

    // 打印前 12 个 + 末尾几个，直观看到全局编号连续
    printf("前 12 个 out[i] =");
    for (int i = 0; i < 12 && i < n; i++) printf(" %d", h_out[i]);
    printf("\n末尾 3 个 out[i] =");
    for (int i = n - 3; i < n; i++) printf(" %d", h_out[i]);
    printf("\n(最后一个合法元素是 n-1=%d)\n", h_out[n - 1]);

    CHECK(cudaFree(d_out));
    free(h_out);
    return 0;
}