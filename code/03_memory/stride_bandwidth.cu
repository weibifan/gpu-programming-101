// =====================================================================
// 03_memory / stride_bandwidth.cu
// 合并访问 vs 未合并访问：只改线程访问的"步长(stride)"，实测显存带宽
//
// 编译 / 运行（远端 Win10，GTX 1080，CUDA 11.6）：
//   nvcc stride_bandwidth.cu -o stride_bandwidth -arch=sm_61
//   .\stride_bandwidth.exe
//
// 对应文档：docs/03_cuda_advanced.md §5（合并访问）。
//
// 结论（docs/03_cuda_advanced.md §5.1）：
//   stride=1   线程 t 读 a[t]        相邻线程相邻地址 -> 1 个 128B 事务 -> 带宽拉满
//   stride=32  线程 t 读 a[t*32]     相邻线程隔 128 字节 -> 32 个事务 -> 有效带宽暴跌
//
// 术语"有效带宽" = 真正读到的字节数 ÷ 耗时。本实验让每次运行的
// "总事务数"基本不变，但 stride 越大，同样事务数里能带回来的有效字节越少，
// 于是有效带宽线性下降——这就是"未合并访问 ≈ 带宽利用率打折扣"的实测。
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

// 线程 t 依次读 a[t*stride]、a[(t+T)*stride]、a[(t+2T)*stride]、... 共 iters 次。
// T = 总线程数。相邻线程访问间隔 stride 个元素 -> stride>1 时未合并。
__global__ void read_strided(const float* a, float* sink, int stride, int iters) {
    int t  = blockIdx.x * blockDim.x + threadIdx.x;
    int T  = blockDim.x * gridDim.x;
    float acc = 0.f;
    for (int i = 0; i < iters; i++)
        acc += a[(t + i * T) * stride];
    // 结果从不等于 -1.2345e38，但这一行让编译器没法把整个读循环优化掉
    if (acc == -1.2345e38f) sink[0] = acc;
}

int main() {
    // 数组总元素数 n = T * M。M 取 128，保证 stride ∈ {1,2,4,8,16,32,64} 都能整除。
    const int T = 1 << 20;      // 总线程数 1,048,576
    const int M = 128;
    const size_t n = (size_t)T * M;            // 1.34 亿个 float ≈ 512 MB
    const size_t bytes = n * sizeof(float);
    const int strides[] = {1, 2, 4, 8, 16, 32, 64};

    printf("数组大小 = %.0f MB，总线程数 = %d\n\n", bytes / 1024.0 / 1024.0, T);

    float *d_a, *d_sink;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CHECK(cudaMemset(d_a, 0, bytes));   // 全 0，读出来就是 0，只测访存带宽

    // 测每个 stride 前先跑一次热身，排除驱动/分配抖动
    read_strided<<<(T + 255) / 256, 256>>>(d_a, d_sink, 1, M);
    CHECK(cudaDeviceSynchronize());

    const int n_strides = (int)(sizeof(strides) / sizeof(strides[0]));
    float ms_all[8];

    for (int si = 0; si < n_strides; si++) {
        int stride = strides[si];
        int iters = M / stride;                       // 整除，每线程恰好 iters 次读

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        read_strided<<<(T + 255) / 256, 256>>>(d_a, d_sink, stride, iters);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

        ms_all[si] = ms;
    }

    // 汇总：有效带宽 = 有效字节 / 秒；相对 stride=1 的利用率
    double base = (double)T * (M / strides[0]) * sizeof(float) / (ms_all[0] * 1e6);
    printf("%-8s %-14s %-14s %-10s\n", "stride", "有效带宽 GB/s", "相对利用率", "耗时 ms");
    for (int si = 0; si < n_strides; si++) {
        double gbps = (double)T * (M / strides[si]) * sizeof(float) / (ms_all[si] * 1e6);
        printf("%-8d %-14.1f %-13.0f%% %-10.3f\n",
               strides[si], gbps, gbps / base * 100.0, ms_all[si]);
    }

    printf("\n结论（docs/03_cuda_advanced.md §5.2）：stride 越大，每个 128B 事务里真正用到的\n");
    printf("字节越少（stride=32 时每事务只用 4B/128B），所以有效带宽随 stride 反比下降——\n");
    printf("这就是\"未合并访问 ≈ 带宽利用率打折扣\"的实测证据。\n");

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_sink));
    return 0;
}
