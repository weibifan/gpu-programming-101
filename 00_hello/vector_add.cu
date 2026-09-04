// =====================================================================
// 00_hello / vector_add.cu
// 第一个 CUDA 程序：两个数组对应位置相加（M1 产出物）。
//
// 数据流：CPU 内存准备 -> 拷贝到显存 -> GPU 海量线程并行相加
//        -> 拷贝回 CPU -> 与 CPU 参考结果比对校验。
//
// 编译 / 运行（远端 Win10，需 MSVC + CUDA 11.6；Windows 生成 .exe）：
//   nvcc vector_add.cu -o vector_add
//   ./vector_add [N]          # 可选指定元素个数，默认 1<<20 = 1048576
//
// 对应文档：docs/02_cuda_basics.md §5.2 / §6.1（kernel、host/device、<<<>>>）
// =====================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 统一 CUDA 错误检查：调用失败立即打印并退出
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d : %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err_));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                   \
    } while (0)

// ---------------- kernel（在 GPU 上跑的 C 函数） ----------------
// 每个线程处理一对元素。n 不一定能被 blockDim 整除，所以用 if 防越界。
__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    // 全局线程唯一编号 = blockIdx.x * blockDim.x + threadIdx.x
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {          // 末尾不足一个 block 的线程会走到这里被拦下
        c[i] = a[i] + b[i];
    }
}

// ---------------- CPU 参考实现，用来对 GPU 结果做校验 ----------------
void vec_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int n = 1 << 20;                       // 默认 1,048,576 个元素
    if (argc > 1) n = atoi(argv[1]);
    if (n <= 0) { fprintf(stderr, "N 必须 > 0\n"); return 1; }
    size_t bytes = (size_t)n * sizeof(float);
    printf("向量长度 N = %d\n", n);

    // ---------- 1) 主机（CPU）内存：准备数据 ----------
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);      // GPU 的结果
    float *h_r = (float *)malloc(bytes);      // CPU 参考结果
    if (!h_a || !h_b || !h_c || !h_r) { perror("malloc"); return 1; }
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)(i % 1000) / 1000.0f;
        h_b[i] = (float)(i % 1000) / 1000.0f;
    }

    // ---------- 2) 设备（显存）分配 ----------
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // ---------- 3) 数据搬运：CPU 内存 -> 显存 ----------
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ---------- 4) 启动 kernel：<<<block 数, 每 block 线程数>>> ----------
    int threads  = 256;
    int blocks   = (n + threads - 1) / threads;   // 向上取整
    printf("启动 vec_add<<<%d, %d>>>，共 %d 个线程\n", blocks, threads, blocks * threads);

    // 计时：cudaEvent 高精度测量 GPU 计算耗时
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK(cudaGetLastError());                // 捕获启动参数错误
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    // ---------- 5) 结果搬回：显存 -> CPU 内存 ----------
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ---------- 6) 校验与耗时 ----------
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("kernel 耗时: %.3f ms\n", ms);

    vec_add_cpu(h_a, h_b, h_r, n);
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)h_c[i] - (double)h_r[i]);
        if (err > max_err) max_err = err;
    }
    printf("最大误差: %g  =>  %s\n", max_err,
           max_err < 1e-5 ? "校验通过" : "校验失败");

    // ---------- 7) 清理 ----------
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c); free(h_r);
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return 0;
}