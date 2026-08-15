// =====================================================================
// 08_cuda_libs / 03_cufft_fft.cu
// 用 cuFFT 对 1D 复数序列做 FFT，观察"一个周期的余弦"的频谱
//
// 编译（需链接 cuFFT）：
//   nvcc 03_cufft_fft.cu -o 03_cufft_fft -lcufft
// 运行：
//   .\03_cufft_fft.exe
//
// 核心知识点（docs/02_cuda_basics.md §9.1）：
//   cuFFT 用法类似 FFTW：先创建 plan，再执行变换
// =====================================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) do {                                                     \
    cudaError_t e = (call);                                                       \
    if (e != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,               \
                cudaGetErrorString(e));                                           \
        exit(1);                                                                  \
    } } while (0)

#define CHECK_CUFFT(call) do {                                                    \
    cufftResult r = (call);                                                       \
    if (r != CUFFT_SUCCESS) {                                                     \
        fprintf(stderr, "cuFFT 错误 %s:%d: %d\n", __FILE__, __LINE__, (int)r);     \
        exit(1);                                                                  \
    } } while (0)

int main() {
    const int N = 8;
    cufftComplex h_in[N], h_out[N];

    // 一个完整的余弦周期：cos(2*pi*i/8)，i=0..7
    for (int i = 0; i < N; i++) {
        h_in[i].x = cosf(2.0f * M_PI * i / N);
        h_in[i].y = 0.0f;
    }

    cufftComplex *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // 1. 创建 plan：1D、复数->复数、batch = 1
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    // 2. 执行正向 FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    // 3. 销毁 plan
    CHECK_CUFFT(cufftDestroy(plan));

    CHECK_CUDA(cudaMemcpy(h_out, d_out, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // 一个余弦周期 = 1Hz 信号：DFT 应在 k=1 和 k=N-1 出现两个峰（幅值各 N/2 = 4）
    printf("输入：cos(2*pi*i/8), i=0..7（1Hz 余弦）\n");
    printf("输出频谱（期望 k=1 和 k=7 幅值为 4.0，其余为 0）：\n");
    for (int k = 0; k < N; k++)
        printf("  k=%d: 幅值 %.3f  (%.3f %+.3fi)\n", k,
               hypotf(h_out[k].x, h_out[k].y), h_out[k].x, h_out[k].y);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
