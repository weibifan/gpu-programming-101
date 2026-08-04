// =====================================================================
// 08_cuda_libs / 02_curand_generate.cu
// 用 cuRAND 在显存里批量生成 100 万个 [0,1) 均匀随机数
//
// 编译（需链接 cuRAND）：
//   nvcc 02_curand_generate.cu -o 02_curand_generate -lcurand
// 运行：
//   .\02_curand_generate.exe
//
// 核心知识点（docs/02_cuda_basics.md §7.2）：
//   随机数直接在 GPU 上生成，不用 CPU 生成再拷贝；固定种子可复现
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>

#define CHECK_CUDA(call) do {                                                     \
    cudaError_t e = (call);                                                       \
    if (e != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,               \
                cudaGetErrorString(e));                                           \
        exit(1);                                                                  \
    } } while (0)

#define CHECK_CURAND(call) do {                                                   \
    curandStatus_t s = (call);                                                    \
    if (s != CURAND_STATUS_SUCCESS) {                                             \
        fprintf(stderr, "cuRAND 错误 %s:%d: %d\n", __FILE__, __LINE__, (int)s);    \
        exit(1);                                                                  \
    } } while (0)

int main() {
    const int N = 1000000;
    float *d_r, *h_r;
    h_r = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMalloc(&d_r, N * sizeof(float)));

    // 1. 创建生成器
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // 2. 固定种子（复现用）
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 12345ULL));
    // 3. 直接在显存里生成 N 个 [0,1) 均匀随机数
    CHECK_CURAND(curandGenerateUniform(gen, d_r, N));
    // 4. 销毁生成器
    CHECK_CURAND(curandDestroyGenerator(gen));

    CHECK_CUDA(cudaMemcpy(h_r, d_r, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 简单统计：均匀分布 [0,1) 的均值理论值是 0.5
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += h_r[i];
    printf("均值 = %.4f（均匀分布 [0,1) 理论值 0.5）\n", sum / N);
    printf("前 5 个：");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_r[i]);
    printf("\n");

    CHECK_CUDA(cudaFree(d_r));
    free(h_r);
    return 0;
}
