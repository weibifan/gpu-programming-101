// =====================================================================
// 08_cuda_libs / 01_cublas_sgemm.cu
// 用 cuBLAS 做单精度矩阵乘：C(M,N) = A(M,K) * B(K,N)
//
// 编译（在远端 Win10，需链接 cuBLAS）：
//   nvcc 01_cublas_sgemm.cu -o 01_cublas_sgemm -lcublas
// 运行：
//   .\01_cublas_sgemm.exe
//
// 核心知识点（docs/02_cuda_basics.md §9.1）：
//   1. cuBLAS 按"列主序"（Fortran 风格）解释矩阵，与 C 的行主序相反
//   2. 因此这里显式用列主序布局（lda/ldb/ldc = 每列的元素数），避免转置混淆
//   3. 本程序自带 CPU 参考实现，运行时会自动对比，输出 PASS/FAIL
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do {                                                     \
    cudaError_t e = (call);                                                       \
    if (e != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,               \
                cudaGetErrorString(e));                                           \
        exit(1);                                                                  \
    } } while (0)

#define CHECK_CUBLAS(call) do {                                                   \
    cublasStatus_t s = (call);                                                    \
    if (s != CUBLAS_STATUS_SUCCESS) {                                             \
        fprintf(stderr, "cuBLAS 错误 %s:%d: %d\n", __FILE__, __LINE__, (int)s);    \
        exit(1);                                                                  \
    } } while (0)

// CPU 参考实现：按列主序存取（A[i + k*M]，B[k + j*K]，C[i + j*M]）
void cpu_matmul_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[i + k * M] * B[k + j * K];
            C[i + j * M] = s;
        }
}

int main() {
    const int M = 2, N = 2, K = 2;

    // 按列主序存两个矩阵：
    //   A = [[1, 2], [3, 4]]   ->  列主序 [1, 3, 2, 4]
    //   B = [[5, 6], [7, 8]]   ->  列主序 [5, 7, 6, 8]
    float h_A[M*K] = {1, 3, 2, 4};
    float h_B[K*N] = {5, 7, 6, 8};
    float h_C[M*N] = {0};

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // C = alpha * A * B + beta * C（列主序，lda = M、ldb = K、ldc = M）
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_A, M,
                             d_B, K,
                             &beta,
                             d_C, M));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CHECK_CUBLAS(cublasDestroy(handle));

    // 与 CPU 参考对比
    float ref[M*N];
    cpu_matmul_ref(h_A, h_B, ref, M, N, K);
    int ok = 1;
    for (int i = 0; i < M*N; i++)
        if (h_C[i] != ref[i]) { ok = 0; break; }

    // 按数学上的行序打印（元素 (i,j) = h_C[i + j*M]），期望 [[19,22],[43,50]]
    printf("C = A * B（期望 [[19, 22], [43, 50]]）\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%8.1f ", h_C[i + j * M]);
        printf("\n");
    }
    printf("%s\n", ok ? "结果与 CPU 参考一致  PASS" : "结果不一致  FAIL");

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return ok ? 0 : 1;
}
