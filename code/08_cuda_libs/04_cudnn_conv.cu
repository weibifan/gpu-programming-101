// =====================================================================
// 08_cuda_libs / 04_cudnn_conv.cu
// 用 cuDNN 做一次前向卷积：5x5 输入、3x3 卷积核 -> 3x3 输出
//
// 编译（cuDNN 不随 CUDA Toolkit 提供，需先单独安装，见 README）：
//   nvcc 04_cudnn_conv.cu -o 04_cudnn_conv -lcudnn
// 运行：
//   .\04_cudnn_conv.exe
//
// 核心知识点（docs/02_cuda_basics.md §9.1）：
//   1. cuDNN 用"描述符"（descriptor）描述张量/卷积参数，再执行
//   2. 同一个卷积有多种算法（algo），运行时可自动挑最快的
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call) do {                                                     \
    cudaError_t e = (call);                                                       \
    if (e != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,               \
                cudaGetErrorString(e));                                           \
        exit(1);                                                                  \
    } } while (0)

#define CHECK_CUDNN(call) do {                                                    \
    cudnnStatus_t s = (call);                                                     \
    if (s != CUDNN_STATUS_SUCCESS) {                                              \
        fprintf(stderr, "cuDNN 错误 %s:%d: %s\n", __FILE__, __LINE__,              \
                cudnnGetErrorString(s));                                          \
        exit(1);                                                                  \
    } } while (0)

int main() {
    // 布局 NCHW：1 个样本、1 个通道、5x5
    const int N = 1, C = 1, H = 5, W = 5;
    // 卷积核：1 个输出通道、1 个输入通道、3x3
    const int K = 1, R = 3, S = 3;
    // stride=1, pad=0 -> 输出 3x3
    const int OH = H - R + 1, OW = W - S + 1;

    // 输入填 1..25；卷积核中心=1 其余=0（相当于"恒等平移"）
    float h_in[N*C*H*W], h_w[K*C*R*S], h_out[N*K*OH*OW] = {0};
    for (int i = 0; i < N*C*H*W; i++) h_in[i] = (float)(i + 1);
    for (int i = 0; i < K*C*R*S; i++) h_w[i] = 0.f;
    h_w[(R/2) * S + (S/2)] = 1.f;      // 中心 (1,1) = 1

    float *d_in, *d_w, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  sizeof(float) * N*C*H*W));
    CHECK_CUDA(cudaMalloc(&d_w,   sizeof(float) * K*C*R*S));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * N*K*OH*OW));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(float) * N*C*H*W, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w,  h_w,  sizeof(float) * K*C*R*S, cudaMemcpyHostToDevice));

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 1. 张量描述符（输入 / 输出）
    cudnnTensorDescriptor_t in_desc, out_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, OH, OW));

    // 2. 卷积核描述符
    cudnnFilterDescriptor_t w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

    // 3. 卷积描述符：pad=0, stride=1, dilation=1，互相关模式
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 4. 自动选一个最快的算法，并申请工作区
    cudnnConvolutionFwdAlgo_t algo;
    {
        cudnnConvolutionFwdAlgoPerf_t perf[1];
        int returned = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, in_desc, w_desc, conv_desc,
                                                           out_desc, 1, &returned, perf));
        algo = perf[0].algo;
    }
    size_t ws_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, w_desc, conv_desc,
                                                        out_desc, algo, &ws_size));
    void* ws = nullptr;
    if (ws_size > 0) CHECK_CUDA(cudaMalloc(&ws, ws_size));

    // 5. 执行前向卷积：out = alpha * conv(in, w) + beta * out
    float alpha = 1.f, beta = 0.f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, in_desc, d_in, w_desc, d_w,
                                        conv_desc, algo, ws, ws_size, &beta, out_desc, d_out));

    CHECK_CUDA(cudaMemcpy(h_out, d_out, sizeof(float) * N*K*OH*OW, cudaMemcpyDeviceToHost));

    // 中心=1 的 3x3 卷积核 = 恒等平移：输出应等于输入去掉边界
    // 输入 1..25 去掉边界后：7 8 9 / 12 13 14 / 17 18 19
    printf("输出 3x3（期望 7 8 9 / 12 13 14 / 17 18 19）：\n");
    for (int i = 0; i < OH; i++) {
        for (int j = 0; j < OW; j++) printf("%6.1f ", h_out[i*OW + j]);
        printf("\n");
    }

    if (ws) cudaFree(ws);
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(in_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
