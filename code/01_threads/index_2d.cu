// =====================================================================
// 01_threads / index_2d.cu
// 二维 block + 二维 grid：用 (row, col) 处理二维矩阵。
// 展示如何从 blockIdx.y/blockDim.y/threadIdx.y 解析出"行"，
// 再把 (row,col) 压平为行主序一维下标。
//
// 核心对应（docs/02_cuda_basics.md §5.2）：
//   col = blockIdx.x*blockDim.x + threadIdx.x   （列，最快）
//   row = blockIdx.y*blockDim.y + threadIdx.y   （行）
//   idx = row * width + col                     （行主序压平）
//
// 编译 / 运行：
//   nvcc index_2d.cu -o index_2d        # Windows 生成 index_2d.exe
//   ./index_2d
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

// 每个线程写一个格子：值 = row*10000 + col，方便肉眼校验位置
__global__ void fill_2d(float *out, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int idx = row * width + col;         // 行主序
        out[idx] = row * 10000.0f + col;
    }
}

int main() {
    const int W = 7, H = 5;                  // 7 列 × 5 行
    const int BS = 4;                        // 每个 block：4×4 线程
    dim3 block(BS, BS, 1);
    dim3 grid((W + BS - 1) / BS, (H + BS - 1) / BS, 1);
    const size_t bytes = (size_t)W * H * sizeof(float);

    float *h_out = (float *)malloc(bytes);
    float *d_out;
    CHECK(cudaMalloc(&d_out, bytes));

    printf("grid=(%u,%u)，block=(%u,%u)，线程=%u×%u=%u\n",
           grid.x, grid.y, block.x, block.y,
           grid.x * block.x, grid.y * block.y,
           grid.x * block.x * grid.y * block.y);

    fill_2d<<<grid, block>>>(d_out, W, H);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 校验并打印矩阵
    int ok = 1;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (h_out[r * W + c] != r * 10000.0f + c) ok = 0;
    printf("校验：%s\n", ok ? "正确" : "有误");
    printf("out[r][c] = row*10000+col：\n");
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++)
            printf("%7g ", (double)h_out[r * W + c]);
        printf("\n");
    }

    CHECK(cudaFree(d_out));
    free(h_out);
    return 0;
}