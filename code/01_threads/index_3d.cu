// =====================================================================
// 01_threads / index_3d.cu
// 三维 block + 三维 grid：体数据场景的线程组织。
// 每个线程从 blockIdx/threadIdx 的 .x/.y/.z 解出 (x,y,z) 坐标，
// 再压平为一维下标（约定：x 最快变化）。
//
// 核心对应（docs/02_cuda_basics.md §5.2）：
//   x = blockIdx.x*blockDim.x + threadIdx.x
//   y = blockIdx.y*blockDim.y + threadIdx.y
//   z = blockIdx.z*blockDim.z + threadIdx.z
//   idx = (z*H + y)*W + x        （行主序，x 最快）
//
// 编译 / 运行：
//   nvcc index_3d.cu -o index_3d        # Windows 生成 index_3d.exe
//   ./index_3d
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

// 写一个体素：值 = x*100 + y*10 + z，便于肉眼校验
__global__ void fill_3d(float *out, int W, int H, int D) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < W && y < H && z < D) {
        int idx = (z * H + y) * W + x;      // 行主序，x 最快
        out[idx] = x * 100.0f + y * 10.0f + z;
    }
}

int main() {
    const int W = 3, H = 2, D = 2;          // 3×2×2 的小体
    dim3 block(2, 2, 1);                    // 2×2×1 = 4 线程/block
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              (D + block.z - 1) / block.z); // 2×1×2 = 4 个 block
    const size_t bytes = (size_t)W * H * D * sizeof(float);

    float *h_out = (float *)malloc(bytes);
    float *d_out;
    CHECK(cudaMalloc(&d_out, bytes));

    printf("grid=(%u,%u,%u)，block=(%u,%u,%u)，总线程=%u\n",
           grid.x, grid.y, grid.z, block.x, block.y, block.z,
           grid.x * grid.y * grid.z * block.x * block.y * block.z);

    fill_3d<<<grid, block>>>(d_out, W, H, D);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = 1;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int idx = (z * H + y) * W + x;
                if (h_out[idx] != x * 100.0f + y * 10.0f + z) ok = 0;
            }
    printf("校验：%s\n", ok ? "正确" : "有误");

    printf("体素 out(x,y,z)=x*100+y*10+z：\n");
    for (int z = 0; z < D; z++) {
        printf("z=%d:\n", z);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = (z * H + y) * W + x;
                printf("%7g ", (double)h_out[idx]);
            }
            printf("\n");
        }
    }

    CHECK(cudaFree(d_out));
    free(h_out);
    return 0;
}