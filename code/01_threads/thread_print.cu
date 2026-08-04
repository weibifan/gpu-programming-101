// =====================================================================
// 01_threads / thread_print.cu
// 用 device printf 观察"每个线程自己看到的世界"：
//   blockIdx.x、threadIdx.x、blockDim.x、gridDim.x。
// 网格故意设小（2 blocks × 8 threads），便于看清层级结构。
//
// 思考题（跑完再看）：
//   1) 打印顺序是 0..15 吗？为什么"看起来乱"？
//      -> GPU 不保证线程执行顺序，warp 调度以硬件调度为准（docs/02 §5.2）
//   2) 全局编号 = blockIdx.x*blockDim.x + threadIdx.x 的关系是否成立？
//
// 编译 / 运行：
//   nvcc thread_print.cu -o thread_print   # Windows 生成 thread_print.exe
//   ./thread_print
// =====================================================================

#include <stdio.h>
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

__global__ void print_ids() {
    // device printf：每个线程打印一行
    printf("thread(%d) <- block(%d)，blockDim=%d gridDim=%d，全局ID=%d\n",
           threadIdx.x, blockIdx.x, blockDim.x, gridDim.x,
           blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    dim3 grid(2, 1, 1);     // 网格：x 方向 2 个 block
    dim3 block(8, 1, 1);    // 每个 block：x 方向 8 个线程

    print_ids<<<grid, block>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());   // 必须等 GPU 把 printf 缓冲刷出来
    return 0;
}