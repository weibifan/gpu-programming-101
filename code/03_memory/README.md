# 03_memory — 内存模型（M2）

目标：亲眼看到"合并访问"和"共享内存"对性能的悬殊影响。

| 文件 | 演示内容 |
|---|---|
| `stride_bandwidth.cu` | **合并访问 vs 未合并**：只改线程访问的步长 stride，实测有效带宽从 320 GB/s 一路跌到几分之一 |
| `matrix_transpose.cu` | **矩阵转置陷阱**：naive（读合并写不合并）vs 共享内存中转（两头都合并），看加速比 |

## 编译 / 运行

先加载 MSVC 环境（否则 `nvcc` 找不到 `cl.exe`，见 `docs/01_environment.md` §3.3）：

```powershell
cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && nvcc stride_bandwidth.cu -o stride_bandwidth.exe -arch=sm_61 && .\stride_bandwidth.exe"

# 矩阵转置同理（可选指定边长 N，默认 2048）
cmd /c "\"...\vcvars64.bat\" && nvcc matrix_transpose.cu -o matrix_transpose.exe -arch=sm_61 && .\matrix_transpose.exe 4096"
```

预期输出：

* `stride_bandwidth`：stride=1 带宽最高；stride 每翻倍，有效带宽约减半。
* `matrix_transpose`：shared 版明显快于 naive 版；N 越大差距越明显。

## 要点回顾

* 全局内存按 **128 字节事务** 搬运，一个 warp 的 32 个线程尽量访问**连续地址**才能合并成最少事务。
* 转置里 naive 版"读写总有一头不合并"，**用共享内存转一道**就能两头都合并——这是所有高性能 kernel 的通用套路。
* 共享内存的 tile 行长建议 `TILE+1`（padding），避免 32 的倍数造成 bank conflict（详见 `docs/03_memory.md` §3.4 / §6）。

对应文档：`docs/03_memory.md` §3、§5、§6。
