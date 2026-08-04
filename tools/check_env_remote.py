#!/usr/bin/env python3
"""远端 Win10（GTX 1080）开发环境体检脚本。

在目标机器上运行：
    python tools/check_env_remote.py [输出文件]

会检查：NVIDIA 驱动 / CUDA Toolkit / cuDNN / MSVC（编译 .cu 必需）/
Python & pip / PyTorch GPU 是否可用 / 关键环境变量，并在最后给汇总表。
运行结果同时保存为 txt（默认当前目录 env_check_<主机名>_<时间>.txt，
UTF-8 带 BOM，Windows 记事本可直接打开）。
本机无 GPU 时也能运行，对应项会显示"未检测到"。
"""

import datetime
import glob
import os
import platform
import re
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CUDA_INSTALL_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"


def cuda_path():
    """返回 CUDA 安装目录：优先读进程环境，其次读注册表，最后探测默认目录。"""
    for var in ("CUDA_PATH", "CUDA_HOME"):
        v = os.environ.get(var)
        if v and os.path.isdir(v):
            return v
    for key in ("HKCU", "HKLM"):
        try:
            out = subprocess.run(
                ["reg", "query", key + r"\Environment"],
                capture_output=True, text=True).stdout
        except Exception:
            continue
        m = re.search(r"CUDA_PATH\s+REG_\w+\s+(\S+)", out or "")
        if m and os.path.isdir(m.group(1)):
            return m.group(1)
    cands = sorted(glob.glob(os.path.join(CUDA_INSTALL_ROOT, "v*")))
    if cands:
        return cands[-1]
    return None

results = []


class Tee:
    """同时输出到控制台和文件的流。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def setup_log(output=None):
    """把 sys.stdout 替换为 Tee，结果同时写控制台和 txt 文件。"""
    if not output:
        node = platform.node() or "unknown"
        output = "env_check_{}_{}.txt".format(node, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    f = open(output, "w", encoding="utf-8-sig")
    sys.stdout = Tee(sys.stdout, f)
    return output, f


def run(args, timeout=15):
    try:
        out = subprocess.run(args, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
        return (out.stdout or out.stderr).strip()
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "(命令超时)"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def record(name, ok, detail):
    results.append((name, ok, detail))


# ---------- 1. NVIDIA 驱动 ----------

def check_driver():
    section("1. NVIDIA 驱动")
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("未检测到 nvidia-smi → 无 NVIDIA GPU，或驱动未安装")
        record("NVIDIA 驱动", False, "未检测到 nvidia-smi")
        return
    info = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    print(info)
    cc = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    print("计算能力(compute capability):", cc)
    record("NVIDIA 驱动", True, info)


# ---------- 2. CUDA Toolkit ----------

def check_cuda_toolkit():
    section("2. CUDA Toolkit (nvcc)")
    ver = run(["nvcc", "--version"])
    if not ver:
        print("未检测到 nvcc → CUDA Toolkit 未安装，或 bin 未加入 PATH")
        record("CUDA Toolkit", False, "未检测到 nvcc")
        return
    print(ver)
    cuda_path = os.environ.get("CUDA_PATH")
    print("CUDA_PATH:", cuda_path or "(未设置)")
    record("CUDA Toolkit", True, ver.splitlines()[0] if ver else ver)


# ---------- 3. cuDNN ----------

def check_cudnn():
    section("3. cuDNN")
    cpath = cuda_path()
    if not cpath:
        print("未定位到 CUDA 安装目录，跳过")
        record("cuDNN", None, "未定位到 CUDA")
        return
    cudnn = os.path.join(cpath, "bin", "cudnn64.dll")
    ok = os.path.isfile(cudnn)
    print("系统 cudnn64.dll:", "存在" if ok else "未找到（纯 PyTorch 场景非必需）")
    torch_dll = None
    try:
        import torch
        torch_dir = os.path.dirname(torch.__file__)
        hits = glob.glob(os.path.join(torch_dir, "lib", "cudnn*.dll"))
        torch_dll = hits[0] if hits else None
    except Exception:
        pass
    bundled = torch_dll is not None
    print("PyTorch 自带 cuDNN dll:",
          "存在" if bundled else "未找到")
    record("cuDNN", ok or bundled, "系统层找到" if ok
           else ("PyTorch 自带" if bundled else "未找到"))


# ---------- 4. MSVC 编译器 ----------

def check_msvc():
    section("4. MSVC 编译器（CUDA C/C++ 编译必需）")
    cl = shutil.which("cl")
    if cl:
        print("cl.exe:", cl)
        print("提示: 推荐在 'x64 Native Tools Command Prompt' 中编译 .cu")
        record("MSVC", True, cl)
        return
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.isfile(vswhere):
        out = run([vswhere, "-latest", "-products", "*",
                   "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                   "-property", "installationPath"])
        if out:
            print("Visual Studio 安装:", out)
            vc = os.path.join(out, "VC", "Tools", "MSVC")
            if os.path.isdir(vc):
                vers = sorted(os.listdir(vc))
                print("VC 工具集版本:", vers)
                if vers:
                    cl_path = os.path.join(vc, vers[-1], "bin", "Hostx64", "x64", "cl.exe")
                    ok = os.path.isfile(cl_path)
                    print("cl.exe:", cl_path, "存在" if ok else "(未找到，可能缺 x64 工具)")
                    record("MSVC", ok, cl_path)
                    return
                record("MSVC", False, "VC 工具集目录为空")
            else:
                print("未找到 VC/Tools/MSVC → 需安装 '使用 C++ 的桌面开发' 工作负载")
                record("MSVC", False, "缺少 C++ 工具集")
        else:
            print("已安装 Visual Studio，但缺少 C++ 工具集组件")
            record("MSVC", False, "VS 缺 C++ 组件")
    else:
        print("未找到 vswhere / cl.exe → 未安装 MSVC，nvcc 无法编译 .cu")
        record("MSVC", False, "未安装")


# ---------- 5. Python & pip ----------

def check_python():
    section("5. Python & pip")
    print("Python:", sys.version.split()[0], "|", sys.executable)
    pipv = run([sys.executable, "-m", "pip", "--version"])
    print("pip:", pipv or "(异常)")
    cfg = run([sys.executable, "-m", "pip", "config", "list"])
    print("pip 镜像配置:", cfg or "(未配置)")
    record("Python", True, sys.version.split()[0])


# ---------- 6. PyTorch ----------

def check_pytorch():
    section("6. PyTorch")
    try:
        import torch
        print("torch 版本:", torch.__version__)
        print("PyTorch 内置 CUDA:", torch.version.cuda)
        try:
            print("PyTorch 内置 cuDNN:", torch.backends.cudnn.version())
        except Exception:
            pass
        print("CUDA 可用:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("计算能力:", torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            print("显存总量:", f"{props.total_memory / 1e9:.1f} GB")
            x = torch.randn(500, 500, device="cuda")
            y = x @ x
            print("GPU 实测: 500×500 矩阵乘 OK")
            del x, y
            record("PyTorch GPU", True, torch.__version__)
        else:
            print("当前是 CPU 版 PyTorch")
            record("PyTorch GPU", False, f"{torch.__version__} CPU 版")
    except ImportError:
        print("PyTorch 未安装")
        record("PyTorch GPU", False, "未安装")


def check_compat():
    """核对工具链版本是否匹配：nvcc vs PyTorch 内置 CUDA vs 驱动支持。"""
    section("8. 工具链匹配")
    try:
        import torch
    except ImportError:
        print("PyTorch 未安装，跳过匹配检查")
        record("工具链匹配", True, "PyTorch 未安装，跳过")
        return

    torch_cuda = getattr(torch.version, "cuda", None)
    nvcc = run(["nvcc", "--version"])
    m = re.search(r"release\s+([\d.]+)", nvcc) if nvcc else None
    nvcc_cuda = m.group(1) if m else None

    driver = None
    if shutil.which("nvidia-smi"):
        driver = run(["nvidia-smi", "--query-gpu=driver_version",
                      "--format=csv,noheader"])

    lines = []
    if torch_cuda and nvcc_cuda and torch_cuda.rsplit('.', 0)[0]:
        tc = tuple(int(x) for x in torch_cuda.split(".")[:2])
        nc = tuple(int(x) for x in nvcc_cuda.split(".")[:2])
        if tc == nc:
            lines.append(f"OK   nvcc({nvcc_cuda}) == PyTorch 内置 CUDA({torch_cuda})")
        else:
            lines.append(
                f"注意 nvcc({nvcc_cuda}) != PyTorch 内置 CUDA({torch_cuda})；"
                f"编译 .cu 需安装对应 nvcc 版本的 CUDA Toolkit")
    elif torch_cuda:
        lines.append(f"PyTorch 内置 CUDA: {torch_cuda}（nvcc 未安装，无法用 nvcc 编译）")
    else:
        lines.append("未获取到 CUDA 信息")

    if driver:
        try:
            dv = float(driver.strip())
            if dv < 452.39:
                lines.append(f"警告 驱动 {driver} 过旧，不支持 CUDA 11 及以上")
            elif dv < 525.60:
                lines.append(f"驱动 {driver} 支持 CUDA 11.x（最高到 11.8/12.0 附近）")
            else:
                lines.append(f"驱动 {driver} 支持 CUDA 12.x")
        except ValueError:
            lines.append(f"驱动 {driver}：无法解析版本号")
    else:
        lines.append("未检测到 nvidia-smi")

    for ln in lines:
        print(ln)
    record("工具链匹配", not any(l.startswith("注意") for l in lines),
           "；".join(lines))


# ---------- 7. 环境变量 ----------

def check_env():
    section("7. 关键环境变量")
    for var in ["CUDA_PATH", "CUDA_HOME", "HF_HUB_DISABLE_SYMLINKS_WARNING"]:
        print(f"{var}:", os.environ.get(var, "(未设置)"))
    path = os.environ.get("PATH", "")
    hit = [p for p in path.split(";") if "cuda" in p.lower() or "nvidia" in p.lower()]
    print("PATH 中的 CUDA/NVIDIA 项:", hit or "(无)")


# ---------- 汇总 ----------

def summary():
    section("体检汇总")
    print(f"{'组件':<22}{'状态':<8}说明")
    print("-" * 60)
    for name, ok, detail in results:
        if ok is True:
            status, mark = "OK", "[OK]"
        elif ok is None:
            status, mark = "--", "[--]"
        else:
            status, mark = "MISSING", "[!]"
        print(f"{mark} {name:<22}{status:<8}{detail or ''}")
    missing = [r[0] for r in results if r[1] is False]
    print("-" * 60)
    if missing:
        print("缺失项:", "、".join(missing))
        print("其中 MSVC 缺失会导致 nvcc 无法编译 .cu；CUDA Toolkit 缺失会导致无法用 nvcc。")
        print("注意: cuDNN 缺失仅影响系统级 CUDA 显式调用，PyTorch 自带 cuDNN 可正常使用。")
    else:
        print("关键组件齐全，可以开始 CUDA 编程。")

    if not shutil.which("nvidia-smi"):
        print("\n提示: 本机未检测到 NVIDIA GPU，脚本可在无 GPU 机器上运行，GPU 相关项会 MISSING。")


def main():
    output_arg = sys.argv[1] if len(sys.argv) > 1 else None
    output, f = setup_log(output_arg)
    console = sys.stdout.streams[0]
    print("远端 Win10 开发环境体检")
    print("结果文件:", output)
    print("主机:", platform.node(), "|", platform.system(), platform.release(),
          "|", platform.machine())
    check_driver()
    check_cuda_toolkit()
    check_cudnn()
    check_msvc()
    check_python()
    check_pytorch()
    check_env()
    check_compat()
    summary()
    print("\n结果已保存到:", output)
    f.close()
    sys.stdout = console


if __name__ == "__main__":
    main()
