# =====================================================================
# 05_llm / 03_quantize_gguf.py
# 量化对比：用 llama.cpp 的 GGUF 模型测 tokens/s 与显存占用
#
# 运行（AutoDL / Linux 首选；需先安装 llama-cpp-python 并下载 GGUF）：
#   pip install llama-cpp-python
#   python 03_quantize_gguf.py --model qwen2-7b-instruct-q4_k_m.gguf
#
#   对比不同量化档位时，把同一个模型的 q4_0 / q4_K_M / q8_0 / f16 各跑一遍：
#   for f in *.gguf; do python 03_quantize_gguf.py --model "$f" --n-tokens 100; done
#
# 对应 docs/05_llm_acceleration.md §4（量化）与 §5（llama.cpp）
#
# 说明：
#   * 量化把每个权重从 FP16(2B) 压到 INT4(0.5B)，搬运量省 4 倍，
#     解码速度理论接近 4 倍（docs/05 §4.1）。
#   * 困惑度(perplexity)对比一般用 lm-evaluation-harness，脚本只测速度/显存。
# =====================================================================
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="GGUF 模型文件路径")
    ap.add_argument("--n-tokens", type=int, default=100, help="生成 token 数")
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="offload 到 GPU 的层数（-1=全部，小显存可调小）")
    args = ap.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("缺少依赖，请先安装：pip install llama-cpp-python")
        raise SystemExit(1)

    print(f"加载模型 {args.model} ...")
    t0 = time.perf_counter()
    llm = Llama(model_path=args.model, n_gpu_layers=args.n_gpu_layers, verbose=False)
    print(f"加载耗时: {time.perf_counter() - t0:.1f} s")

    prompt = "讲一个 GPU 编程的故事。"

    # ---- 测速：tokens/s ----
    t0 = time.perf_counter()
    out = llm(prompt, max_tokens=args.n_tokens, echo=False)
    dt = time.perf_counter() - t0

    n_gen = len(out["choices"][0]["text"].split())          # 粗略字数
    gen_tokens = out["usage"]["completion_tokens"]
    speed = gen_tokens / dt if dt > 0 else 0.0
    print(f"生成 {gen_tokens} tokens，耗时 {dt:.1f} s")
    print(f"decode 速度    : {speed:.1f} tokens/s")
    print(f"生成内容开头  : {out['choices'][0]['text'][:60]!r}")

    # ---- 显存占用（linux 下可用 nvidia-smi 读）----
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        print("显存占用 (MB)  :", r.stdout.strip().splitlines()[0])
    except Exception:
        print("（无法读取 nvidia-smi，跳过显存统计）")


if __name__ == "__main__":
    main()
