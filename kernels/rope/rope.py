import torch
import time
import math
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="rope",
    sources=["rope.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, out)
    else:
        for i in range(warmup):
            _ = perf_func(a)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, out)
    else:
        for i in range(iters):
            out = perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[a.shape[1]:3+a.shape[1]]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>20}: {out_val}, time:{mean_time:.6f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def naive_rope(
    x: torch.Tensor,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = x.shape[-1]
    seq_len = x.shape[-2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.cat((freqs, freqs), dim=-1)
    t = torch.arange(seq_len , device=freqs.device)
    freqs = torch.outer(t, freqs).float().cuda()
    cos = freqs.cos()
    sin = freqs.sin()
    out = apply_rotary_pos_emb(x, cos, sin)
    return out

print("-" * 100)
M = [4096, 8192]
N = [512, 1024]
MN = [[m, n] for m in M for n in N]
for M,N in MN:
    print(" " * 40 + f"M={M}, N={N}")
    print("-" * 100)
    x = torch.randn((M, N)).cuda().float().contiguous()
    out = torch.zeros_like(x).cuda().float().contiguous()
    run_benchmark(lib.rope_f32,          x, "f32",          out)
    run_benchmark(lib.rope_f32x4_pack,   x, "f32x4_pack",   out)
    run_benchmark(naive_rope,            x, "f32_th")
    print("-" * 100)     
