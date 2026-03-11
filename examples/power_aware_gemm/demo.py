#!/usr/bin/env python
"""Power-aware F.linear demo — cuBLAS Lt with algorithm selection & profiling.

Examples
--------
# Basic run (uses best algorithm, bfloat16, no bias)
python demo.py --m 1024 --n 1024 --k 1024

# With bias
python demo.py --m 1024 --n 4096 --k 1024 --bias

# Choose a specific algorithm
python demo.py --m 1024 --n 1024 --k 1024 --algo-index 2

# Enable torch profiler and save trace to cwd
python demo.py --m 1024 --n 1024 --k 1024 --profile

# Save trace to a specific directory
python demo.py --m 1024 --n 1024 --k 1024 --profile --trace-dir ./traces
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# Allow ``python demo.py`` from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cublas_linear import CublasLtLinear  # noqa: E402


# ---------------------------------------------------------------------------
# Profiler utilities (adapted from sglang)
# ---------------------------------------------------------------------------

def start_profile(record_shapes: bool = True):
    """Start a torch profiler with CPU + CUDA activities."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        with_stack=True,
        record_shapes=record_shapes,
    )
    profiler.start()
    return profiler


def stop_profile(profiler, trace_path: str | None = None):
    """Stop profiler and optionally export a chrome trace.

    Parameters
    ----------
    profiler : torch.profiler.profile
    trace_path : str | None
        If given, the chrome-trace JSON is written here.
    """
    profiler.stop()
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)
        print(f"[profiler] Chrome trace saved to {trace_path}")
    return profiler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Power-aware F.linear demo using cuBLAS Lt",
    )
    parser.add_argument("--m", type=int, default=1024, help="Rows (batch)")
    parser.add_argument("--n", type=int, default=1024, help="Output features")
    parser.add_argument("--k", type=int, default=1024, help="Input features")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument("--bias", action="store_true", help="Include bias term")
    parser.add_argument(
        "--algo-index",
        type=int,
        default=0,
        help="Algorithm index from get_algorithms (0 = best)",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable torch profiler"
    )
    parser.add_argument(
        "--trace-dir",
        default=".",
        help="Directory for profiler trace output (default: cwd)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--repeat", type=int, default=20, help="Timed iterations"
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    m, n, k = args.m, args.n, args.k

    print(f"Problem : m={m}, n={n}, k={k}, dtype={args.dtype}, bias={args.bias}")

    # ---- Create inputs -------------------------------------------------------
    torch.manual_seed(42)
    x = torch.randn(m, k, device=device, dtype=dtype)
    weight = torch.randn(n, k, device=device, dtype=dtype)
    bias = (
        torch.randn(n, device=device, dtype=torch.float16) if args.bias else None
    )

    # ---- Setup cuBLAS Lt & query algorithms ----------------------------------
    cublas_linear = CublasLtLinear()
    algos = cublas_linear.get_algorithms(m, n, k, args.dtype)
    print(f"Found {len(algos)} algorithm(s)")
    for i, algo in enumerate(algos[:5]):
        print(
            f"  [{i}] tile={algo['tile_name']:<20s} "
            f"stages={algo['stages_id']:<4}  "
            f"waves={algo['waves_count']:.2f}  "
            f"workspace={algo['workspace_size']}"
        )
    if len(algos) > 5:
        print(f"  ... ({len(algos) - 5} more)")

    algo_idx = args.algo_index
    if algo_idx >= len(algos):
        print(f"Warning: --algo-index {algo_idx} >= {len(algos)}, falling back to 0")
        algo_idx = 0
    print(
        f"Selected algorithm [{algo_idx}]: tile={algos[algo_idx]['tile_name']}"
    )

    # ---- Warmup --------------------------------------------------------------
    print(f"\nWarmup ({args.warmup} iters) ...")
    for _ in range(args.warmup):
        out = cublas_linear(x, weight, bias, algo_idx)
    torch.cuda.synchronize()

    # ---- Correctness check ---------------------------------------------------
    ref = F.linear(
        x.float(),
        weight.float(),
        bias.float() if bias is not None else None,
    )
    max_err = (out.float() - ref).abs().max().item()
    print(f"Max |error| vs torch.F.linear : {max_err:.6f}")

    print(out)
    print(ref)

    # ---- Benchmark (with optional profiling) ---------------------------------
    profiler = None
    if args.profile:
        profiler = start_profile(record_shapes=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        out = cublas_linear(x, weight, bias, algo_idx)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) / args.repeat * 1000
    # FLOPs for matmul: 2*m*n*k; bias add is negligible
    tflops = 2.0 * m * n * k / (elapsed_ms / 1000) / 1e12

    print(f"\nPerformance ({args.repeat} iters):")
    print(f"  Avg latency : {elapsed_ms:.3f} ms")
    print(f"  Throughput  : {tflops:.2f} TFLOPS")

    # ---- Save profiler trace -------------------------------------------------
    if profiler is not None:
        os.makedirs(args.trace_dir, exist_ok=True)
        trace_name = f"linear_m{m}_n{n}_k{k}_{args.dtype}"
        if args.bias:
            trace_name += "_bias"
        trace_name += f"_algo{algo_idx}.json.gz"
        trace_path = os.path.join(args.trace_dir, trace_name)
        stop_profile(profiler, trace_path)


if __name__ == "__main__":
    main()
