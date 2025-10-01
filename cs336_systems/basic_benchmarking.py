from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from einops import einsum
import contextlib
import math
import argparse
import statistics
import timeit
import torch
from torch import Tensor
from jaxtyping import Int
import pandas as pd


MODEL_SIZES = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def _get_nvtx_ctx(use_nvtx: bool, name: str):
    if not use_nvtx:
        return contextlib.nullcontext()
    try:
        import torch.cuda.nvtx as _nvtx
        return _nvtx.range(name)
    except Exception:
        return contextlib.nullcontext()


def run_step(
    batch: Int[Tensor, " batch seq"],
    model: BasicsTransformerLM,
    mode: str,
    optimizer: torch.optim.Optimizer | None,
    use_nvtx: bool,
) -> None:
    with _get_nvtx_ctx(use_nvtx, "step"):
        if mode == "fwd":
            with _get_nvtx_ctx(use_nvtx, "forward"), torch.inference_mode():
                model.forward(batch)
        else:
            # backward and train modes
            model.zero_grad(set_to_none=True)
            with _get_nvtx_ctx(use_nvtx, "forward"):
                logits = model.forward(batch)
            with _get_nvtx_ctx(use_nvtx, "backward"):
                loss = logits.sum()
                loss.backward()
            if mode == "train":
                assert optimizer is not None
                with _get_nvtx_ctx(use_nvtx, "optimizer"):
                    optimizer.step()

        if batch.device.type == "cuda":
            torch.cuda.synchronize()


def annotated_scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
):
    with _get_nvtx_ctx(True, "scaled dot product attention"):
        d_k = K.shape[-1]
        with _get_nvtx_ctx(True, "computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with _get_nvtx_ctx(True, "computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with _get_nvtx_ctx(True, "final matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def benchmark_once(
    *,
    batch_size: int,
    warmup_iter: int,
    n_iter: int,
    mode: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    dtype: torch.dtype | None,
    device_str: str | None,
    lr: float,
    use_nvtx: bool,
):
    device = (
        torch.device(device_str)
        if device_str is not None and device_str != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device)
    if mode == "fwd":
        model.eval()
    else:
        model.train(True)

    batch = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) if mode == "train" else None

    if device.type == "cuda":
        torch.cuda.synchronize()

    with _get_nvtx_ctx(use_nvtx, "warmup"):
        for _ in range(warmup_iter):
            run_step(batch, model, mode, optimizer, use_nvtx)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    per_iter_times = []
    timer = timeit.default_timer
    with _get_nvtx_ctx(use_nvtx, "measure"):
        for _ in range(n_iter):
            t0 = timer()
            run_step(batch, model, mode, optimizer, use_nvtx)
            t1 = timer()
            per_iter_times.append(t1 - t0)

    peak_mem_bytes = None
    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)

    mean_s = statistics.fmean(per_iter_times)
    std_s = statistics.pstdev(per_iter_times) if n_iter > 1 else 0.0

    return {
        "device": str(device),
        "mode": mode,
        "batch_size": batch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "warmup_iter": warmup_iter,
        "n_iter": n_iter,
        "mean_s": mean_s,
        "std_s": std_s,
        "peak_mem_mb": (peak_mem_bytes / (1024**2)) if peak_mem_bytes is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end benchmarking for BasicsTransformerLM (NVTX-ready)")
    parser.add_argument("--sizes", nargs="*", default=["small"], choices=list(MODEL_SIZES.keys()), help="Model sizes to benchmark")
    parser.add_argument("--context-lengths", nargs="*", type=int, default=[128], help="Sequence lengths to benchmark")
    parser.add_argument("--mode", choices=["fwd", "fwd+bwd", "train"], default="fwd", help="Benchmark mode")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rope-theta", type=float, default=1000.0)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for train mode")
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for nsys filtering")
    parser.add_argument("--override-attn", action="store_true", help="Swap in NVTX-annotated attention kernel")
    parser.add_argument("--to-markdown", type=str, default=None, help="Optional path to save markdown table")
    parser.add_argument("--to-latex", type=str, default=None, help="Optional path to save LaTeX table")
    args = parser.parse_args()

    dtype_map = {
        "auto": None,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    if args.override_attn:
        import cs336_basics.model as _m
        _m.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    records = []
    for size_name in args.sizes:
        cfg = MODEL_SIZES[size_name]
        for context_length in args.context_lengths:
            for mode in (["fwd", "fwd+bwd"] if args.mode == "fwd+bwd" else [args.mode]):
                rec = benchmark_once(
                    batch_size=args.batch_size,
                    warmup_iter=args.warmup,
                    n_iter=args.iters,
                    mode=mode,
                    vocab_size=args.vocab_size,
                    context_length=context_length,
                    d_model=cfg["d_model"],
                    num_layers=cfg["num_layers"],
                    num_heads=cfg["num_heads"],
                    d_ff=cfg["d_ff"],
                    rope_theta=args.rope_theta,
                    dtype=dtype,
                    device_str=args.device,
                    lr=args.lr,
                    use_nvtx=args.nvtx,
                )
                rec["size"] = size_name
                records.append(rec)

    df = pd.DataFrame.from_records(records)
    df = df[
        [
            "size",
            "mode",
            "device",
            "batch_size",
            "context_length",
            "d_model",
            "d_ff",
            "num_layers",
            "num_heads",
            "warmup_iter",
            "n_iter",
            "mean_s",
            "std_s",
            "peak_mem_mb",
        ]
    ]

    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if args.to_markdown:
        with open(args.to_markdown, "w") as f:
            f.write(df.to_markdown(index=False))
    if args.to_latex:
        with open(args.to_latex, "w") as f:
            f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()





