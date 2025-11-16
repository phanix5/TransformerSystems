import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import pandas as pd
import triton.testing as ttesting

from cs336_systems.flash_attn_triton import TritonAttention as FlashAttentionTriton


@dataclass
class BenchResult:
    triton_fwd_ms: Optional[float]
    triton_bwd_ms: Optional[float]
    triton_fwbw_ms: Optional[float]
    torch_fwd_ms: Optional[float]
    torch_bwd_ms: Optional[float]
    torch_fwbw_ms: Optional[float]
    triton_status: str
    torch_status: str


def _synchronize_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_forward(impl_apply: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    def fn():
        with torch.inference_mode():
            _ = impl_apply(q, k, v, True)
        _synchronize_if_cuda()
    return float(ttesting.do_bench(fn))


def _bench_backward_only(impl_apply: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    # Build graph once, measure only backward with retain_graph=True
    q_rt = q.clone().detach().requires_grad_(True)
    k_rt = k.clone().detach().requires_grad_(True)
    v_rt = v.clone().detach().requires_grad_(True)
    out = impl_apply(q_rt, k_rt, v_rt, True)
    do = torch.ones_like(out)
    _synchronize_if_cuda()

    def fn():
        torch.autograd.backward(out, do, retain_graph=True)
        _synchronize_if_cuda()
        # clear grads to avoid accumulation across reps
        if q_rt.grad is not None:
            q_rt.grad.zero_()
        if k_rt.grad is not None:
            k_rt.grad.zero_()
        if v_rt.grad is not None:
            v_rt.grad.zero_()
    return float(ttesting.do_bench(fn))


def _bench_fwd_bwd(impl_apply: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    def fn():
        q_i = q.clone().detach().requires_grad_(True)
        k_i = k.clone().detach().requires_grad_(True)
        v_i = v.clone().detach().requires_grad_(True)
        out = impl_apply(q_i, k_i, v_i, True)
        loss = out.sum()
        loss.backward()
        _synchronize_if_cuda()
    return float(ttesting.do_bench(fn))


def _reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    # [B, S, D] @ [B, D, S] -> [B, S, S]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        sq = q.shape[-2]
        sk = k.shape[-2]
        i = torch.arange(sq, device=scores.device)
        j = torch.arange(sk, device=scores.device)
        mask = i[:, None] >= j[None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


try:
    _compiled_reference_attention = torch.compile(_reference_attention, fullgraph=False)  # type: ignore[attr-defined]
except Exception:
    _compiled_reference_attention = _reference_attention


def run_one_config(
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BenchResult:
    # Batch size is always 1
    q = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)

    # Ensure a clean CUDA stream
    _synchronize_if_cuda()

    # Implementations
    torch_apply = _compiled_reference_attention
    triton_apply = FlashAttentionTriton.apply

    triton_fwd_ms = triton_bwd_ms = triton_fwbw_ms = None
    torch_fwd_ms = torch_bwd_ms = torch_fwbw_ms = None
    triton_status = "ok"
    torch_status = "ok"

    # Triton measurements
    try:
        triton_fwd_ms = _bench_forward(triton_apply, q, k, v)
    except Exception:
        triton_status = "error_fwd"
    if triton_status == "ok":
        try:
            triton_bwd_ms = _bench_backward_only(triton_apply, q, k, v)
        except Exception:
            triton_status = "error_bwd"
    if triton_status == "ok":
        try:
            triton_fwbw_ms = _bench_fwd_bwd(triton_apply, q, k, v)
        except Exception:
            triton_status = "error_fwbw"

    # PyTorch measurements
    try:
        torch_fwd_ms = _bench_forward(torch_apply, q, k, v)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch_status = "oom_fwd"
        else:
            torch_status = "error_fwd"
    except Exception:
        torch_status = "error_fwd"

    if torch_status == "ok":
        try:
            torch_bwd_ms = _bench_backward_only(torch_apply, q, k, v)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch_status = "oom_bwd"
            else:
                torch_status = "error_bwd"
        except Exception:
            torch_status = "error_bwd"

    if torch_status == "ok":
        try:
            torch_fwbw_ms = _bench_fwd_bwd(torch_apply, q, k, v)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch_status = "oom_fwbw"
            else:
                torch_status = "error_fwbw"
        except Exception:
            torch_status = "error_fwbw"

    return BenchResult(
        triton_fwd_ms=triton_fwd_ms,
        triton_bwd_ms=triton_bwd_ms,
        triton_fwbw_ms=triton_fwbw_ms,
        torch_fwd_ms=torch_fwd_ms,
        torch_bwd_ms=torch_bwd_ms,
        torch_fwbw_ms=torch_fwbw_ms,
        triton_status=triton_status,
        torch_status=torch_status,
    )


def _default_seq_lens():
    return [2 ** p for p in range(7, 17)]  # 128 .. 65536


def _default_d_models():
    return [2 ** p for p in range(4, 8)]   # 16, 32, 64, 128


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 Triton vs PyTorch using triton.testing.do_bench")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on (single H100 expected)")
    parser.add_argument("--seq-lens", nargs="*", type=int, default=_default_seq_lens(), help="Sequence lengths to test")
    parser.add_argument("--d-models", nargs="*", type=int, default=_default_d_models(), help="Embedding sizes to test")
    parser.add_argument("--precisions", nargs="*", type=str, default=["bf16", "fp32"], choices=["bf16", "fp32"], help="Precisions to test")
    parser.add_argument("--rep", type=int, default=256, help="Number of repetitions per timing")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations (do_bench handles warmup internally)")
    parser.add_argument("--to-markdown", type=str, default=None, help="Optional path to save the results as Markdown")
    args = parser.parse_args()

    # Configure do_bench defaults
    ttesting.BENCHMARKS_DEFAULTS["rep"] = args.rep
    ttesting.BENCHMARKS_DEFAULTS["warmup"] = args.warmup

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        name = torch.cuda.get_device_name(device)
        print(f"Running on CUDA device: {name}")
    else:
        print("Running on CPU (note: Triton kernels require CUDA; Triton rows will error).")

    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    precisions = [dtype_map[p] for p in args.precisions]

    rows = []
    for dtype in precisions:
        for d_model in args.d_models:
            for seq_len in args.seq_lens:
                # Skip bf16 on CPU
                if device.type != "cuda" and dtype == torch.bfloat16:
                    rows.append({
                        "device": str(device),
                        "precision": "bf16",
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "triton_fwd_ms": None,
                        "triton_bwd_ms": None,
                        "triton_fwbw_ms": None,
                        "torch_fwd_ms": None,
                        "torch_bwd_ms": None,
                        "torch_fwbw_ms": None,
                        "triton_status": "unsupported",
                        "torch_status": "unsupported",
                    })
                    continue
                try:
                    res = run_one_config(seq_len, d_model, dtype, device)
                    rows.append({
                        "device": str(device),
                        "precision": "bf16" if dtype == torch.bfloat16 else "fp32",
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "triton_fwd_ms": res.triton_fwd_ms,
                        "triton_bwd_ms": res.triton_bwd_ms,
                        "triton_fwbw_ms": res.triton_fwbw_ms,
                        "torch_fwd_ms": res.torch_fwd_ms,
                        "torch_bwd_ms": res.torch_bwd_ms,
                        "torch_fwbw_ms": res.torch_fwbw_ms,
                        "triton_status": res.triton_status,
                        "torch_status": res.torch_status,
                    })
                except RuntimeError as e:
                    status = "oom" if "out of memory" in str(e).lower() else "error"
                    rows.append({
                        "device": str(device),
                        "precision": "bf16" if dtype == torch.bfloat16 else "fp32",
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "triton_fwd_ms": None,
                        "triton_bwd_ms": None,
                        "triton_fwbw_ms": None,
                        "torch_fwd_ms": None,
                        "torch_bwd_ms": None,
                        "torch_fwbw_ms": None,
                        "triton_status": status,
                        "torch_status": status,
                    })

    df = pd.DataFrame(rows)
    # Order and present the table with both implementations' latencies
    cols = [
        "device", "precision", "seq_len", "d_model",
        "triton_fwd_ms", "triton_bwd_ms", "triton_fwbw_ms",
        "torch_fwd_ms", "torch_bwd_ms", "torch_fwbw_ms",
        "triton_status", "torch_status",
    ]
    df = df[cols]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.to_markdown:
        with open(args.to_markdown, "w") as f:
            f.write(df.to_markdown(index=False, float_align="right"))


if __name__ == "__main__":
    main()


