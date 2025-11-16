import argparse
import torch
import triton.testing as ttesting

from cs336_systems.flash_attn_triton import TritonAttention as FlashAttentionTriton


def _synchronize_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Leaderboard-style benchmark for FlashAttention Triton implementation"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Device to run on (CUDA expected)")
    parser.add_argument("--seq-len", type=int, default=16384, help="Sequence length (default: 16384)")
    parser.add_argument("--n-heads", type=int, default=16, help="Number of heads treated as batch (default: 16)")
    parser.add_argument("--d-head", type=int, default=64, help="Head dimension (default: 64)")
    parser.add_argument("--rep", type=int, default=1000, help="Number of repetitions for timing")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations for timing")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for the autograd function")
    parser.add_argument("--debug-exceptions", action="store_true", help="Print exceptions during benchmarking")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA.")

    torch.cuda.init()
    device_name = torch.cuda.get_device_name(device)
    print(f"Running on CUDA device: {device_name}")

    # Leaderboard setup (BF16 + causal)
    dtype = torch.bfloat16
    causal = True

    n_heads = args.n_heads
    seq_len = args.seq_len
    d_head = args.d_head

    # Treat heads as the batch dimension: [n_heads, seq_len, d_head]
    q = torch.randn(n_heads, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(n_heads, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(n_heads, seq_len, d_head, device=device, dtype=dtype, requires_grad=True)

    flash_apply = FlashAttentionTriton.apply
    if not args.no-compile:
        try:
            flash_apply = torch.compile(flash_apply, fullgraph=False)  # type: ignore[attr-defined]
            compiled = True
        except Exception as e:
            compiled = False
            print(f"[WARN] torch.compile failed or unavailable: {e}. Continuing without compile.")
    else:
        compiled = False

    def flash_forward_backward():
        o = flash_apply(q, k, v, causal)
        loss = o.sum()
        loss.backward()
        # Avoid grad accumulation across repetitions
        q.grad = None
        k.grad = None
        v.grad = None
        _synchronize_if_cuda()

    print(
        f"Config: dtype=bf16, causal={causal}, n_heads(as batch)={n_heads}, "
        f"d_head={d_head}, seq_len={seq_len}, compiled={compiled}"
    )

    try:
        results_ms = float(ttesting.do_bench(flash_forward_backward, rep=args.rep, warmup=args.warmup))
    except Exception:
        if args.debug_exceptions:
            import traceback
            print("[DEBUG] Benchmark execution failed:")
            traceback.print_exc()
        raise

    print(f"Forward+Backward latency (ms): {results_ms:.3f}")


if __name__ == "__main__":
    main()


