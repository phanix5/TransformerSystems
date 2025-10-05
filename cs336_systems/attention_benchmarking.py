import argparse
import logging
import timeit
import statistics
import os
from typing import Any

import torch
import pandas as pd

from cs336_basics.model import scaled_dot_product_attention as sdp_attention


logger = logging.getLogger("cs336_systems.attention_benchmarking")


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def _warmup_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n: int, device: torch.device):
    for _ in range(n):
        _ = sdp_attention(Q, K, V)
        if device.type == "cuda":
            torch.cuda.synchronize()


def _time_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n: int,
    device: torch.device,
    mem_snapshot_fwd: str | None,
):
    timer = timeit.default_timer
    times = []
    for i in range(n):
        # Start CUDA memory recording on first measured forward, if requested
        if device.type == "cuda" and i == 0 and mem_snapshot_fwd is not None and hasattr(torch.cuda, "memory") and hasattr(torch.cuda.memory, "_record_memory_history"):
            try:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            except Exception:
                pass

        t0 = timer()
        _ = sdp_attention(Q, K, V)
        if device.type == "cuda":
            torch.cuda.synchronize()
            if i == 0 and mem_snapshot_fwd is not None and hasattr(torch.cuda.memory, "_dump_snapshot"):
                try:
                    # Ensure directory exists
                    try:
                        snapshot_dir = os.path.dirname(mem_snapshot_fwd)
                        if snapshot_dir:
                            os.makedirs(snapshot_dir, exist_ok=True)
                    except Exception:
                        pass
                    torch.cuda.memory._dump_snapshot(mem_snapshot_fwd)
                    logger.info("CUDA forward memory snapshot saved to %s", mem_snapshot_fwd)
                except Exception:
                    logger.warning("Failed to dump CUDA forward memory snapshot to %s", mem_snapshot_fwd)
                finally:
                    try:
                        torch.cuda.memory._record_memory_history(enabled=None)
                    except Exception:
                        pass
        t1 = timer()
        times.append(t1 - t0)
    return statistics.fmean(times), (statistics.pstdev(times) if n > 1 else 0.0)


def _time_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n: int,
    device: torch.device,
    mem_snapshot: str | None,
):
    timer = timeit.default_timer
    times = []
    mem_before_backward_mb = None

    for i in range(n):
        # Fresh inputs for each iteration to avoid retain_graph where possible
        Q_i = Q.clone().detach().requires_grad_(True)
        K_i = K.clone().detach().requires_grad_(True)
        V_i = V.clone().detach().requires_grad_(True)

        out = sdp_attention(Q_i, K_i, V_i)
        loss = out.sum()

        if device.type == "cuda":
            torch.cuda.synchronize()
            if i == 0:
                mem_before_backward_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

                # Start CUDA memory history recording only for the first iteration
                if (
                    mem_snapshot is not None
                    and hasattr(torch.cuda, "memory")
                    and hasattr(torch.cuda.memory, "_record_memory_history")
                ):
                    try:
                        torch.cuda.memory._record_memory_history(max_entries=1000000)
                    except Exception:
                        pass

        t0 = timer()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            # Dump snapshot after the first backward iteration and stop recording
            if i == 0 and mem_snapshot is not None and hasattr(torch.cuda, "memory") and hasattr(torch.cuda.memory, "_dump_snapshot"):
                try:
                    # Ensure directory exists
                    try:
                        snapshot_dir = os.path.dirname(mem_snapshot)
                        if snapshot_dir:
                            os.makedirs(snapshot_dir, exist_ok=True)
                    except Exception:
                        pass
                    torch.cuda.memory._dump_snapshot(mem_snapshot)
                    logger.info("CUDA memory snapshot saved to %s", mem_snapshot)
                except Exception:
                    logger.warning("Failed to dump CUDA memory snapshot to %s", mem_snapshot)
                finally:
                    try:
                        torch.cuda.memory._record_memory_history(enabled=None)
                    except Exception:
                        pass
        t1 = timer()
        times.append(t1 - t0)

        # Clear grads between iterations
        for tensor in (Q_i, K_i, V_i):
            if tensor.grad is not None:
                tensor.grad = None

    return (
        statistics.fmean(times),
        (statistics.pstdev(times) if n > 1 else 0.0),
        mem_before_backward_mb,
    )


def benchmark_attention_once(
    *,
    batch_size: int,
    d_model: int,
    seq_len: int,
    warmup: int,
    iters: int,
    device: torch.device,
    mem_snapshot: str | None,
    mem_snapshot_fwd: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "d_model": d_model,
        "seq_len": seq_len,
        "fw_mean_s": None,
        "fw_std_s": None,
        "bw_mean_s": None,
        "bw_std_s": None,
        "mem_before_bwd_mb": None,
        "status": "ok",
        "error": None,
    }

    try:
        # Create inputs (no multihead; shapes: (batch, seq, d_model))
        Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
        K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
        V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        # Forward warmup and timing without autograd
        with torch.inference_mode():
            _warmup_forward(Q, K, V, warmup, device)
            fw_mean_s, fw_std_s = _time_forward(Q, K, V, iters, device, mem_snapshot_fwd)
        record["fw_mean_s"], record["fw_std_s"] = fw_mean_s, fw_std_s

        # Backward warmup (build graph) and timing
        # Simple warmup: a few forward+backward steps
        for _ in range(min(warmup, 3)):
            Q_w = Q.clone().detach().requires_grad_(True)
            K_w = K.clone().detach().requires_grad_(True)
            V_w = V.clone().detach().requires_grad_(True)
            out_w = sdp_attention(Q_w, K_w, V_w)
            loss_w = out_w.sum()
            loss_w.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()

        bw_mean_s, bw_std_s, mem_before_bwd_mb = _time_backward(Q, K, V, iters, device, mem_snapshot)
        record["bw_mean_s"], record["bw_std_s"] = bw_mean_s, bw_std_s
        record["mem_before_bwd_mb"] = mem_before_bwd_mb

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            record["status"] = "oom"
            record["error"] = "OOM"
            logger.warning("OOM at d_model=%d seq_len=%d on %s", d_model, seq_len, device)
        else:
            record["status"] = "error"
            record["error"] = msg
            logger.exception("Error at d_model=%d seq_len=%d on %s", d_model, seq_len, device)
    except Exception as e:
        record["status"] = "error"
        record["error"] = str(e)
        logger.exception("Error at d_model=%d seq_len=%d on %s", d_model, seq_len, device)
    finally:
        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    return record


def main():
    parser = argparse.ArgumentParser(description="Scaled Dot-Product Attention Benchmarking (no multihead)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--d-models", nargs="*", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", nargs="*", type=int, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for each phase")
    parser.add_argument("--iters", type=int, default=100, help="Measurement iterations for forward/backward")
    parser.add_argument("--log-level", choices=["CRITICAL","ERROR","WARNING","INFO","DEBUG"], default="INFO")
    parser.add_argument("--to-markdown", type=str, default=None, help="Optional path to save markdown table")
    parser.add_argument("--to-latex", type=str, default=None, help="Optional path to save LaTeX table")
    parser.add_argument("--mem-snapshot", type=str, default=None, help="CUDA memory snapshot output path for first backward iter")
    parser.add_argument("--mem-snapshot-fwd", type=str, default=None, help="CUDA memory snapshot output path for first forward iter")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    device = _device_from_arg(args.device)

    records: list[dict[str, Any]] = []

    fixed_batch_size = 8
    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            logger.info("Benchmark d_model=%d seq_len=%d", d_model, seq_len)
            rec = benchmark_attention_once(
                batch_size=fixed_batch_size,
                d_model=d_model,
                seq_len=seq_len,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                mem_snapshot=args.mem_snapshot,
                mem_snapshot_fwd=args.mem_snapshot_fwd,
            )
            records.append(rec)

    df = pd.DataFrame.from_records(records)
    # Order columns similar to basic_benchmarking
    cols = [
        "device",
        "batch_size",
        "d_model",
        "seq_len",
        "fw_mean_s",
        "fw_std_s",
        "bw_mean_s",
        "bw_std_s",
        "mem_before_bwd_mb",
        "status",
        "error",
    ]
    df = df[cols]

    # Print to stdout
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Optional markdown output
    if args.to_markdown:
        try:
            md = df.to_markdown(index=False)
        except Exception:
            header = "| " + " | ".join(df.columns) + " |\n"
            sep = "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
            rows = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.itertuples(index=False, name=None))
            md = header + sep + rows
        # Ensure directory exists
        try:
            md_dir = os.path.dirname(args.to_markdown)
            if md_dir:
                os.makedirs(md_dir, exist_ok=True)
        except Exception:
            pass
        with open(args.to_markdown, "w") as f:
            f.write(md)

    if args.to_latex:
        # Ensure directory exists
        try:
            tex_dir = os.path.dirname(args.to_latex)
            if tex_dir:
                os.makedirs(tex_dir, exist_ok=True)
        except Exception:
            pass
        with open(args.to_latex, "w") as f:
            f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == '__main__':
    main()