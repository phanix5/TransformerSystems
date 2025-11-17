
## Flash Attention Time

The following shows example benchmark and test runs:

```
(main) root@C.27939648:/workspace/TransformerSystems$ uv run python -u cs336_systems/flash_attn_leaderboard_benchmark.py --rep 10000 --warmup 1000

Running on CUDA device: NVIDIA H100 80GB HBM3

Config: dtype=bf16, causal=True, n_heads(as batch)=16, d_head=64, seq_len=16384, compiled=True

Forward+Backward latency (ms): 7.499

(main) root@C.27939648:/workspace/TransformerSystems$ uv run pytest tests/test_attention.py

========================================================================= test session starts =========================================================================

platform linux -- Python 3.11.13, pytest-8.4.1, pluggy-1.6.0

rootdir: /workspace/TransformerSystems

configfile: pyproject.toml

plugins: anyio-4.11.0, jaxtyping-0.3.2

collected 8 items                                                                                                                                                     


tests/test_attention.py::test_flash_forward_pass_pytorch PASSED

tests/test_attention.py::test_flash_forward_pass_triton[False] PASSED

tests/test_attention.py::test_flash_forward_pass_triton[True] PASSED

tests/test_attention.py::test_flash_backward_pytorch PASSED

tests/test_attention.py::test_flash_backward_triton[False] PASSED

tests/test_attention.py::test_flash_backward_triton[True] PASSED

tests/test_attention.py::test_forward_pytorch_and_triton_same_input_no_assertions PASSED

tests/test_attention.py::test_backward_pytorch_and_triton_same_input_no_assertions PASSED
```