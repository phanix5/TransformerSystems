import torch
from torch import Tensor
from jaxtyping import Float
import math
from einops import einsum

class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: Float[Tensor, "... seq_len d_k"], K: Float[Tensor, "... seq_len d_k"], V: Float[Tensor, "... seq_len d_k"], is_causal=False) -> Float[Tensor, "... seq_len d_k"]:
        # Input tensors are guaranteed to be exactly divisible by 16
        b_r = b_c = 16 # block size
        n_r = math.ceil(Q.shape[-2]/b_r)
        n_c = math.ceil(K.shape[-2]/b_c)
        D = Q.shape[-1]
        scale = 1.0 / math.sqrt(D)
        O = torch.zeros_like(Q)
        L_out = torch.empty(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)
        for i in range(n_r):
            st_index_r = i * b_r
            ed_index_r = (i+1) * b_r
            q = Q[..., st_index_r:ed_index_r, :]

            o = torch.zeros_like(q)
            l = torch.zeros(*q.shape[:-1], device=q.device, dtype=q.dtype)
            m = torch.full((*q.shape[:-1],), float('-inf'), device=q.device, dtype=q.dtype)

            for j in range(n_c):
                st_index = j * b_c
                ed_index = (j+1) * b_c
                k = K[..., st_index:ed_index, :]
                v = V[..., st_index:ed_index, :]

                s_j = einsum(q, k, " ... b_r d, ... b_c d -> ... b_r b_c") * scale # QK^T

                if is_causal:
                    mask_r = torch.arange(st_index_r, ed_index_r, device=q.device)
                    mask_c = torch.arange(st_index, ed_index, device=q.device)
                    mask = mask_r[:, None] >= mask_c[None, :]
                    s_j = torch.where(mask, s_j, float('-inf'))

                s_row_max = torch.amax(s_j, dim=-1)
                m_new = torch.maximum(m, s_row_max)
                p = torch.exp(s_j - m_new[..., None])
                l = torch.exp(m - m_new) * l + torch.sum(p, dim=-1)
                o = torch.exp((m - m_new)[..., None]) * o + einsum(p, v, "... q k, ... k d -> ... q d")

                m = m_new
            
            o = o / l[..., None]
            L_blk = m + torch.log(l)

            O[..., st_index_r:ed_index_r, :] = o
            L_out[..., st_index_r:ed_index_r] = L_blk
        # Save tensors needed for backward
        ctx.save_for_backward(L_out, K, Q, V, O)
        ctx.block_sizes = (b_r, b_c)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.softmax_scale = scale
        return O

    def backward(ctx, dO):
        L_out, K, Q, V, O = ctx.saved_tensors

        D = torch.sum(O * dO, dim=-1, keepdim=True)

        S = einsum(Q, K, "... q d, ... k d -> ... q k") * ctx.softmax_scale
        P = torch.exp(S - L_out.unsqueeze(-1))

        if ctx.is_causal:
            P = torch.tril(P)

        dV = einsum(P, dO, "... q k, ... q d -> ... k d")
        print(f"Debug dV: {dV}")
        dP = einsum(dO, V, "... q d, ... k d -> ... q k")

        dS = P * (dP - D)

        dQ = ctx.softmax_scale * einsum(dS, K, "... q k, ... k d -> ... q d")
        dK = ctx.softmax_scale * einsum(dS, Q, "... q k, ... q d -> ... k d")

        # Gradients must match forward inputs order: Q, K, V, is_causal
        return dQ, dK, dV, None






