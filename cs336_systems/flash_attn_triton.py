import torch
from torch import Tensor
import triton
from jaxtyping import Float
import triton.language as tl
import math


@triton.jit
def _attn_fwd(
    Q, K, V,
    O, L,
    stride_batch,
    stride_seq,
    stride_dim,
    softmax_scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    block_idx_q = tl.program_id(0)
    batch_idx = tl.program_id(1)

    q_block_pointer = tl.make_block_ptr(
        Q + stride_batch * batch_idx,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_seq, stride_dim),
        offsets = (BLOCK_SIZE_Q * block_idx_q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    # Load K as [HEAD_DIM, SEQ_LEN] so that tl.dot(Q, K_block) computes Q @ K^T
    k_block_pointer = tl.make_block_ptr(
        K + stride_batch * batch_idx,
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (stride_dim, stride_seq),
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1)
    )

    v_block_pointer = tl.make_block_ptr(
        V + stride_batch * batch_idx,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_seq, stride_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0)
    )

    o_block_pointer = tl.make_block_ptr(
        O + stride_batch * batch_idx,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_seq, stride_dim),
        offsets = (BLOCK_SIZE_Q * block_idx_q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    l_block_pointer = tl.make_block_ptr(
        L + SEQ_LEN * batch_idx,
        shape = (SEQ_LEN,),
        strides = (1,),
        offsets = (BLOCK_SIZE_Q * block_idx_q,),
        block_shape = (BLOCK_SIZE_Q,),
        order = (0,)
    )

    # Load Q block into SRAM
    # q_block = q_11 q_12 ... q_1d
    #           q_21 q_22 ... q_2d
    q_block = tl.load(q_block_pointer)

    tl.device_print("q_block", q_block)

    # o_block = o_11 o_12 ... o_1d
    #           o_21 o_22 ... o_2d
    o_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # m_i = -inf
    #       -inf
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("-inf")

    # l_i = 1
    #       1
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1

    for i in range(tl.cdiv(SEQ_LEN, BLOCK_SIZE_KV)):

        # k_block = k_11 k_21
        #           k_12 k_22
        #           .    .
        #           .    .
        #           k_1d k_2d
        k_block = tl.load(k_block_pointer)
        
        # v_block = v_11 v_12 ... v_1d
        #           v_21 v_22 ... v_2d
        v_block = tl.load(v_block_pointer)

        # First, calculate QK^T, note that we already loaded K as transposed
        # kq_block = qk_11 qk_12
        #            qk_21 qk_22
        kq_block = tl.dot(q_block, k_block) * softmax_scale

        ## DEBUG
        o_block = kq_block

        # Next, apply mask
        # if block_idx_q == i and IS_CAUSAL:
        #     mask_i = tl.arange(BLOCK_SIZE_Q)
        #     mask_j = tl.arange(BLOCK_SIZE_Q)
        #     mask = mask_i[None, :] <= mask_j[:, None]
        #     kq_block = tl.where(mask, kq_block, float("-inf"))

        # Next, find max in block
        # m_ij = max(-inf, max(qk_11, qk_12))
        #        max(-inf, max(qk_21, qk_22))
        #m_ij = tl.maximum(m_i, tl.max(kq_block, 1))

        # Softmax safety: subtract by max till now
        #kq_block -= m_ij[:, None]

        # Next, find new l
        #p_block = tl.math.exp(kq_block)

        # Sum the exponentials
        #l_ij = tl.sum(p_block, 1)

        # correction factor exp(m_old - m_new)
        #alpha = tl.math.exp(m_i - m_ij)

        # add to running sum of exps with correction factor
        #l_i = l_ij + l_i * alpha

        #o_block = o_block * alpha[:, None]
        #o_block = tl.dot(p_block, v_block, o_block)

        #m_i = m_ij

        #v_block_pointer = tl.advance(v_block_pointer, (BLOCK_SIZE_KV, 0))
        #k_block_pointer = tl.advance(k_block_pointer, (0, BLOCK_SIZE_KV))

    softmax_factor = 1.0 / l_i
    #o_block = o_block * softmax_factor[:, None]

    #l_block = m_i + tl.log(l_i)

    tl.store(o_block_pointer, o_block)
    #tl.store(l_block_pointer, l_block)



class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: Float[Tensor, "batch_size seq_len d_k"], K, V, is_causal=False):
        BATCH_SIZE, SEQ_LEN, HEAD_DIM = Q.shape

        # O is like Q
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=Q.dtype)

        softmax_scale = 1 / math.sqrt(Q.shape[-1])
        stride_batch = SEQ_LEN * HEAD_DIM
        stride_sq = HEAD_DIM
        stride_dim = 1


        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE,
            1
        )
        _attn_fwd[grid](
            Q, K, V, O, L,
            stride_batch, stride_sq, stride_dim, softmax_scale, 
            SEQ_LEN, HEAD_DIM, 
            BLOCK_SIZE_Q = 16, BLOCK_SIZE_KV = 16,
            IS_CAUSAL = is_causal
        )
        # Save only L as required by tests
        ctx.save_for_backward(L)
        print(f"kq^t: {O}")
        return O
