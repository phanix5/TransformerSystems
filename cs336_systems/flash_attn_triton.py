from doctest import DONT_ACCEPT_BLANKLINE
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
    BLOCK_SIZE_KV: tl.constexpr
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

    # o_block = o_11 o_12 ... o_1d
    #           o_21 o_22 ... o_2d
    o_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # m_i = -inf
    #       -inf
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + float("-inf")

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

        # Next, apply mask
        # if block_idx_q == i and IS_CAUSAL:
        #     mask_i = tl.arange(BLOCK_SIZE_Q)
        #     mask_j = tl.arange(BLOCK_SIZE_Q)
        #     mask = mask_i[None, :] <= mask_j[:, None]
        #     kq_block = tl.where(mask, kq_block, float("-inf"))

        # Next, find max in block
        # m_ij = max(-inf, max(qk_11, qk_12))
        #        max(-inf, max(qk_21, qk_22))
        m_ij = tl.maximum(m_i, tl.max(kq_block, 1))

        # Softmax safety: subtract by max till now
        kq_block -= m_ij[:, None]

        # Next, find new l
        p_block = tl.math.exp(kq_block)

        # Sum the exponentials
        l_ij = tl.sum(p_block, 1)

        # correction factor exp(m_old - m_new)
        alpha = tl.math.exp(m_i - m_ij)

        # add to running sum of exps with correction factor
        l_i = l_ij + l_i * alpha

        o_block = o_block * alpha[:, None]
        o_block = tl.dot(p_block, v_block, o_block)

        m_i = m_ij

        v_block_pointer = tl.advance(v_block_pointer, (BLOCK_SIZE_KV, 0))
        k_block_pointer = tl.advance(k_block_pointer, (0, BLOCK_SIZE_KV))

    softmax_factor = 1.0 / l_i
    o_block = o_block * softmax_factor[:, None]

    l_block = m_i + tl.log(l_i)

    tl.store(o_block_pointer, o_block)
    tl.store(l_block_pointer, l_block)


@triton.jit
def _attn_fwd_causal(
    Q, K, V,
    O, L,
    stride_batch,
    stride_seq,
    stride_dim,
    softmax_scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr
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

    # o_block = o_11 o_12 ... o_1d
    #           o_21 o_22 ... o_2d
    o_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # m_i = -inf
    #       -inf
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + float("-inf")

    # l_i = 1
    #       1
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1

    # Only loop till K_i < Q_j
    for i in range(block_idx_q + 1):

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

        # Next, apply mask
        mask_i = block_idx_q * BLOCK_SIZE_Q + tl.arange(BLOCK_SIZE_Q)
        mask_j = i * BLOCK_SIZE_KV + tl.arange(BLOCK_SIZE_KV)
        mask = mask_i[:, None] >= mask_j[None, :]
        kq_block = tl.where(mask, kq_block, float("-inf"))

        # Next, find max in block
        # m_ij = max(-inf, max(qk_11, qk_12))
        #        max(-inf, max(qk_21, qk_22))
        m_ij = tl.maximum(m_i, tl.max(kq_block, 1))

        # Softmax safety: subtract by max till now
        kq_block -= m_ij[:, None]

        # Next, find new l
        p_block = tl.math.exp(kq_block)

        # Sum the exponentials
        l_ij = tl.sum(p_block, 1)

        # correction factor exp(m_old - m_new)
        alpha = tl.math.exp(m_i - m_ij)

        # add to running sum of exps with correction factor
        l_i = l_ij + l_i * alpha

        o_block = o_block * alpha[:, None]
        o_block = tl.dot(p_block, v_block, o_block)

        m_i = m_ij

        v_block_pointer = tl.advance(v_block_pointer, (BLOCK_SIZE_KV, 0))
        k_block_pointer = tl.advance(k_block_pointer, (0, BLOCK_SIZE_KV))

    softmax_factor = 1.0 / l_i
    o_block = o_block * softmax_factor[:, None]

    l_block = m_i + tl.log(l_i)

    tl.store(o_block_pointer, o_block)
    tl.store(l_block_pointer, l_block)

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)

    O_block = tl.load(
        O
        + index_batch * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) # (BLOCK_SIZE, HEAD_DIM)

    dO_block = tl.load(
        dO
        + index_batch * SEQ_LEN
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :]
    ) # (BLOCK_SIZE, HEAD_DIM)

    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptrs = D + index_batch * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)

@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V,
    softmax_scale,
    dO, dQ, dK, dV,
    L, D,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    num_warps,
    num_stages
):
    index_batch = tl.program_id(2)
    offset_batch = index_batch * SEQ_LEN * HEAD_DIM
    offset_batch_for_ld = index_batch * SEQ_LEN

    Q += offset_batch
    K += offset_batch
    V += offset_batch
    dO += offset_batch
    dQ += offset_batch
    dK += offset_batch
    dV += offset_batch

    L += offset_batch_for_ld
    D += offset_batch_for_ld

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(BLOCK_KV)
    offs_dim = tl.arange(0, HEAD_DIM)

    K_block = tl.load(K + offs_kv[:, None] * HEAD_DIM + offs_dim[None, :])
    V_block = tl.load(V + offs_kv[:, None] * HEAD_DIM + offs_dim[None, :])

    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    offs_q = tl.arange(0, BLOCK_Q)

    # load Q as transposed
    q_ptrs = Q + offs_q[:, None] + offs_dim[None, :] * HEAD_DIM
    qt_ptrs = tl.trans(q_ptrs)
    dO_ptrs = dO + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]

    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        qT_block = tl.load(qt_ptrs)

        # Load logsumexp
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        L_block = tl.load(L + offs_q)

        # KQ^T = P^T
        QKt_block = softmax_scale * tl.dot(K_block, qT_block)
        # Apply softmax, L_block[None, :] beacuse P is transposed
        Pt_block = tl.math.exp(QKt_block - L_block[None, :])
        dO_block = tl.load(dO_ptrs)

        if STAGE == 3:
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )
            Pt_block = tl.where(mask_block, Pt_block, 0.0)
        
        dV_block += tl.dot(Pt_block.to(tl.float16), dO_block)
        Di = tl.load(D + offs_q)

        # Calculate V_j * dO_i^T
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        
        
        dST_block = Pt_block * (dpT_block - Di[None, :])
        dST_block = dST_block.to(tl.float16)

        dK_block += softmax_scale * tl.dot(dST_block, tl.trans(qT_block))

        curr_q += BLOCK_Q
        qt_ptrs += BLOCK_Q * HEAD_DIM
        dO_ptrs += BLOCK_Q * HEAD_DIM
    
    # Write the dV and DK blocks
    dV_block_ptrs = dV + offs_kv[:, None] * HEAD_DIM + offs_dim[None, :]
    tl.store(dV_block_ptrs, dV_block)


    dK_block_ptrs = dK + offs_kv[:, None] * HEAD_DIM + offs_dim[None, :]
    tl.store(dK_block_ptrs, dV_block)
        

@triton.jit
def _attn_bwd_dq(
    Q, K, V,
    softmax_scale,
    dO, dQ, dK, dV,
    L, D,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
    num_warps,
    num_stages
):
    index_batch = tl.program_id(2)
    offset_batch = index_batch * SEQ_LEN * HEAD_DIM
    offset_batch_for_ld = index_batch * SEQ_LEN

    Q += offset_batch
    K += offset_batch
    V += offset_batch
    dO += offset_batch
    dQ += offset_batch
    dK += offset_batch
    dV += offset_batch

    L += offset_batch_for_ld
    D += offset_batch_for_ld

    index_block_q = tl.program_id(0)
    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    offs_dim = tl.arange(0, HEAD_DIM)

    Q_block = tl.load(Q + offs_q[:, None] * HEAD_DIM + offs_dim[None, :])
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    dO_block = tl.load(dO + offs_q[:, None] * HEAD_DIM + offs_dim[None, :])

    L_block = tl.load(L + offs_q)
    L_block = L_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    kt_ptrs = K + offs_kv[None, :] * HEAD_DIM + offs_dim[:, None]
    vt_ptrs = V + offs_kv[None, :] * HEAD_DIM + offs_dim[:, None]

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV

    for bulk_id in range(num_steps):
        Kt_block = tl.load(kt_ptrs)
        Vt_block = tl.load(vt_ptrs)

        L_block = tl.load(L + offs_q)

        # Calculate P

        QK_block = softmax_scale * tl.dot(Q_block, Kt_block)
        P_block = tl.math.exp(QK_block - L_block) # BLOCK_Q x BLOCK_KV

        if STAGE == 3:
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            )
            P_block = tl.where(mask_block, P_block, 0.0)
        
        dp_block = tl.dot(dO_block, Vt_block).to(tl.float32) # BLOCK_Q x BLOCK_KV

        dS_block = P_block * (dp_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale*tl.dot(dS_block, tl.trans(Kt_block))

        curr_kv += BLOCK_KV
        kt_ptrs += BLOCK_KV * HEAD_DIM
        vt_ptrs += BLOCK_KV * HEAD_DIM
    
    # Write Dq blocks
    dq_block_ptrs = dQ + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]
    tl.store(dq_block_ptrs, dQ_block)

        


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
        kernel = _attn_fwd_causal if is_causal else _attn_fwd
        kernel[grid](
            Q, K, V, O, L,
            stride_batch, stride_sq, stride_dim, softmax_scale,
            SEQ_LEN, HEAD_DIM,
            BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=16,
        )
        # Save only L as required by tests
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.causal = is_causal
        ctx.softamax_scale = softmax_scale
        print(f"Debug: {O}")
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, SEQ_LEN, HEAD_DIM = Q.shape
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE)
        D = torch.empty_like(L)
        
        # Compute Di = dOi * O
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN= SEQ_LEN,
            BLOCK_SIZE_Q = BLOCK_SIZE_MACRO,
            HEAD_DIM= HEAD_DIM
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE)

        stage = 3 if ctx.causal else 1

        _attn_bwd_dk_dv[grid](
            Q, K, V,
            ctx.softmax_scale,
            dO, dQ, dK, dV,
            L, D,
            SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )

        print(f"Debug Triton dV: {dV}")

        _attn_bwd_dq[grid](
            Q, K, V,
            ctx.softmax_scale,
            dO, dQ, dK, dV,
            L, D,
            SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )

        return dO, dK, dV, None, None






