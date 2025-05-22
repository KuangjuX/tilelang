# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
# ruff: noqa
import math
import torch
import triton
import triton.language as tl
import flash_attn
import torch.nn.functional as F

import tilelang
from tilelang import language as T
from tilelang.profiler import do_bench


def is_hip():
    return False


def get_sparse_attn_mask_from_topk(x, topk, use_dense_for_last_block=False):
    bsz, num_head, downsample_len, _ = x.shape
    # N_CTX = downsample_len * BLOCK
    sparse_index = torch.topk(x, topk, dim=-1).indices
    dense_mask = torch.full([bsz, num_head, downsample_len, downsample_len],
                            False,
                            dtype=torch.bool,
                            device=x.device)
    dense_mask.scatter_(-1, sparse_index, True)
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


def get_sparse_attn_mask_from_threshold(x, threshold, use_dense_for_last_block=False):
    dense_mask = x > threshold
    if use_dense_for_last_block:
        dense_mask[:, :, -2:, :] = True
    dense_mask.tril_()
    return dense_mask


# TileLang implementation
def blocksparse_flashattn(batch, heads, seq_len, dim, downsample_len, is_causal):
    block_M = 64
    block_N = 64
    num_stages = 2
    threads = 128
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    block_mask_shape = [batch, heads, downsample_len, downsample_len]

    dtype = "float16"
    accum_dtype = "float"
    block_mask_dtype = "bool"

    def kernel_func(block_M, block_N, num_stages, threads):
        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape, dtype),
            V_shared: T.SharedBuffer([block_M, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                BlockSparseMask: T.Tensor(block_mask_shape, block_mask_dtype),
                Output: T.Tensor(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                block_mask = T.alloc_local([downsample_len], block_mask_dtype)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for vj in T.serial(downsample_len):
                    block_mask[vj] = BlockSparseMask[bz, by, bx, vj]

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    if block_mask[k]:
                        MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                        Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                                scores_sum, logsum)
                        Rescale(acc_o, scores_scale)
                        MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

        return main

    return kernel_func(block_M, block_N, num_stages, threads)


# Triton implementation
@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    k_block_col_idx,
    m_i_ptr,
    k_ptrs,
    v_ptrs,
    offs_m,
    offs_n,
    stride_kt,
    stride_vt,
    stride_bmask_n,
    sm_scale,
    seqlen_k,
    past_len,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    mask_val = tl.load(m_i_ptr + k_block_col_idx * stride_bmask_n)

    if mask_val == True:
        start_n = k_block_col_idx * BLOCK_N
        k = tl.load(k_ptrs + start_n * stride_kt)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if LAST_K_BLOCK:
            qk += tl.where(offs_m[:, None] + past_len >= (start_n + offs_n[None, :]), 0,
                           float('-inf'))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + start_n * stride_vt)
        p = p.to(v.type.element_ty)
        acc += tl.dot(p, v)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, block_mask_ptr, Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_bmz, stride_bmh, stride_bmm, stride_bmn,
    stride_oz, stride_oh, stride_om, stride_od,
    H, N_CTX, PAST_LEN,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    block_mask_ptr += off_z * stride_bmz + off_h * stride_bmh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    mask_ptrs = block_mask_ptr + start_m * stride_bmm

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)

    k_block_start = 0
    k_block_end = tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N)

    for col_idx in range(k_block_start, k_block_end):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc, l_i, m_i, q, col_idx, mask_ptrs, k_ptrs, v_ptrs,
            offs_m, offs_n, stride_kn, stride_vn, stride_bmn,
            sm_scale, N_CTX, PAST_LEN, col_idx == k_block_end - 1,
            BLOCK_M, BLOCK_N,
        )

    m_i += tl.math.log(l_i)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(Out.dtype.element_ty)

    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[
        None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def _forward(ctx, q, k, v, block_sparse_mask, sm_scale, BLOCK_M=64, BLOCK_N=64, num_warps=None, num_stages=1, out=None):
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    o = out if out is not None else torch.empty_like(q).contiguous()
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    assert q.shape[-1] in [64, 128]
    BLOCK_DMODEL = q.shape[-1]

    if is_hip():
        num_warps, num_stages = 8, 1
    else:
        num_warps, num_stages = 4, 2

    N_CTX = k.shape[2]
    PAST_LEN = N_CTX - q.shape[2]
    H = q.shape[1]

    _fwd_kernel[grid](
        q, k, v, sm_scale, block_sparse_mask, o,
        *q.stride(), *k.stride(), *v.stride(),
        *block_sparse_mask.stride(), *o.stride(),
        H, N_CTX, PAST_LEN,
        BLOCK_M, BLOCK_N, BLOCK_DMODEL,
        num_warps=num_warps, num_stages=num_stages,
    )

    return o


class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_sparse_dense, sm_scale):
        return _forward(ctx, q, k, v, block_sparse_dense, sm_scale)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("It does not support gradient propagation yet")
        return None, None, None, None, None


block_sparse_triton_fn = _sparse_attention.apply


def benchmark_block_sparse_fmha():
    from benchmark_configs import configs
    torch.manual_seed(0)

    # Config
    for BATCH, N_HEADS, SEQ_LEN, D_HEAD, TOPK, BLOCK in configs:
        print(f"\nBenchmarking with BATCH={BATCH}, N_HEADS={N_HEADS}, SEQ_LEN={SEQ_LEN}, D_HEAD={D_HEAD}, TOPK={TOPK}, BLOCK={BLOCK}")

        # Create inputs
        q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
        k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
        v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)

        sm_scale = 1.0 / (D_HEAD**0.5)

        # Create sparse mask (downsampled to block level)
        downsample_factor = BLOCK
        downsample_len = math.ceil(SEQ_LEN / downsample_factor)
        x_ds = torch.randn([BATCH, N_HEADS, downsample_len, downsample_len],
                           device='cuda',
                           dtype=torch.bfloat16)
        x_ds[:, :, :, 0] = 100
        block_mask = get_sparse_attn_mask_from_topk(x_ds, topk=TOPK)

        # Benchmark Flash Attention
        def flash_attn_fn():
            return flash_attn.flash_attn_func(q, k, v, causal=True)

        flash_attn_time = do_bench(flash_attn_fn, warmup=10, rep=100)
        print(f"Flash Attention: {flash_attn_time:.3f} ms")

        # Benchmark TileLang implementation
        program = blocksparse_flashattn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, downsample_len, is_causal=True)
        kernel = tilelang.compile(program, out_idx=4)

        def tilelang_fn():
            return kernel(q, k, v, block_mask)

        tilelang_time = do_bench(tilelang_fn, warmup=10, rep=100)
        print(f"TileLang: {tilelang_time:.3f} ms")

        # Benchmark Triton implementation
        def triton_fn():
            return block_sparse_triton_fn(q, k, v, block_mask, sm_scale)

        triton_time = do_bench(triton_fn, warmup=10, rep=100)
        print(f"Triton: {triton_time:.3f} ms")

        # Benchmark PyTorch implementation
        def torch_fn():
            full_mask = torch.kron(block_mask.float(), torch.ones(BLOCK, BLOCK, device='cuda'))
            full_mask = full_mask[..., :SEQ_LEN, :SEQ_LEN].bool()
            full_mask = full_mask & torch.tril(torch.ones_like(full_mask))
            attn = torch.einsum('bhsd,bhtd->bhst', q, k) * sm_scale
            attn = attn.masked_fill(~full_mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            return torch.einsum('bhst,bhtd->bhsd', attn, v)

        torch_time = do_bench(torch_fn, warmup=10, rep=100)
        print(f"PyTorch: {torch_time:.3f} ms")


if __name__ == "__main__":
    benchmark_block_sparse_fmha()




