# Fresh Prefill Bottleneck Analysis

## Motivation

In agentic LLM serving, each turn of a multi-turn session introduces fresh tokens (tool outputs) that must be processed against a large cached context. The question: **when does attention computation over these fresh tokens become the bottleneck, rather than loading/creating the KV cache?**

Mooncake (Moonshot AI, USENIX FAST 2025 best paper) reports that fresh tool output ("orange tokens") is typically ~500 tokens (~8% of input) and treats it as negligible. We investigate whether this holds on Grace Hopper unified memory hardware, particularly as context lengths grow.

---

## Setup

### Hardware: NVIDIA Grace Hopper (GH200)

| Parameter | Value |
|---|---|
| HBM3 bandwidth | ~4800 GB/s |
| CPU-GPU unified memory (C2C) | 900 GB/s |
| FP16 dense TFLOPS | ~989 TFLOPS |
| Effective TFLOPS (70% efficiency) | ~692 TFLOPS |

**Simplifying assumption:** All KV cache resides on the CPU side of unified memory, so cache retrieval uses the 900 GB/s C2C bandwidth. This is conservative — if some cache is in HBM, memory access is faster, making the compute bottleneck even more pronounced.

### Model: Qwen2.5-7B-Instruct

| Parameter | Value |
|---|---|
| Layers (L) | 28 |
| Hidden size (H) | 3584 |
| Heads | 28 |
| Head dim | 128 |
| Precision | FP16 (2 bytes/element) |

---

## Key Concept: Why KV Caching Works

Autoregressive LLMs use **causal masking** — each token can only attend to tokens at the same or earlier positions. This creates a lower-triangular attention mask.

The consequence: when new tokens are added, old tokens' K and V projections don't change (they never see the new tokens). So we can cache and reuse them, only computing K and V for the fresh tokens.

This transforms what would be a full N x N attention computation into a much smaller problem when most of the context is cached.

---

## Three Attention Regimes

Let **s** = fresh (new) tokens, **c** = cached tokens, **N** = s + c total.

### 1. Full Prefill (c = 0)

Processing s tokens from scratch, no cache.

- Attention matrix shape: **s x s** (square)
- Compute: O(s^2) — quadratic in sequence length
- No memory loading — everything is computed fresh
- Bottleneck is purely compute vs compute (attention vs KV projection)

### 2. Decode (s = 1)

Single token generation against full cached context.

- Attention matrix shape: **1 x c** (thin row)
- Compute: O(c) — linear in context
- Memory: O(c) — must load full KV cache
- Bottleneck is almost always memory-bound (loading KV cache)

### 3. Mixed Prefill (s > 1, c > 0)

Fresh tokens processed against cached context. This is the agentic multi-turn case.

- Attention matrix shape: **s x (c + s)** (wide rectangle)
- Compute: O(s*c + s^2)
- Memory: O(c) — load cached KV
- The interesting regime: transition between memory-bound and compute-bound

**Key insight:** Mixed prefill is mechanically the same as decode, except with s > 1 query tokens instead of 1. It is NOT the same as batched decode — all s tokens share the same KV cache (single session), whereas batched decode has independent caches per request.

---

## Cost Derivations

### Single-layer costs (Qwen2.5-7B-Instruct)

All costs below are per single transformer layer. With H = 3584 and head_dim = 128:

**KV cache memory loading (cached tokens):**
Each cached token stores K and V vectors. Per layer in FP16:

    Bytes = c * head_dim * num_kv_heads * 2 (K,V) * 2 (FP16)

For Qwen2.5-7B: **2048 * c bytes** per layer.

**KV compute for fresh tokens (Q, K, V projections):**
Each fresh token requires linear projections through weight matrices.

    FLOPs = s * H * (H_q + H_k + H_v)

For Qwen2.5-7B: **3,633,152 * s FLOPs** per layer.

**Attention computation:**
QK^T and attention @ V for fresh tokens against all tokens (cached + fresh).

    FLOPs per layer = 2 * num_heads * head_dim * s * (c + s)
                     = 2 * head_dim * num_heads * (s*c + s^2)

For Qwen2.5-7B: **7096 * (s^2 + s*c) FLOPs** per layer (absorbing the factor of 2 into the coefficient).

---

## Converting to Time

Divide each cost by its respective hardware throughput to get seconds:

**KV cache loading time (memory-bound):**

    T_mem = 2048 * c / (900 * 10^9) seconds

Normalizing (multiply through by 10^9 / 2048):

    T_mem_normalized = c / (900 * 10^9 / 2048) ≈ 2.28 * c  (in microseconds)

**KV compute time (compute-bound):**

    T_kv_comp = 3,633,152 * s / (989 * 10^12) seconds

Normalizing:

    T_kv_comp_normalized ≈ 3.67 * s  (in microseconds)

**Attention time (compute-bound):**

    T_attn = 7096 * (s^2 + s*c) / (989 * 10^12) seconds

Normalizing:

    T_attn_normalized ≈ 0.00718 * (s^2 + s*c)  (in microseconds)

---

## The Bottleneck Inequality

The system bottleneck is whichever takes longer. We compare attention time vs KV creation time (memory loading + KV compute):

**Attention dominates when:**

    0.00718 * (s^2 + s*c) > 2.28*c + 3.67*s

### Normalized form

Dividing both sides by 0.00718 (equivalently, by 7096 in the un-normalized form):

    s^2 + s*c > 317*c + 511*s

Rearranging:

    s^2 + s*c - 317*c - 511*s > 0
    s*(s + c - 511) > 317*c

### Finding the boundary

Setting the inequality to equality and solving for s at a given c:

    0.00718*s^2 + 0.00718*s*c - 3.67*s - 2.28*c = 0
    0.00718*s^2 + (0.00718*c - 3.67)*s - 2.28*c = 0

Using the quadratic formula (a = 0.00718, b = 0.00718c - 3.67, solve for s):

    s = [-b + sqrt(b^2 + 4*0.00718*2.28*c)] / (2*0.00718)

### Limiting cases

**When c is large (c >> s):**

The s*c term dominates the left side, and 2.28*c dominates the right:

    0.00718*s*c > 2.28*c
    s > 2.28 / 0.00718 ≈ 318

**When c is small (c → 0, pure prefill):**

    0.00718*s^2 > 3.67*s
    s > 3.67 / 0.00718 ≈ 511

So the boundary ranges from ~511 (no cache) to ~318 (large cache), explaining why it appears nearly vertical on the log-log plot.

---

## Key Findings

1. **The boundary is nearly vertical at ~318 fresh tokens.** Once s exceeds ~318, attention computation dominates regardless of how large c is. Context size barely matters.

2. **Decode (s = 1) is always memory-bound.** At s = 1, the ratio of attention to KV loading time is ~0.00718/2.28 ≈ 0.003 — memory loading is ~300x slower than the attention compute. This is consistent with the well-known understanding.

3. **Pure prefill transitions around ~511 tokens.** Without any cache, the question is whether attention (O(s^2)) or KV projection (O(s)) dominates. Both are compute, but attention's quadratic scaling wins past ~511 tokens.

4. **The agentic regime is attention-bound.** Mooncake reports typical tool outputs of ~500 tokens. At s = 500, we are right at the boundary. Any tool output larger than this — or any scenario where multiple tool outputs accumulate — pushes firmly into the attention-dominated regime.

5. **This holds under conservative assumptions.** We used the slower 900 GB/s C2C bandwidth for memory. If any KV cache is in HBM (~4800 GB/s), memory access is even faster, lowering the threshold further and making the compute bottleneck more pronounced.

---

## Coefficients Summary

| Quantity | Expression | Coefficient (microseconds) |
|---|---|---|
| KV cache memory loading | 2048 * c bytes / 900 GB/s | 2.28 * c |
| KV projection compute | 3,633,152 * s FLOPs / 989 TFLOPS | 3.67 * s |
| Attention compute | 7096 * (s^2 + s*c) FLOPs / 989 TFLOPS | 0.00718 * (s^2 + s*c) |

---

## Plot Description

The bottleneck regime map plots s (fresh tokens) on the x-axis and c (cached tokens) on the y-axis, both in log scale. A continuous blue-red colormap shows the ratio of attention time to KV creation time. Blue regions are KV-creation-dominated; red regions are attention-dominated. The boundary (ratio = 1) is a near-vertical line at s ≈ 318, indicating that the transition depends almost entirely on the number of fresh tokens and is largely independent of context size.

---

## Context: Mooncake

Mooncake solved a different problem — **cross-instance KV cache sharing** in distributed serving. Their baseline 1.7% cache hit rate was caused by load balancers routing the same session to different GPU instances with isolated memory pools. They built an RDMA-based distributed cache (KVPool) to share KV cache across nodes, achieving 92.2% hit rates.

On a single unified-memory node (like GH200), cache hit rates are naturally high (no cross-instance routing problem), making the fresh compute cost the next bottleneck to investigate — which is what this analysis addresses.

---

## Next Steps

- Validate these predictions experimentally on GH200 with SWE-bench agentic traces
- Profile actual attention kernel behavior for mixed prefill shapes
- Investigate KV cache compaction and its effect on the s/c ratio
- Consider weight streaming overhead that may dominate in some regimes
