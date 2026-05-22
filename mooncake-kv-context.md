# Agentic LLM Serving: KV Cache, Compute, and Compaction

## Source
[vLLM x Mooncake blog post](https://vllm.ai/blog/2026-05-06-mooncake-store) — integrating Mooncake's distributed KV cache store into vLLM for agentic workloads (3.8x throughput, 46x lower TTFT on Codex/SWE-bench traces).

---

## The Anatomy of an Agentic Turn

Their Figure 1 shows context broken into color-coded segments:

- **Cached prefix** (hatched): system prompt, skills, past agent decode, past tool output — KV already computed, no recompute needed
- **Orange**: new tool output this turn — must be prefilled, always cold
- **Dark teal**: decode output this turn

The causal chain that creates orange:
```
model decodes → "run grep foo bar.py"    (becomes cached next turn)
     ↓
executor runs it, stdout comes back
     ↓
that stdout = orange (model never saw it, no KV cache exists, must prefill)
```

Orange is the **environment's response** to the model's decoded tool call. The model didn't generate it — so no cache hit is ever possible for it. This is what makes agentic workloads structurally different from chat (where the "new input" is a small typed message).

---

## Variables

```
S    = system prompt tokens (fixed)
O_t  = orange tokens at turn t (new tool output, always cold)
D_t  = decode tokens at turn t (cached next turn)
W    = context window limit
P    = tokens preserved verbatim after compaction
C_t  = cached prefix length at turn t
```

---

## Context Growth

```
C_t = S + Σ(O_i + D_i)   for i < t
```

Linear in t. Grows by `O + D` per turn (assuming roughly constant).

---

## Cache Hit Rate

```
h_t = C_t / (C_t + O_t)
```

Without compaction, `h_t → 1` as t grows, because `C_t` grows but `O_t` stays bounded. This is the ceiling Mooncake is working toward. Mooncake's 92% cache hit rate (vs 1.7% baseline) reflects this — orange is only ~8% of input in their workload.

---

## Compute Cost

**Per-turn prefill cost** (with cache hits for the prefix):
```
prefill_cost_t  ∝  O_t * C_t
```
Each orange token must attend over the full cached context.

Since `C_t ∝ t`, this is `O(t)` per turn.

**Cumulative compute across all turns:**
```
Σ O * C_t  ≈  O * Σ t  =  O(t²)
```

**Cold-miss / full recompute case** (no cache, or post-compaction):
```
cost_t  ∝  C_t²  =  O(t²)  per turn
```
Because attention over full sequence of length `C_t` is quadratic. Cache hit rate is the difference between `O(t)` and `O(t²)` per-turn compute.

---

## Compaction

### When it triggers
```
T_c = (W - S) / (O_avg + D_avg)
```
Every `T_c` turns, total context hits `W` and compaction is forced.

### What it does to cache hit rate

After compaction, only `P` tokens still hit cache (preserved verbatim — system prompt, recent turns). The summary tokens `(C'_after - P)` are new sequences the model has never seen — cold prefill even though they represent old content:

```
h_compaction = P / (C_Tc + O_Tc)    ← hard drop
```

So `h_t` is **sawtooth**: climbs toward 1 over `T_c` turns, then resets to `≈ P/W` at each compaction.

### Amortized compaction cost per turn
```
extra_prefill_per_turn = (C'_after - P) / T_c
```

---

## Shared KV Blocks and Compaction Selection

In Mooncake's distributed pool, multiple sessions share KV blocks (same system prompt, same task template → same prefix → same blocks). The Mooncake Master tracks reference counts per block.

**The tension:** if session A compacts a portion of its context that session B is still using, session B's next prefill is a cold miss — even though those blocks were valid moments ago.

**The natural compaction priority** (not currently implemented by Mooncake, but implied):
```
compact first:  low-refcount blocks   (unique-to-session tool outputs, old state)
preserve:       high-refcount blocks  (system prompt, shared task template)
```

This aligns with semantic intent anyway — system prompts and shared templates are both the most shared AND the most worth preserving verbatim.

Open problem: race condition between compaction decisions and in-flight prefill lookups referencing the same blocks.

---

## Layer Separation

**Mooncake's actual scope:**
- Distributed KV block storage (DRAM/SSD across nodes)
- RDMA transfer of KV blocks (GPU HBM ↔ pool, zero-copy, SM-free)
- Block metadata + lookup routing via Mooncake Master
- Load balancing: route requests to instances that already hold the relevant prefix

**Outside Mooncake's scope:**
- Compaction decisions (agent framework or vLLM scheduler layer)
- Which tokens to summarize
- Awareness of shared-block reference counts at compaction time

Mooncake just sees the resulting different token sequence after compaction and treats it as a new cold prefix. The compaction sawtooth is invisible to it.

**Equations map to layers:**
```
C/(C+O) sawtooth ceiling    →  what Mooncake optimizes within a compaction cycle
compaction period T_c        →  upstream policy, Mooncake is blind to it
```

---

## Open Questions
- Optimal compaction strategy that accounts for block reference counts across sessions
- How to handle the race condition between compaction and in-flight prefill lookups
- What the actual `O_avg` distribution looks like for different agentic workloads (determines whether system is cache-bound or prefill-compute-bound)
- Whether `P/W` (preserved fraction after compaction) is tunable and what the right target is
