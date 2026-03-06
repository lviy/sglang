# DP-Attention NaN Debug Guide

This note is for debugging intermittent NaNs that only reproduce when `dp_attn` is enabled.

## Scope

Use this guide when the symptom matches most of the following:

- The issue reproduces only with `--enable-dp-attention`.
- `rollout-only` can still reproduce the issue.
- The first visible NaN appears around `self_attn -> prepare_mlp`, not inside Megatron training checks.
- Adding dumps or synchronizations changes the reproduction speed.
- Masking extra padding rows mitigates the issue.

Given the above, the current working assumption is:

1. `dp_attn` metadata or copy bounds are wrong for some batches.
2. `dp_attn` collective paths have alias or async race behavior.
3. Padding rows or row-boundary metadata are inconsistent across stages.
4. Pure MLA numerical instability is lower priority.

## Relevant Code Paths

- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/layers/communicator.py`
- `python/sglang/srt/layers/dp_attention.py`
- `python/sglang/srt/debug_utils/nan_diagnosis.py`

## Recommended Debug Order

### 1. Check row-boundary and padding mismatches first

This catches cases where the runtime thinks a tensor has more valid rows than it really does.

Recommended env vars:

```bash
SGLANG_NAN_DIAG_ENABLE=1
SGLANG_NAN_DIAG_BOUNDARY_ROW_CHECK_ENABLE=1
SGLANG_NAN_DIAG_BOUNDARY_ROW_CHECK_ASSERT=1
SGLANG_NAN_DIAG_MLA_GUARD_PADDING=1
SGLANG_NAN_DIAG_MLP_GUARD_PADDING=1
SGLANG_NAN_DIAG_DEEPSEEK_BLOCK_ENABLE=1
SGLANG_NAN_DIAG_DEEPSEEK_BLOCK_LAYER_WATCH=0,1
```

Interpretation:

- If this fails before any NaN dump, the problem is likely in row metadata, padding rows, or `out_cache_loc` / `qo_indptr` mismatch.
- If guard-padding greatly delays reproduction, stale tail rows are probably part of the bug.

### 2. Check DP copy bounds

This validates the copy metadata used by `dp_gather` and `dp_scatter`.

Recommended env vars:

```bash
SGLANG_NAN_DIAG_ENABLE=1
SGLANG_NAN_DIAG_ASSERT_DP_COPY_BOUNDS=1
```

This adds runtime checks before `memcpy_triton`:

- `local_start_pos >= 0`
- `local_num_tokens >= 0`
- source rows are sufficient
- destination rows are sufficient

Interpretation:

- If you hit `dp_gather_allreduce_copy_bounds` or `dp_scatter_copy_bounds`, treat it as a metadata or offset bug first.
- At that point, compare `local_start_pos`, `local_num_tokens`, `global_rows`, `q_total_rows_from_qo_indptr`, and `out_cache_loc_rows`.

### 3. Check collective alias or race behavior

This isolates whether `reduce_scatter` is unsafe when its output comes from a view into the input storage.

Recommended env vars:

```bash
SGLANG_NAN_DIAG_ENABLE=1
SGLANG_NAN_DIAG_FORCE_NO_ALIAS_REDUCE_SCATTER=1
```

Interpretation:

- If reproduction disappears or is greatly delayed, the likely issue is collective aliasing or a race in the `reduce_scatter` path.
- If behavior is unchanged, aliasing is less likely to be the primary cause.

### 4. Check whether synchronization changes the symptom

This helps determine whether the bug is an async boundary issue rather than a pure compute issue.

Recommended env vars:

```bash
SGLANG_NAN_DIAG_ENABLE=1
SGLANG_NAN_DIAG_SYNC_AFTER_SELF_ATTN=1
SGLANG_NAN_DIAG_SYNC_BEFORE_PREPARE_MLP=1
SGLANG_NAN_DIAG_SYNC_BEFORE_DP_GATHER=1
SGLANG_NAN_DIAG_SYNC_AFTER_DP_GATHER=1
```

Interpretation:

- If synchronization makes the issue much slower or much harder to reproduce, prioritize async or race explanations.
- If synchronization has no effect, focus more on deterministic metadata or padding mismatch.

## Useful Optional Filters

To reduce noise while keeping the first anomaly visible:

```bash
SGLANG_NAN_DIAG_RANK_FILTER=0,1
SGLANG_NAN_DIAG_STAGE_INCLUDE=deepseek_layer_hidden_post_self_attn,comm_prepare_mlp,comm_mlp_core,dp_gather
SGLANG_NAN_DIAG_MAX_LOGS=200
SGLANG_NAN_DIAG_MAX_ANOMALY_LOGS=80
```

## Suggested Run Matrix

Run these in order instead of enabling everything at once:

1. Boundary and padding checks only.
2. Copy-bounds assert only.
3. No-alias reduce-scatter only.
4. Synchronization only.
5. Combinations that showed the strongest effect.

This makes it easier to distinguish:

- metadata bug
- padding propagation
- collective alias
- async race

## How To Read The Outcome

- Copy-bounds assert fires:
  The highest-priority suspect is wrong `dp_attn` metadata or row offsets.

- No-alias reduce-scatter helps:
  The highest-priority suspect is collective alias or output-buffer race.

- Guard-padding helps:
  The highest-priority suspect is stale or invalid tail rows.

- Synchronization helps:
  The highest-priority suspect is async boundary behavior.

- None of the above help:
  Revisit lower-priority explanations such as backend-specific MLA behavior.

## Current Summary

For the current DeepSeek + `dp_attn` investigation, the intended priority is:

1. `dp_attn` metadata or offset mismatch
2. `dp_attn` collective alias or race
3. Other causes
