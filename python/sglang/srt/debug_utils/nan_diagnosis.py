import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _parse_rank_filter() -> Optional[set[int]]:
    value = os.getenv("SGLANG_NAN_DIAG_RANK_FILTER")
    if not value:
        return None
    value = value.strip().lower()
    if value in ("all", "*"):
        return None
    ranks = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ranks.add(int(item))
        except ValueError:
            continue
    return ranks if ranks else None


def _parse_stage_filter(name: str) -> Optional[list[str]]:
    value = os.getenv(name)
    if not value:
        return None
    items = [x.strip() for x in value.split(",")]
    items = [x for x in items if x]
    return items if items else None


_ENABLED = _get_bool_env("SGLANG_NAN_DIAG_ENABLE", False)
_LOG_ALL = _get_bool_env("SGLANG_NAN_DIAG_LOG_ALL", False)
_LOG_EVERY = max(1, _get_int_env("SGLANG_NAN_DIAG_LOG_EVERY", 1))
_ANOMALY_LOG_EVERY = max(1, _get_int_env("SGLANG_NAN_DIAG_ANOMALY_LOG_EVERY", 1))
_MAX_LOGS = max(1, _get_int_env("SGLANG_NAN_DIAG_MAX_LOGS", 200))
_MAX_ANOMALY_LOGS = max(1, _get_int_env("SGLANG_NAN_DIAG_MAX_ANOMALY_LOGS", 80))
_MAX_ANOMALY_LOGS_PER_STAGE = max(
    1, _get_int_env("SGLANG_NAN_DIAG_MAX_ANOMALY_LOGS_PER_STAGE", 2)
)
_MAX_DUMPS = max(1, _get_int_env("SGLANG_NAN_DIAG_MAX_DUMPS", 8))
_DUMP_DIR = os.getenv("SGLANG_NAN_DIAG_DUMP_DIR")
_RANK_FILTER = _parse_rank_filter()
_STAGE_INCLUDE = _parse_stage_filter("SGLANG_NAN_DIAG_STAGE_INCLUDE")
_STAGE_EXCLUDE = _parse_stage_filter("SGLANG_NAN_DIAG_STAGE_EXCLUDE")

_GLOBAL_LOG_COUNT = 0
_GLOBAL_ANOMALY_LOG_COUNT = 0
_GLOBAL_DUMP_COUNT = 0
_STAGE_CALL_COUNTER: Dict[str, int] = {}
_STAGE_ANOMALY_COUNTER: Dict[str, int] = {}
_STAGE_ANOMALY_LOG_COUNTER: Dict[str, int] = {}


def _is_rank_enabled() -> bool:
    if _RANK_FILTER is None:
        return True
    return _get_rank() in _RANK_FILTER


def _is_stage_enabled(stage: str) -> bool:
    if _STAGE_INCLUDE is not None and not any(
        stage.startswith(prefix) for prefix in _STAGE_INCLUDE
    ):
        return False
    if _STAGE_EXCLUDE is not None and any(
        stage.startswith(prefix) for prefix in _STAGE_EXCLUDE
    ):
        return False
    return True


def _is_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _next_stage_call_index(stage: str) -> int:
    call_index = _STAGE_CALL_COUNTER.get(stage, 0) + 1
    _STAGE_CALL_COUNTER[stage] = call_index
    return call_index


def _next_stage_anomaly_index(stage: str) -> int:
    anomaly_index = _STAGE_ANOMALY_COUNTER.get(stage, 0) + 1
    _STAGE_ANOMALY_COUNTER[stage] = anomaly_index
    return anomaly_index


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def maybe_log_event(
    stage: str,
    logger: logging.Logger,
    extra: Optional[Dict[str, object]] = None,
    force: bool = False,
) -> bool:
    """Log a structured checkpoint event for flow debugging.

    The event is active only when NaN diagnosis is enabled and rank filter matches.
    """
    global _GLOBAL_LOG_COUNT

    if not _ENABLED or not _is_rank_enabled():
        return False
    if not _is_stage_enabled(stage):
        return False

    if _GLOBAL_LOG_COUNT >= _MAX_LOGS:
        return False

    call_index = _next_stage_call_index(stage)
    if not force:
        if not _LOG_ALL:
            return False
        if call_index % _LOG_EVERY != 0:
            return False
        if _GLOBAL_LOG_COUNT >= _MAX_LOGS:
            return False

    info = {
        "stage": stage,
        "rank": _get_rank(),
        "call_index": call_index,
    }
    if extra:
        info.update(extra)

    if force:
        logger.warning("NaNDiag checkpoint: %s", info)
    else:
        logger.info("NaNDiag checkpoint: %s", info)

    _GLOBAL_LOG_COUNT += 1
    return True


def maybe_log_invariant(
    stage: str,
    ok: bool,
    logger: logging.Logger,
    extra: Optional[Dict[str, object]] = None,
) -> bool:
    """Log an invariant failure as a high-priority checkpoint."""
    if ok:
        return True
    maybe_log_event(stage, logger, extra=extra, force=True)
    return False


def maybe_cuda_synchronize(
    enabled: bool,
    stage: str,
    logger: logging.Logger,
    extra: Optional[Dict[str, object]] = None,
) -> bool:
    """Conditionally synchronize CUDA for race diagnosis."""
    if not enabled or not torch.cuda.is_available():
        return False
    if _is_stream_capturing():
        maybe_log_event(
            stage,
            logger,
            extra={**(extra or {}), "sync_skipped_stream_capturing": True},
            force=True,
        )
        return False

    torch.cuda.synchronize()
    maybe_log_event(
        stage,
        logger,
        extra={**(extra or {}), "cuda_synchronized": True},
        force=True,
    )
    return True


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    data = tensor.detach()
    finite_mask = torch.isfinite(data)
    finite_count = int(finite_mask.sum().item())
    total_count = data.numel()
    non_finite_count = total_count - finite_count
    isnan = False
    isinf = False

    stats = {
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "isnan": False,
        "isinf": False,
        "non_finite_count": non_finite_count,
        "finite_count": finite_count,
        "total_count": total_count,
    }

    if non_finite_count > 0:
        nan_mask = torch.isnan(data)
        inf_mask = torch.isinf(data)
        isnan = bool(nan_mask.any().item())
        isinf = bool(inf_mask.any().item())
        non_finite_mask = ~finite_mask
        first_non_finite_flat_idx = int(non_finite_mask.reshape(-1).nonzero()[0].item())
        stats["isnan"] = isnan
        stats["isinf"] = isinf
        stats["nan_count"] = int(nan_mask.sum().item())
        stats["inf_count"] = int(inf_mask.sum().item())
        stats["first_non_finite_flat_idx"] = first_non_finite_flat_idx

        if data.ndim >= 2:
            row_mask = non_finite_mask.reshape(data.shape[0], -1).any(dim=1)
            non_finite_row_count = int(row_mask.sum().item())
            stats["non_finite_row_count"] = non_finite_row_count
            if non_finite_row_count > 0:
                first_non_finite_row = int(row_mask.nonzero()[0].item())
                first_row_mask = non_finite_mask.reshape(data.shape[0], -1)[
                    first_non_finite_row
                ]
                stats["first_non_finite_row"] = first_non_finite_row
                stats["first_non_finite_row_non_finite_count"] = int(
                    first_row_mask.sum().item()
                )

    if finite_count > 0:
        finite_data = data[finite_mask]
        stats["min"] = float(finite_data.min().item())
        stats["max"] = float(finite_data.max().item())
        stats["mean"] = float(finite_data.float().mean().item())
        stats["std"] = float(finite_data.float().std(unbiased=False).item())
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
        stats["std"] = None

    return stats


def _dump_tensor(
    stage: str,
    tensor: torch.Tensor,
    extra: Optional[Dict[str, object]],
) -> None:
    global _GLOBAL_DUMP_COUNT

    if not _DUMP_DIR or _GLOBAL_DUMP_COUNT >= _MAX_DUMPS:
        return

    dump_dir = Path(_DUMP_DIR)
    dump_dir.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    rank = _get_rank()
    filename = f"{ts_ms}_rank{rank}_pid{os.getpid()}_{stage}.pt"
    path = dump_dir / filename
    payload = {
        "stage": stage,
        "rank": rank,
        "pid": os.getpid(),
        "tensor": tensor.detach().cpu(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    _GLOBAL_DUMP_COUNT += 1


def maybe_log_tensor_stats(
    stage: str,
    tensor: Optional[torch.Tensor],
    logger: logging.Logger,
    extra: Optional[Dict[str, object]] = None,
) -> bool:
    global _GLOBAL_LOG_COUNT, _GLOBAL_ANOMALY_LOG_COUNT

    if not _ENABLED or tensor is None or not isinstance(tensor, torch.Tensor):
        return False
    # CUDA graph capture forbids host sync paths like .item() used by stats.
    if _is_stream_capturing():
        return False
    if not _is_rank_enabled():
        return False
    if not _is_stage_enabled(stage):
        return False

    call_index = _next_stage_call_index(stage)

    stats = _tensor_stats(tensor)
    has_non_finite = stats["isnan"] or stats["isinf"]
    anomaly_index = _next_stage_anomaly_index(stage) if has_non_finite else None
    should_log = has_non_finite or _LOG_ALL
    if not should_log:
        return False
    if has_non_finite and (anomaly_index - 1) % _ANOMALY_LOG_EVERY != 0:
        return True
    if has_non_finite:
        if _GLOBAL_ANOMALY_LOG_COUNT >= _MAX_ANOMALY_LOGS:
            return True
        stage_anomaly_log_count = _STAGE_ANOMALY_LOG_COUNTER.get(stage, 0)
        if stage_anomaly_log_count >= _MAX_ANOMALY_LOGS_PER_STAGE:
            return True
    if not has_non_finite and call_index % _LOG_EVERY != 0:
        return False
    if _GLOBAL_LOG_COUNT >= _MAX_LOGS and not has_non_finite:
        return has_non_finite

    rank = _get_rank()
    info = {
        "stage": stage,
        "rank": rank,
        "call_index": call_index,
        **stats,
    }
    if anomaly_index is not None:
        info["anomaly_index"] = anomaly_index
    if extra:
        info.update(extra)
    if has_non_finite and "first_non_finite_row" in stats:
        first_row = _safe_int(stats.get("first_non_finite_row"))
        local_start = _safe_int(info.get("local_start_pos"))
        local_tokens = _safe_int(info.get("local_num_tokens"))
        if (
            first_row is not None
            and local_start is not None
            and local_tokens is not None
            and local_tokens >= 0
        ):
            local_end = local_start + local_tokens
            info["local_row_begin"] = local_start
            info["local_row_end_exclusive"] = local_end
            info["first_non_finite_in_local_range"] = (
                local_start <= first_row < local_end
            )
        rows_per_shard = _safe_int(info.get("rows_per_attn_dp_shard"))
        if first_row is not None and rows_per_shard is not None and rows_per_shard > 0:
            info["first_non_finite_est_attn_dp_shard"] = first_row // rows_per_shard
            info["first_non_finite_row_in_est_shard"] = first_row % rows_per_shard

    if has_non_finite:
        logger.warning("NaNDiag anomaly: %s", info)
        _GLOBAL_ANOMALY_LOG_COUNT += 1
        _STAGE_ANOMALY_LOG_COUNTER[stage] = (
            _STAGE_ANOMALY_LOG_COUNTER.get(stage, 0) + 1
        )
        _dump_tensor(stage, tensor, extra)
    else:
        logger.info("NaNDiag stats: %s", info)

    _GLOBAL_LOG_COUNT += 1
    return has_non_finite
