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


_ENABLED = _get_bool_env("SGLANG_NAN_DIAG_ENABLE", False)
_LOG_ALL = _get_bool_env("SGLANG_NAN_DIAG_LOG_ALL", False)
_LOG_EVERY = max(1, _get_int_env("SGLANG_NAN_DIAG_LOG_EVERY", 1))
_MAX_LOGS = max(1, _get_int_env("SGLANG_NAN_DIAG_MAX_LOGS", 200))
_MAX_DUMPS = max(1, _get_int_env("SGLANG_NAN_DIAG_MAX_DUMPS", 8))
_DUMP_DIR = os.getenv("SGLANG_NAN_DIAG_DUMP_DIR")
_RANK_FILTER = _parse_rank_filter()

_GLOBAL_LOG_COUNT = 0
_GLOBAL_DUMP_COUNT = 0
_STAGE_CALL_COUNTER: Dict[str, int] = {}


def _is_rank_enabled() -> bool:
    if _RANK_FILTER is None:
        return True
    return _get_rank() in _RANK_FILTER


def _next_stage_call_index(stage: str) -> int:
    call_index = _STAGE_CALL_COUNTER.get(stage, 0) + 1
    _STAGE_CALL_COUNTER[stage] = call_index
    return call_index


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


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    data = tensor.detach()
    isnan = torch.isnan(data).any().item()
    isinf = torch.isinf(data).any().item()
    finite_mask = torch.isfinite(data)
    finite_count = int(finite_mask.sum().item())
    total_count = data.numel()

    stats = {
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "device": str(data.device),
        "isnan": bool(isnan),
        "isinf": bool(isinf),
        "finite_count": finite_count,
        "total_count": total_count,
    }

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
    global _GLOBAL_LOG_COUNT

    if not _ENABLED or tensor is None or not isinstance(tensor, torch.Tensor):
        return False
    if not _is_rank_enabled():
        return False

    call_index = _next_stage_call_index(stage)

    stats = _tensor_stats(tensor)
    has_non_finite = stats["isnan"] or stats["isinf"]
    should_log = has_non_finite or _LOG_ALL
    if not should_log:
        return False
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
    if extra:
        info.update(extra)

    if has_non_finite:
        logger.warning("NaNDiag anomaly: %s", info)
        _dump_tensor(stage, tensor, extra)
    else:
        logger.info("NaNDiag stats: %s", info)

    _GLOBAL_LOG_COUNT += 1
    return has_non_finite
