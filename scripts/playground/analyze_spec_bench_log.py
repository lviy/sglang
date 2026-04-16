#!/usr/bin/env python3
"""Analyze bench_speculative log output.

This script parses logs produced by scripts/playground/bench_speculative.py and
extracts per-configuration metrics from:
- Start lines
- Serving Benchmark Result blocks
- Finish lines

It exports flattened CSV/JSON and prints a short summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

START_RE = re.compile(
    r"^Start i=(?P<i>\d+): batch_size=(?P<batch_size>\d+), "
    r"steps=(?P<steps>\d+), topk=(?P<topk>\d+), num_draft_tokens=(?P<num_draft_tokens>\d+)"
)

FINISH_RE = re.compile(
    r"^Finish i=(?P<i>\d+): batch_size=(?P<batch_size>\d+), "
    r"steps=(?P<steps>\d+), topk=(?P<topk>\d+), num_draft_tokens=(?P<num_draft_tokens>\d+), "
    r"speed=(?P<speed>[0-9]+(?:\.[0-9]+)?) token/s, step_time=(?P<step_time>[0-9]+(?:\.[0-9]+)?) ms"
)

KEY_VALUE_RE = re.compile(r"^(?P<key>[^:]+):\s+(?P<value>.+?)\s*$")

RESULT_BEGIN = "============ Serving Benchmark Result ============"
RESULT_END_PREFIX = "=================================================="


def parse_numeric(value: str):
    v = value.strip()
    if v.lower() in {"none", "nan"}:
        return None
    if v.lower() == "inf":
        return math.inf
    # Remove trailing commas or percent signs if present.
    v = v.rstrip(",")
    if v.endswith("%"):
        v = v[:-1]
    try:
        if re.fullmatch(r"[-+]?\d+", v):
            return int(v)
        if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", v) or re.fullmatch(
            r"[-+]?\d+[eE][-+]?\d+", v
        ):
            return float(v)
    except ValueError:
        return value
    return value


@dataclass
class RunRecord:
    i: int
    batch_size: int
    steps: int
    topk: int
    num_draft_tokens: int
    start_line: int
    command: Optional[str] = None
    result_blocks: List[Dict[str, object]] = field(default_factory=list)
    finished: bool = False
    finish_line: Optional[int] = None
    speed_token_s: Optional[float] = None
    step_time_ms: Optional[float] = None


class SpecBenchLogParser:
    def __init__(self, keep_all_blocks: bool = True):
        self.keep_all_blocks = keep_all_blocks

    def parse(self, path: Path) -> List[RunRecord]:
        runs: List[RunRecord] = []
        cur: Optional[RunRecord] = None

        in_result = False
        current_result: Dict[str, object] = {}

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")

                m_start = START_RE.match(line)
                if m_start:
                    if cur is not None:
                        # Previous run did not finish.
                        runs.append(cur)
                    cur = RunRecord(
                        i=int(m_start.group("i")),
                        batch_size=int(m_start.group("batch_size")),
                        steps=int(m_start.group("steps")),
                        topk=int(m_start.group("topk")),
                        num_draft_tokens=int(m_start.group("num_draft_tokens")),
                        start_line=lineno,
                    )
                    in_result = False
                    current_result = {}
                    continue

                if cur is None:
                    continue

                if line.startswith("command="):
                    cur.command = line[len("command=") :].strip()
                    continue

                if line == RESULT_BEGIN:
                    in_result = True
                    current_result = {}
                    continue

                if in_result:
                    if line.startswith(RESULT_END_PREFIX):
                        if self.keep_all_blocks:
                            cur.result_blocks.append(current_result)
                        else:
                            cur.result_blocks = [current_result]
                        in_result = False
                        current_result = {}
                        continue

                    m_kv = KEY_VALUE_RE.match(line)
                    if m_kv:
                        k = m_kv.group("key").strip().lower().replace(" ", "_")
                        v = parse_numeric(m_kv.group("value"))
                        current_result[k] = v
                    continue

                m_finish = FINISH_RE.match(line)
                if m_finish:
                    finish_i = int(m_finish.group("i"))
                    if finish_i == cur.i:
                        cur.finished = True
                        cur.finish_line = lineno
                        cur.speed_token_s = float(m_finish.group("speed"))
                        cur.step_time_ms = float(m_finish.group("step_time"))
                        runs.append(cur)
                        cur = None
                        in_result = False
                        current_result = {}
                        continue

        # Flush dangling run
        if cur is not None:
            runs.append(cur)

        return runs


def to_flat_row(run: RunRecord, result_block_index: int = -1) -> Dict[str, object]:
    row: Dict[str, object] = {
        "i": run.i,
        "batch_size": run.batch_size,
        "steps": run.steps,
        "topk": run.topk,
        "num_draft_tokens": run.num_draft_tokens,
        "finished": run.finished,
        "start_line": run.start_line,
        "finish_line": run.finish_line,
        "num_result_blocks": len(run.result_blocks),
        "speed_token_s": run.speed_token_s,
        "step_time_ms": run.step_time_ms,
        "command": run.command,
    }

    block = None
    if run.result_blocks:
        try:
            block = run.result_blocks[result_block_index]
        except IndexError:
            block = run.result_blocks[-1]

    if block is not None:
        for k, v in block.items():
            row[f"metric_{k}"] = v

    return row


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(runs: List[RunRecord], path: Path) -> None:
    serializable = [
        {
            "i": r.i,
            "batch_size": r.batch_size,
            "steps": r.steps,
            "topk": r.topk,
            "num_draft_tokens": r.num_draft_tokens,
            "start_line": r.start_line,
            "command": r.command,
            "result_blocks": r.result_blocks,
            "finished": r.finished,
            "finish_line": r.finish_line,
            "speed_token_s": r.speed_token_s,
            "step_time_ms": r.step_time_ms,
        }
        for r in runs
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def best_by_metric(
    rows: List[Dict[str, object]],
    metric: str,
    descending: bool = True,
) -> Optional[Dict[str, object]]:
    candidates = [r for r in rows if isinstance(r.get(metric), (int, float))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[metric], reverse=descending)[0]


def summarize(rows: List[Dict[str, object]], topn: int = 8) -> str:
    completed = [r for r in rows if r.get("finished")]
    incomplete = [r for r in rows if not r.get("finished")]

    lines: List[str] = []
    lines.append(f"Total runs: {len(rows)} (completed={len(completed)}, incomplete={len(incomplete)})")

    if not completed:
        return "\n".join(lines)

    global_best_speed = best_by_metric(completed, "speed_token_s", descending=True)
    global_best_tput = best_by_metric(
        completed, "metric_output_token_throughput_(tok/s)", descending=True
    )

    if global_best_speed:
        lines.append(
            "Best by speed_token_s: "
            f"i={global_best_speed['i']} bs={global_best_speed['batch_size']} "
            f"steps={global_best_speed['steps']} topk={global_best_speed['topk']} "
            f"draft={global_best_speed['num_draft_tokens']} "
            f"speed={global_best_speed['speed_token_s']:.2f}"
        )

    if global_best_tput:
        lines.append(
            "Best by output_throughput: "
            f"i={global_best_tput['i']} bs={global_best_tput['batch_size']} "
            f"steps={global_best_tput['steps']} topk={global_best_tput['topk']} "
            f"draft={global_best_tput['num_draft_tokens']} "
            f"out_tput={global_best_tput['metric_output_token_throughput_(tok/s)']:.2f}"
        )

    lines.append("Top runs by speed_token_s:")
    ranked = sorted(
        [r for r in completed if isinstance(r.get("speed_token_s"), (int, float))],
        key=lambda x: x["speed_token_s"],
        reverse=True,
    )[:topn]

    for r in ranked:
        out_tput = r.get("metric_output_token_throughput_(tok/s)")
        ttft = r.get("metric_mean_ttft_(ms)")
        tpot = r.get("metric_mean_tpot_(ms)")
        lines.append(
            f"  i={r['i']:>2} bs={r['batch_size']:>3} cfg=({r['steps']},{r['topk']},{r['num_draft_tokens']}) "
            f"speed={r['speed_token_s']:.2f} tok/s, "
            f"out_tput={out_tput if isinstance(out_tput, (int, float)) else 'NA'}, "
            f"ttft={ttft if isinstance(ttft, (int, float)) else 'NA'} ms, "
            f"tpot={tpot if isinstance(tpot, (int, float)) else 'NA'} ms"
        )

    # Per-batch best config by speed.
    lines.append("Best config per batch_size by speed_token_s:")
    by_bs: Dict[int, List[Dict[str, object]]] = {}
    for r in completed:
        by_bs.setdefault(int(r["batch_size"]), []).append(r)

    for bs in sorted(by_bs):
        best = best_by_metric(by_bs[bs], "speed_token_s", descending=True)
        if not best:
            continue
        lines.append(
            f"  bs={bs}: cfg=({best['steps']},{best['topk']},{best['num_draft_tokens']}), "
            f"speed={best['speed_token_s']:.2f} tok/s, "
            f"step_time={best.get('step_time_ms', 'NA')} ms"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze bench_speculative log.")
    parser.add_argument("--log", required=True, help="Path to the benchmark log file.")
    parser.add_argument(
        "--out-dir",
        default="./spec_bench_analysis",
        help="Output directory for parsed files.",
    )
    parser.add_argument(
        "--result-block-index",
        type=int,
        default=-1,
        help=(
            "Which result block to use when flattening metrics per run. "
            "-1 means last block (recommended)."
        ),
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=8,
        help="Top N entries to print in terminal summary.",
    )
    parser.add_argument(
        "--latest-block-only",
        action="store_true",
        help="Only keep the latest benchmark result block per run in JSON output.",
    )
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parser_obj = SpecBenchLogParser(keep_all_blocks=not args.latest_block_only)
    runs = parser_obj.parse(log_path)

    rows = [to_flat_row(r, result_block_index=args.result_block_index) for r in runs]

    json_path = out_dir / "runs_full.json"
    csv_path = out_dir / "runs_flat.csv"
    summary_path = out_dir / "summary.txt"

    write_json(runs, json_path)
    write_csv(rows, csv_path)
    summary_text = summarize(rows, topn=args.topn)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    print(summary_text)
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
