#!/usr/bin/env python3
"""统计有效 related code 的 instance，并输出对应 ENRE entity 计数。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

RELATION_KEYS = [
    "calls",
    "called_by",
    "inherits_from",
    "implements",
    "instantiated_in",
    "subclasses",
]

ENTITY_RE = re.compile(r'"entityNum"\s*:\s*\{(?P<body>.*?)\}', re.DOTALL)
KV_RE = re.compile(r'"(?P<key>[^"]+)"\s*:\s*(?P<value>\d+)')


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict, str, tuple, set)):
        return len(value) > 0
    return bool(value)


def is_valid_retrieved_context(ctx: dict[str, Any]) -> bool:
    total_relations = ctx.get("total_relations")
    if isinstance(total_relations, (int, float)) and total_relations > 0:
        return True

    for key in RELATION_KEYS:
        if is_non_empty(ctx.get(key)):
            return True

    return False


def has_valid_related_code(record: dict[str, Any]) -> bool:
    retrieved_code = record.get("retrieved_code")
    if not isinstance(retrieved_code, list):
        return False

    for item in retrieved_code:
        if not isinstance(item, dict):
            continue
        ctx = item.get("retrieved_context")
        if isinstance(ctx, dict) and is_valid_retrieved_context(ctx):
            return True

    return False


def extract_entity_num(report_path: Path) -> dict[str, int]:
    file_size = report_path.stat().st_size
    read_size = min(file_size, 1024 * 1024)

    with report_path.open("rb") as f:
        f.seek(file_size - read_size)
        tail = f.read(read_size).decode("utf-8", errors="ignore")

    matches = list(ENTITY_RE.finditer(tail))
    if matches:
        body = matches[-1].group("body")
        return {m.group("key"): int(m.group("value")) for m in KV_RE.finditer(body)}

    # 兜底：少数报告如果 entityNum 不在最后 1MB，回退为全文件扫描。
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(ENTITY_RE.finditer(text))
    if not matches:
        return {}
    body = matches[-1].group("body")
    return {m.group("key"): int(m.group("value")) for m in KV_RE.finditer(body)}


def build_report_stats(instance_id: str, reports_dir: Path) -> dict[str, int] | None:
    report_path = reports_dir / f"{instance_id}_enre_report.json"
    if not report_path.exists():
        return None

    entity_num = extract_entity_num(report_path)
    class_num = int(entity_num.get("Class", 0))
    function_num = int(entity_num.get("Method", 0))
    interface_num = int(entity_num.get("Interface", 0))

    return {
        "class": class_num,
        "function": function_num,
        "interface": interface_num,
        "total": class_num + function_num + interface_num,
    }


def iter_input_files(
    input_jsonl: str | None,
    input_dir: str | None,
    input_pattern: str,
) -> list[Path]:
    files: list[Path] = []

    if input_jsonl:
        files.append(Path(input_jsonl))

    if input_dir:
        base = Path(input_dir)
        files.extend(sorted(base.glob(input_pattern)))

    dedup: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            dedup.append(rp)

    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(description="统计有效 related code 的实体数量")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="related code JSONL 文件路径，例如 related_code_beam_map_same_as_core.jsonl",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="related code JSONL 所在目录，配合 --input-pattern 批量处理",
    )
    parser.add_argument(
        "--input-pattern",
        default="related_code_*_map_same_as_core.jsonl",
        help="在 --input-dir 下匹配输入文件的 glob 模式",
    )
    parser.add_argument(
        "--reports-dir",
        default="/root/autodl-tmp/view1/enre_py_reports/java_reports",
        help="ENRE 报告目录（默认 Java 报告目录）",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="输出 JSONL 文件路径",
    )
    args = parser.parse_args()

    input_files = iter_input_files(args.input_jsonl, args.input_dir, args.input_pattern)
    if not input_files:
        raise SystemExit("没有可处理的输入文件，请提供 --input-jsonl 或 --input-dir")

    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output_jsonl)

    valid_instance_ids: set[str] = set()

    for input_path in input_files:
        with input_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                instance_id = record.get("instance_id")
                if not isinstance(instance_id, str) or not instance_id:
                    continue

                if has_valid_related_code(record):
                    valid_instance_ids.add(instance_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_reports = 0
    with output_path.open("w", encoding="utf-8") as out:
        for instance_id in sorted(valid_instance_ids):
            stats = build_report_stats(instance_id, reports_dir)
            if stats is None:
                missing_reports += 1
                continue
            row = {"instance_id": instance_id, **stats}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"valid_instances={len(valid_instance_ids)}")
    print(f"written={written}")
    print(f"missing_reports={missing_reports}")
    print(f"input_files={len(input_files)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
