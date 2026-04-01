#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def load_shas(path: Path, field_path: tuple[str, ...]) -> list[str]:
    shas: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {idx} is not valid JSON: {exc}") from exc

            cur = obj
            for key in field_path:
                if not isinstance(cur, dict) or key not in cur:
                    raise ValueError(
                        f"{path}: line {idx} missing field path {'.'.join(field_path)}"
                    )
                cur = cur[key]

            if not isinstance(cur, str) or not cur.strip():
                raise ValueError(
                    f"{path}: line {idx} field {'.'.join(field_path)} is empty or not string"
                )
            shas.append(cur.strip())
    return shas


def find_duplicates(values: list[str]) -> list[tuple[str, int]]:
    c = Counter(values)
    return sorted((k, v) for k, v in c.items() if v > 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare subset_entry.commit_sha and query-sha consistency across two JSONL files."
    )
    parser.add_argument(
        "--core-file",
        default="C4Gen/view1/dataset/dpsk_chat_core_entities_5000/core_entities_java_nonempty_enriched.jsonl",
        help="JSONL file containing subset_entry.commit_sha",
    )
    parser.add_argument(
        "--result-file",
        default="C4Gen/view1/dataset/similar_diff_message/results_Java_BM25_dense_5_5_Jina.jsonl",
        help="JSONL file containing query-sha",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="How many sample missing SHAs to print from each side",
    )
    args = parser.parse_args()

    core_file = Path(args.core_file)
    result_file = Path(args.result_file)

    core_shas = load_shas(core_file, ("subset_entry", "commit_sha"))
    result_shas = load_shas(result_file, ("query-sha",))

    core_set = set(core_shas)
    result_set = set(result_shas)

    core_only = sorted(core_set - result_set)
    result_only = sorted(result_set - core_set)

    core_dups = find_duplicates(core_shas)
    result_dups = find_duplicates(result_shas)

    print("=== SHA Consistency Report ===")
    print(f"core file:   {core_file}")
    print(f"result file: {result_file}")
    print(f"core rows:   {len(core_shas)}")
    print(f"result rows: {len(result_shas)}")
    print(f"core unique SHAs:   {len(core_set)}")
    print(f"result unique SHAs: {len(result_set)}")
    print(f"core-only SHAs:   {len(core_only)}")
    print(f"result-only SHAs: {len(result_only)}")
    print(f"core duplicate SHA entries:   {sum(v - 1 for _, v in core_dups)}")
    print(f"result duplicate SHA entries: {sum(v - 1 for _, v in result_dups)}")

    if core_only:
        n = min(args.show_samples, len(core_only))
        print(f"\nSample core-only SHAs ({n}/{len(core_only)}):")
        for sha in core_only[:n]:
            print(sha)

    if result_only:
        n = min(args.show_samples, len(result_only))
        print(f"\nSample result-only SHAs ({n}/{len(result_only)}):")
        for sha in result_only[:n]:
            print(sha)

    matched = (
        len(core_shas) == len(result_shas)
        and len(core_only) == 0
        and len(result_only) == 0
    )
    print("\nFINAL:", "MATCHED (5000-to-5000, no missing SHA)" if matched else "NOT MATCHED")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
