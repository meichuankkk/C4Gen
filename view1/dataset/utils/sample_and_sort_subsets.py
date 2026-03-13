import argparse
import json
import os
import random


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_and_sort(items, sample_size, seed):
    if sample_size > len(items):
        raise ValueError(f"样本数不足，要求 {sample_size} 条，实际仅 {len(items)} 条")
    rng = random.Random(seed)
    sampled = rng.sample(items, sample_size)
    sampled.sort(key=lambda x: str(x.get("git_url", "")))
    return sampled


def process_file(input_path, output_path, sample_size, seed):
    items = read_jsonl(input_path)
    sampled_sorted = sample_and_sort(items, sample_size, seed)
    write_jsonl(output_path, sampled_sorted)
    return len(items), len(sampled_sorted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--java_input",
        default="/data/data_public/riverbag/C4Gen/dataset/subset/java_subset.jsonl",
    )
    parser.add_argument(
        "--python_input",
        default="/data/data_public/riverbag/C4Gen/dataset/subset/python_subset.jsonl",
    )
    parser.add_argument(
        "--java_output",
        default="/data/data_public/riverbag/C4Gen/dataset/subset/java_subset_5000_sorted.jsonl",
    )
    parser.add_argument(
        "--python_output",
        default="/data/data_public/riverbag/C4Gen/dataset/subset/python_subset_5000_sorted.jsonl",
    )
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    java_total, java_out = process_file(
        input_path=args.java_input,
        output_path=args.java_output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    python_total, python_out = process_file(
        input_path=args.python_input,
        output_path=args.python_output,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(f"java 输入条数: {java_total}，输出条数: {java_out}，输出文件: {args.java_output}")
    print(f"python 输入条数: {python_total}，输出条数: {python_out}，输出文件: {args.python_output}")


if __name__ == "__main__":
    main()
