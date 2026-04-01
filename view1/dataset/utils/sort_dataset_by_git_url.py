import argparse
import json
import os


def read_jsonl(input_path):
    items = []
    invalid_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines += 1
    return items, invalid_lines


def write_jsonl(items, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="/data/data_public/riverbag/C4Gen/view1/dataset/newsubset_1500/python_subset_1500.jsonl",
    )
    parser.add_argument(
        "--output_jsonl",
        default="/data/data_public/riverbag/C4Gen/view1/dataset/newsubset_1500/python_subset_1500_sorted_by_git_url.jsonl",
    )
    args = parser.parse_args()

    items, invalid_lines = read_jsonl(args.input_jsonl)
    items.sort(key=lambda x: str(x.get("git_url", "")))
    write_jsonl(items, args.output_jsonl)

    print(f"输入条数: {len(items)}")
    print(f"无效行数: {invalid_lines}")
    print(f"输出文件: {args.output_jsonl}")


if __name__ == "__main__":
    main()
