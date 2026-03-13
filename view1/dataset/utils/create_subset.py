import argparse
import json
import os
import random


def load_cpp_projects(cpp_projects_path):
    with open(cpp_projects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cpp_projects" in data:
        projects = data["cpp_projects"]
    elif isinstance(data, list):
        projects = data
    else:
        raise ValueError("cpp_projects.json 格式不正确，需为列表或包含 cpp_projects 字段")
    if not isinstance(projects, list):
        raise ValueError("cpp_projects 必须是列表")
    return {str(name).strip() for name in projects if str(name).strip()}


def is_cpp_file(path, extensions):
    lower_path = path.lower()
    return any(lower_path.endswith(ext) for ext in extensions)


def files_all_cpp(files, extensions):
    if not isinstance(files, list) or len(files) == 0:
        return False
    for path in files:
        if not isinstance(path, str) or not is_cpp_file(path, extensions):
            return False
    return True


def sample_cpp_subset(input_jsonl, cpp_projects, sample_size, seed, extensions):
    random.seed(seed)
    sampled = []
    eligible_count = 0
    scanned_count = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            scanned_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            repo = item.get("repo")
            if repo not in cpp_projects:
                continue

            files = item.get("files")
            if not files_all_cpp(files, extensions):
                continue

            eligible_count += 1
            if len(sampled) < sample_size:
                sampled.append(item)
            else:
                j = random.randint(1, eligible_count)
                if j <= sample_size:
                    sampled[j - 1] = item

    random.shuffle(sampled)
    return sampled, scanned_count, eligible_count


def write_jsonl(items, output_jsonl):
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="/data/data_public/riverbag/C4Gen/dataset/full.jsonl",
    )
    parser.add_argument(
        "--cpp_projects_json",
        default="/data/data_public/riverbag/C4Gen/dataset/cpp_projects.json",
    )
    parser.add_argument(
        "--output_jsonl",
        default="/data/data_public/riverbag/C4Gen/dataset/cpp_subset_5000.jsonl",
    )
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"],
    )
    args = parser.parse_args()

    cpp_projects = load_cpp_projects(args.cpp_projects_json)
    sampled, scanned_count, eligible_count = sample_cpp_subset(
        input_jsonl=args.input_jsonl,
        cpp_projects=cpp_projects,
        sample_size=args.sample_size,
        seed=args.seed,
        extensions=tuple(ext.lower() for ext in args.extensions),
    )
    write_jsonl(sampled, args.output_jsonl)

    print(f"已扫描数据条数: {scanned_count}")
    print(f"符合条件条数: {eligible_count}")
    print(f"输出条数: {len(sampled)}")
    print(f"输出文件: {args.output_jsonl}")


if __name__ == "__main__":
    main()
