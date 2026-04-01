import argparse
import json
import os
import random


def load_project_groups(projects_json_path):
    with open(projects_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        # 兼容旧格式：直接给出 cpp 项目列表。
        return {
            "cpp": {str(name).strip() for name in data if str(name).strip()},
            "java": set(),
            "python": set(),
        }

    if not isinstance(data, dict):
        raise ValueError("projects json 格式不正确，需为对象或列表")

    key_mapping = {
        "java": "java_projects",
        "python": "python_projects",
        "cpp": "cpp_projects",
    }

    groups = {}
    for lang, key in key_mapping.items():
        value = data.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"{key} 必须是列表")
        groups[lang] = {str(name).strip() for name in value if str(name).strip()}

    return groups


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


def has_non_empty_diff(item):
    diff = item.get("diff")
    return isinstance(diff, str) and diff.strip() != ""


def sample_subsets(input_jsonl, project_groups, sample_size, seed, extensions_by_lang):
    random.seed(seed)
    sampled = {"java": [], "python": [], "cpp": []}
    eligible_count = {"java": 0, "python": 0, "cpp": 0}
    scanned_count = 0
    skipped_empty_diff = 0

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

            if not has_non_empty_diff(item):
                skipped_empty_diff += 1
                continue

            repo = item.get("repo")
            if not isinstance(repo, str):
                continue

            files = item.get("files")
            for lang in ("java", "python", "cpp"):
                if repo not in project_groups[lang]:
                    continue

                if not files_all_cpp(files, extensions_by_lang[lang]):
                    continue

                eligible_count[lang] += 1
                if len(sampled[lang]) < sample_size:
                    sampled[lang].append(item)
                else:
                    j = random.randint(1, eligible_count[lang])
                    if j <= sample_size:
                        sampled[lang][j - 1] = item

    for lang in sampled:
        random.shuffle(sampled[lang])

    return sampled, scanned_count, eligible_count, skipped_empty_diff


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
        default="/data/data_public/riverbag/IST/dataset/full.jsonl",
    )
    parser.add_argument(
        "--projects_json",
        default="/data/data_public/riverbag/C4Gen/view1/dataset/newsubset/three_language_projects.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/data/data_public/riverbag/C4Gen/view1/dataset/newsubset",
    )
    parser.add_argument(
        "--java_output_jsonl",
        default="",
    )
    parser.add_argument(
        "--python_output_jsonl",
        default="",
    )
    parser.add_argument(
        "--cpp_output_jsonl",
        default="",
    )
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpp_extensions",
        nargs="+",
        default=[".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx",".c", ".h"],
    )
    parser.add_argument(
        "--java_extensions",
        nargs="+",
        default=[".java"],
    )
    parser.add_argument(
        "--python_extensions",
        nargs="+",
        default=[".py"],
    )

    # 兼容旧参数名。
    parser.add_argument(
        "--cpp_projects_json",
        default="",
        help="兼容旧参数，等同于 --projects_json（仅支持 cpp 列表场景）",
    )
    parser.add_argument(
        "--output_jsonl",
        default="",
        help="兼容旧参数，等同于 --cpp_output_jsonl",
    )

    args = parser.parse_args()

    projects_json = args.projects_json
    if args.cpp_projects_json:
        projects_json = args.cpp_projects_json

    project_groups = load_project_groups(projects_json)

    output_dir = args.output_dir
    java_output = args.java_output_jsonl or os.path.join(output_dir, "java_subset.jsonl")
    python_output = args.python_output_jsonl or os.path.join(output_dir, "python_subset.jsonl")
    cpp_output = args.cpp_output_jsonl or args.output_jsonl or os.path.join(output_dir, "cpp_subset.jsonl")

    sampled, scanned_count, eligible_count, skipped_empty_diff = sample_subsets(
        input_jsonl=args.input_jsonl,
        project_groups=project_groups,
        sample_size=args.sample_size,
        seed=args.seed,
        extensions_by_lang={
            "java": tuple(ext.lower() for ext in args.java_extensions),
            "python": tuple(ext.lower() for ext in args.python_extensions),
            "cpp": tuple(ext.lower() for ext in args.cpp_extensions),
        },
    )

    write_jsonl(sampled["java"], java_output)
    write_jsonl(sampled["python"], python_output)
    write_jsonl(sampled["cpp"], cpp_output)

    print(f"已扫描数据条数: {scanned_count}")
    print(f"跳过空 diff 条数: {skipped_empty_diff}")
    print(f"java 符合条件条数: {eligible_count['java']}")
    print(f"python 符合条件条数: {eligible_count['python']}")
    print(f"cpp 符合条件条数: {eligible_count['cpp']}")
    print(f"java 输出条数: {len(sampled['java'])}")
    print(f"python 输出条数: {len(sampled['python'])}")
    print(f"cpp 输出条数: {len(sampled['cpp'])}")
    print(f"java 输出文件: {java_output}")
    print(f"python 输出文件: {python_output}")
    print(f"cpp 输出文件: {cpp_output}")


if __name__ == "__main__":
    main()
