import argparse
import json
import os
import random
from typing import Dict, List, Set, Tuple


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


def is_language_file(path, extensions):
    lower_path = path.lower()
    return any(lower_path.endswith(ext) for ext in extensions)


def files_all_in_language(files, extensions):
    if not isinstance(files, list) or len(files) == 0:
        return False
    for path in files:
        if not isinstance(path, str) or not is_language_file(path, extensions):
            return False
    return True


def item_fingerprint(item):
    # 使用稳定序列化作为去重键，避免和已有5000条重复。
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


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

                if not files_all_in_language(files, extensions_by_lang[lang]):
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


def load_existing_fingerprints(existing_jsonl):
    fingerprints = set()
    loaded_count = 0

    if not existing_jsonl or not os.path.exists(existing_jsonl):
        return fingerprints, loaded_count

    with open(existing_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            fingerprints.add(item_fingerprint(item))
            loaded_count += 1

    return fingerprints, loaded_count


def sample_additional_subsets(
    input_jsonl,
    project_groups,
    sample_size,
    seed,
    extensions_by_lang,
    existing_fingerprints_by_lang,
):
    random.seed(seed)
    sampled = {"java": [], "python": [], "cpp": []}
    sampled_fingerprints = {"java": set(), "python": set(), "cpp": set()}
    eligible_count = {"java": 0, "python": 0, "cpp": 0}
    scanned_count = 0
    skipped_empty_diff = 0
    skipped_existing_overlap = {"java": 0, "python": 0, "cpp": 0}

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
            fp = item_fingerprint(item)
            for lang in ("java", "python", "cpp"):
                if repo not in project_groups[lang]:
                    continue

                if not files_all_in_language(files, extensions_by_lang[lang]):
                    continue

                if fp in existing_fingerprints_by_lang[lang]:
                    skipped_existing_overlap[lang] += 1
                    continue

                if fp in sampled_fingerprints[lang]:
                    continue

                eligible_count[lang] += 1
                if len(sampled[lang]) < sample_size:
                    sampled[lang].append(item)
                    sampled_fingerprints[lang].add(fp)
                else:
                    j = random.randint(1, eligible_count[lang])
                    if j <= sample_size:
                        removed = sampled[lang][j - 1]
                        removed_fp = item_fingerprint(removed)
                        sampled_fingerprints[lang].discard(removed_fp)
                        sampled[lang][j - 1] = item
                        sampled_fingerprints[lang].add(fp)

    for lang in sampled:
        random.shuffle(sampled[lang])

    return (
        sampled,
        scanned_count,
        eligible_count,
        skipped_empty_diff,
        skipped_existing_overlap,
    )


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
        "--existing_dir",
        default="/data/data_public/riverbag/C4Gen/view1/dataset/newSubDateset",
        help="已有5000条子集所在目录（用于去重）",
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
    parser.add_argument("--additional_sample_size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--additional_seed", type=int, default=43)
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
    parser.add_argument(
        "--java_existing_jsonl",
        default="",
    )
    parser.add_argument(
        "--python_existing_jsonl",
        default="",
    )
    parser.add_argument(
        "--cpp_existing_jsonl",
        default="",
    )
    parser.add_argument(
        "--java_additional_output_jsonl",
        default="",
    )
    parser.add_argument(
        "--python_additional_output_jsonl",
        default="",
    )
    parser.add_argument(
        "--cpp_additional_output_jsonl",
        default="",
    )
    parser.add_argument(
        "--skip_base_generation",
        action="store_true",
        help="仅生成新增数据集，不重新生成5000条基础子集",
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

    extensions_by_lang = {
        "java": tuple(ext.lower() for ext in args.java_extensions),
        "python": tuple(ext.lower() for ext in args.python_extensions),
        "cpp": tuple(ext.lower() for ext in args.cpp_extensions),
    }

    if not args.skip_base_generation:
        sampled, scanned_count, eligible_count, skipped_empty_diff = sample_subsets(
            input_jsonl=args.input_jsonl,
            project_groups=project_groups,
            sample_size=args.sample_size,
            seed=args.seed,
            extensions_by_lang=extensions_by_lang,
        )

        write_jsonl(sampled["java"], java_output)
        write_jsonl(sampled["python"], python_output)
        write_jsonl(sampled["cpp"], cpp_output)

        print("=== 基础子集(5000) ===")
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

    java_existing = args.java_existing_jsonl or os.path.join(args.existing_dir, "java_subset.jsonl")
    python_existing = args.python_existing_jsonl or os.path.join(args.existing_dir, "python_subset.jsonl")
    cpp_existing = args.cpp_existing_jsonl or os.path.join(args.existing_dir, "cpp_subset.jsonl")

    java_additional_output = args.java_additional_output_jsonl or os.path.join(output_dir, "java_subset_1500.jsonl")
    python_additional_output = args.python_additional_output_jsonl or os.path.join(output_dir, "python_subset_1500.jsonl")
    cpp_additional_output = args.cpp_additional_output_jsonl or os.path.join(output_dir, "cpp_subset_1500.jsonl")

    existing_fingerprints_by_lang = {}
    existing_loaded_count = {}
    existing_files = {
        "java": java_existing,
        "python": python_existing,
        "cpp": cpp_existing,
    }
    for lang in ("java", "python", "cpp"):
        fingerprints, loaded_count = load_existing_fingerprints(existing_files[lang])
        existing_fingerprints_by_lang[lang] = fingerprints
        existing_loaded_count[lang] = loaded_count

    (
        additional_sampled,
        additional_scanned_count,
        additional_eligible_count,
        additional_skipped_empty_diff,
        skipped_existing_overlap,
    ) = sample_additional_subsets(
        input_jsonl=args.input_jsonl,
        project_groups=project_groups,
        sample_size=args.additional_sample_size,
        seed=args.additional_seed,
        extensions_by_lang=extensions_by_lang,
        existing_fingerprints_by_lang=existing_fingerprints_by_lang,
    )

    write_jsonl(additional_sampled["java"], java_additional_output)
    write_jsonl(additional_sampled["python"], python_additional_output)
    write_jsonl(additional_sampled["cpp"], cpp_additional_output)

    print("=== 新增子集(1500, 与已有5000不重复) ===")
    print(f"已扫描数据条数: {additional_scanned_count}")
    print(f"跳过空 diff 条数: {additional_skipped_empty_diff}")
    print(f"java 已有去重条数: {existing_loaded_count['java']}")
    print(f"python 已有去重条数: {existing_loaded_count['python']}")
    print(f"cpp 已有去重条数: {existing_loaded_count['cpp']}")
    print(f"java 因与已有重复被跳过: {skipped_existing_overlap['java']}")
    print(f"python 因与已有重复被跳过: {skipped_existing_overlap['python']}")
    print(f"cpp 因与已有重复被跳过: {skipped_existing_overlap['cpp']}")
    print(f"java 符合新增条件条数: {additional_eligible_count['java']}")
    print(f"python 符合新增条件条数: {additional_eligible_count['python']}")
    print(f"cpp 符合新增条件条数: {additional_eligible_count['cpp']}")
    print(f"java 新增输出条数: {len(additional_sampled['java'])}")
    print(f"python 新增输出条数: {len(additional_sampled['python'])}")
    print(f"cpp 新增输出条数: {len(additional_sampled['cpp'])}")
    print(f"java 新增输出文件: {java_additional_output}")
    print(f"python 新增输出文件: {python_additional_output}")
    print(f"cpp 新增输出文件: {cpp_additional_output}")


if __name__ == "__main__":
    main()
