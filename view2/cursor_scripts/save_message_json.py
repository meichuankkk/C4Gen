#!/usr/bin/env python3
"""
遍历 python_test.jsonl 的每条 entry，根据 repo 与 commit_sha 前8位在
downloaded_repositories 下查找对应目录；若存在，将该条 entry 保存为该目录下的 message.json（与 zip 同级）。
"""

import json
import re
from pathlib import Path


def sanitize_dir_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip(' .')
    if len(name) > 200:
        name = name[:200]
    return name


def pair_dir_name(repo: str, commit_sha: str, sha_length: int = 8) -> str:
    repo_clean = sanitize_dir_name(repo)
    sha_prefix = commit_sha[:sha_length] if commit_sha else ''
    return f"{repo_clean}_{sha_prefix}"


def main():
    # 若实际为 CMG_dataset，可改为 CMG_dataset
    base_dir = Path("/root/autodl-tmp/CMG_data/downloaded_repositories")
    if not base_dir.exists():
        base_dir = Path("/root/autodl-tmp/CMG_dataset/downloaded_repositories")
    jsonl_path = Path(__file__).resolve().parent.parent / "ApacheCM-mini" / "python_test.jsonl"

    if not jsonl_path.exists():
        print(f"错误: 找不到 {jsonl_path}")
        return
    if not base_dir.exists():
        print(f"错误: 找不到目录 {base_dir}")
        return

    found = 0
    missed = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            repo = entry.get("repo", "")
            commit_sha = entry.get("commit_sha", "")
            dir_name = pair_dir_name(repo, commit_sha)
            pair_dir = base_dir / dir_name
            if pair_dir.is_dir():
                out_file = pair_dir / "message.json"
                with open(out_file, "w", encoding="utf-8") as out:
                    json.dump(entry, out, ensure_ascii=False, indent=2)
                found += 1
                if found <= 3 or found % 200 == 0:
                    print(f"  [{i+1}] 已写入: {dir_name}/message.json")
            else:
                missed += 1

    print(f"完成: 找到并写入 {found} 个目录，未找到目录 {missed} 条")


if __name__ == "__main__":
    main()
