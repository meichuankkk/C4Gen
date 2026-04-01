#!/usr/bin/env python3
"""
读取 JSONL（diff）+ cluster 补充信息（含代码），调用 API 生成 commit message。

基础 user prompt 来自 generate_commit_message.py/experiment.py，
并追加核心实体所在 cluster 的补充信息（含 code）。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from test_gpt import DeepSeekAPIClient, APITimeoutError, APIError, APIConnectionError

# 配置
JSONL_FILE = "my_dataset/ApacheCM/python_subset.jsonl"
CLUSTER_DIR = Path("./output")  # integrated_pipeline 输出的 cluster 文件目录
OUTPUT_DIR = Path("./output/commit_messages_with_clusters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_MESSAGES_FILE = OUTPUT_DIR / "original_messages.txt"
GENERATED_MESSAGES_FILE = OUTPUT_DIR / "generated_messages_with_clusters.txt"


def load_jsonl_data(file_path: str, n_records: int) -> List[Dict]:
    """从 JSONL 文件加载前 n 条数据"""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  跳过第 {i+1} 行（JSON 解析失败）: {e}")
    return records


def load_cluster_context(repo: str, commit_sha: str) -> str:
    """
    读取 cluster 文件，生成补充上下文字符串（包含 code）。
    仅使用 integrated_pipeline 生成的精简 cluster 文件：{repo}-{commit_sha}-clusters.json
    """
    cluster_file = CLUSTER_DIR / f"{repo}-{commit_sha}-clusters.json"
    if not cluster_file.exists():
        print(f"⚠️  未找到 cluster 文件: {cluster_file}")
        return ""

    try:
        with open(cluster_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  读取 cluster 文件失败: {e}")
        return ""

    clusters = data.get("clusters", [])
    if not clusters:
        return ""

    lines = []
    lines.append("Additional context from core-entity clusters:")
    for cluster in clusters:
        cid = cluster.get("cluster_id")
        size = cluster.get("size")
        lines.append(f"- Cluster {cid} (size={size}):")
        for ent in cluster.get("entities", []):
            qn = ent.get("qualifiedName", "")
            cat = ent.get("category", "")
            file_path = ent.get("File", "")
            code = ent.get("code", "") or ""
            # 截断 code，避免 prompt 过长
            max_code_len = 800
            if len(code) > max_code_len:
                code_display = code[:max_code_len] + "\n... (truncated)"
            else:
                code_display = code
            lines.append(f"  * {cat} {qn} ({file_path})")
            if code_display.strip():
                lines.append("    code:")
                for line in code_display.splitlines():
                    lines.append(f"      {line}")
    return "\n".join(lines)


def build_user_prompt(diff: str, cluster_context: str) -> str:
    """
    基础 prompt（experiment.py 596-598） + diff + cluster 补充信息
    """
    base_prompt = (
        'You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit.\n'
        'Output format: A JSON object with a single key "commit_message" containing the concise commit message.\n'
        'Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}\n\n'
    )
    prompt = base_prompt
    if cluster_context:
        prompt += cluster_context + "\n\n"
    prompt += "provided .diff content:\n" + diff
    return prompt


def generate_commit_message_with_clusters(
    diff: str, cluster_context: str, api_client: DeepSeekAPIClient
) -> Dict[str, Any] | None:
    """调用 API 生成 commit message，带 cluster 补充信息"""
    system_prompt = (
        "You are a helpful assistant which is Very professional in the field of software engineering. "
        "Always respond in valid JSON format."
    )
    user_prompt = build_user_prompt(diff, cluster_context)

    try:
        # 截断过长 diff
        max_diff_length = 40000
        if len(diff) > max_diff_length:
            print(f"⚠️  diff 内容过长 ({len(diff)} 字符)，将截断到 {max_diff_length} 字符")
            diff = diff[:max_diff_length] + "\n... (内容已截断)"
            user_prompt = build_user_prompt(diff, cluster_context)

        print("📡 调用 API 生成 commit message...")
        response = api_client.call_api(user_prompt, system_prompt, model="deepseek-chat")
        return response

    except (APITimeoutError, APIError, APIConnectionError) as e:
        print(f"❌ API 调用失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 生成 commit message 时出错: {e}")
        return None


def save_result_text(record: Dict, commit_message_response: Dict[str, Any] | None):
    """追加保存原始/生成消息到文本文件"""
    original_message = record.get("message", "")
    generated_message = (
        commit_message_response.get("commit_message", "") if commit_message_response else ""
    )

    with open(ORIGINAL_MESSAGES_FILE, "a", encoding="utf-8") as f:
        f.write(original_message + "\n")

    with open(GENERATED_MESSAGES_FILE, "a", encoding="utf-8") as f:
        f.write(generated_message + "\n")


def process_single_record(record: Dict, api_client: DeepSeekAPIClient, index: int):
    """处理单条记录"""
    repo = record.get("repo", "")
    commit_sha = record.get("commit_sha", "")
    diff = record.get("diff", "")

    if not repo or not commit_sha:
        print("⚠️  跳过记录（缺少 repo 或 commit_sha）")
        return False
    if not diff:
        print("⚠️  跳过（diff 为空）")
        return False

    print(f"\n{'='*80}")
    print(f"[{index}] 处理: {repo} @ {commit_sha[:8]}")
    print(f"{'='*80}")

    # 读取 cluster 补充信息
    cluster_context = load_cluster_context(repo, commit_sha)

    # 调用 API
    response = generate_commit_message_with_clusters(diff, cluster_context, api_client)
    if not response:
        print("⚠️  未获取到 commit message，跳过")
        return False

    # 打印结果
    generated_message = response.get("commit_message", "")
    original_message = record.get("message", "")
    print("\n📝 原始 commit message:")
    print(f"   {original_message}")
    print("\n🤖 生成的 commit message:")
    print(f"   {generated_message}")

    # 保存结果（文本追加）
    save_result_text(record, response)
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python generate_commit_message_with_clusters.py <number>")
        sys.exit(1)

    n_records = int(sys.argv[1])

    print("=" * 80)
    print("生成 Commit Message（含核心实体 cluster 补充信息）")
    print("=" * 80)

    # 读取 API key
    try:
        with open("deepseek_api.txt", "r") as f:
            api_key = f.read().strip()
        if not api_key:
            print("❌ API key 为空")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ 未找到 deepseek_api.txt 文件")
        sys.exit(1)

    # 初始化 client
    api_client = DeepSeekAPIClient(api_key, timeout=600)

    # 清空输出文件
    if ORIGINAL_MESSAGES_FILE.exists():
        ORIGINAL_MESSAGES_FILE.unlink()
    if GENERATED_MESSAGES_FILE.exists():
        GENERATED_MESSAGES_FILE.unlink()

    # 加载数据
    print(f"\n📂 加载数据: {JSONL_FILE}")
    records = load_jsonl_data(JSONL_FILE, n_records)
    print(f"✓ 共加载 {len(records)} 条记录")

    # 处理
    success = 0
    for i, record in enumerate(records, 1):
        try:
            if process_single_record(record, api_client, i):
                success += 1
        except Exception as e:
            print(f"❌ 处理记录时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"✅ 完成！共处理 {len(records)} 条，成功 {success} 条")
    print(f"   原始消息保存在: {ORIGINAL_MESSAGES_FILE}")
    print(f"   生成消息保存在: {GENERATED_MESSAGES_FILE}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


