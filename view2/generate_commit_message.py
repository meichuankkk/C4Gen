#!/usr/bin/env python3
"""
读取 JSONL 文件，调用 API 生成 commit message 并保存结果
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from test_gpt import DeepSeekAPIClient, APITimeoutError, APIError, APIConnectionError

# 配置
JSONL_FILE = "my_dataset/ApacheCM/python_subset.jsonl"
OUTPUT_DIR = Path("./output/commit_messages")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ORIGINAL_MESSAGES_FILE = OUTPUT_DIR / "original_messages.txt"
GENERATED_MESSAGES_FILE = OUTPUT_DIR / "generated_messages.txt"


def load_jsonl_data(file_path: str, n_records: int) -> List[Dict]:
    """从 JSONL 文件加载前 n 条数据"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  跳过第 {i+1} 行（JSON 解析失败）: {e}")
                continue
    return records


def generate_commit_message(diff: str, api_client: DeepSeekAPIClient) -> Dict[str, Any]:
    """
    调用 API 生成 commit message
    
    Args:
        diff: 代码差异内容
        api_client: API 客户端
    
    Returns:
        API 返回的 JSON 响应
    """
    # System prompt 参考 test_gpt.py
    system_prompt = "You are a helpful assistant which is Very professional in the field of software engineering. Always respond in valid JSON format."
    
    # User prompt 参考 experiment.py (596-598)
    user_prompt = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}
    
    provided .diff content:\n""" + diff
    
    try:
        # 检查 diff 大小
        max_diff_length = 40000
        if len(diff) > max_diff_length:
            print(f"⚠️  diff 内容过长 ({len(diff)} 字符)，将截断到 {max_diff_length} 字符")
            diff = diff[:max_diff_length] + "\n... (内容已截断)"
            user_prompt = user_prompt.split("\nprovided .diff content:\n")[0] + "\nprovided .diff content:\n" + diff
        
        print(f"📡 调用 API 生成 commit message...")
        response = api_client.call_api(user_prompt, system_prompt, model="deepseek-chat")
        
        return response
        
    except (APITimeoutError, APIError, APIConnectionError) as e:
        print(f"❌ API 调用失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 生成 commit message 时出错: {e}")
        return None


def save_result(record: Dict, commit_message_response: Dict[str, Any]):
    """保存结果到文本文件（原始消息和生成消息分别保存）"""
    original_message = record.get("message", "")
    generated_message = commit_message_response.get("commit_message", "") if commit_message_response else ""
    
    # 保存原始消息（追加模式）
    with open(ORIGINAL_MESSAGES_FILE, 'a', encoding='utf-8') as f:
        f.write(original_message + '\n')
    
    # 保存生成的消息（追加模式）
    with open(GENERATED_MESSAGES_FILE, 'a', encoding='utf-8') as f:
        f.write(generated_message + '\n')
    
    print(f"💾 结果已保存到: {ORIGINAL_MESSAGES_FILE} 和 {GENERATED_MESSAGES_FILE}")


def process_single_record(record: Dict, api_client: DeepSeekAPIClient, output_dir: Path, index: int):
    """处理单条记录"""
    repo = record.get("repo", "")
    commit_sha = record.get("commit_sha", "")
    diff = record.get("diff", "")
    
    if not repo or not commit_sha:
        print(f"⚠️  跳过记录（缺少 repo 或 commit_sha）")
        return
    
    print(f"\n{'='*80}")
    print(f"[{index}] 处理: {repo} @ {commit_sha[:8]}")
    print(f"{'='*80}")
    
    if not diff:
        print(f"⚠️  跳过（diff 为空）")
        return
    
    # 调用 API 生成 commit message
    commit_message_response = generate_commit_message(diff, api_client)
    
    if not commit_message_response:
        print(f"⚠️  未获取到 commit message，跳过")
        return
    
    # 显示结果
    generated_message = commit_message_response.get("commit_message", "")
    original_message = record.get("message", "")
    print(f"\n📝 原始 commit message:")
    print(f"   {original_message}")
    print(f"\n🤖 生成的 commit message:")
    print(f"   {generated_message}")
    
    # 保存结果
    save_result(record, commit_message_response)


def main():
    if len(sys.argv) < 2:
        print("用法: python generate_commit_message.py <number>")
        print("   number: 要处理的记录数量")
        sys.exit(1)
    
    n_records = int(sys.argv[1])
    
    print("=" * 80)
    print("生成 Commit Message - API 调用")
    print("=" * 80)
    
    # 加载 API key
    try:
        with open("deepseek_api.txt", "r") as f:
            api_key = f.read().strip()
        if not api_key:
            print("❌ API key 为空")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ 未找到 deepseek_api.txt 文件")
        sys.exit(1)
    
    # 初始化 API client
    api_client = DeepSeekAPIClient(api_key, timeout=600)
    
    # 清空输出文件（如果存在）
    if ORIGINAL_MESSAGES_FILE.exists():
        ORIGINAL_MESSAGES_FILE.unlink()
    if GENERATED_MESSAGES_FILE.exists():
        GENERATED_MESSAGES_FILE.unlink()
    
    # 加载 JSONL 数据
    print(f"\n📂 加载数据: {JSONL_FILE}")
    records = load_jsonl_data(JSONL_FILE, n_records)
    print(f"✓ 共加载 {len(records)} 条记录")
    
    # 处理每条记录
    success_count = 0
    for i, record in enumerate(records, 1):
        try:
            process_single_record(record, api_client, OUTPUT_DIR, i)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理记录时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ 完成！共处理 {len(records)} 条记录，成功 {success_count} 条")
    print(f"   原始消息保存在: {ORIGINAL_MESSAGES_FILE}")
    print(f"   生成消息保存在: {GENERATED_MESSAGES_FILE}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

