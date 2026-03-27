#!/usr/bin/env python3
"""
整合流程：从 JSONL 读取数据 -> API 获取 core entities -> 处理 ENRE 报告 -> 聚类 -> 保存结果
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 导入必要的函数
from test_gpt import DeepSeekAPIClient, APITimeoutError, APIError, APIConnectionError
from parse_enre_report import load_report, filter_variables, enrich_entities, save_report
from entity_clustering_improved import embedding_sentence_bert, embedding_tfidf_fallback

# 配置
JSONL_FILE = "my_dataset/ApacheCM/python_subset.jsonl"
REPOS_BASE_DIR = Path("./repos/")
KEEP_CATEGORIES = {"Class", "Function"}


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


def get_core_entities_from_api(diff: str, api_client: DeepSeekAPIClient) -> List[str]:
    """
    调用 API 获取 core entities 列表
    
    返回: core entity 的简单名称列表（不包含完整路径）
    """
    system_prompt = "You are a helpful assistant which is Very professional in the field of software engineering. Always respond in valid JSON format."
    
    user_prompt = """In the .diff file, 'core classes' and 'core functions' are the core carriers of this code change — all key logic, functions, and bug fixes are concentrated in these 'core classes' or 'core functions'. All modifications involved in a single commit usually start with 'core classes' and 'core functions' and spread outward, meaning their changes lead to other necessary modifications. They are the key to understanding the intent of this commit. Please identify the 'core classes' and 'core functions' in this commit based on the provided .diff content, and note that their quantity should not account for a too large proportion of the .diff file.
Output format: A JSON object with a single key 'core entities' containing the list of concise entity name.Try to ensure that there is at least one entity in the diff.
Example output: {"core entities": ["class1", "class2", "function1", "function2", ...]}
provided .diff content:\n""" + diff
    
    try:
        # 检查 diff 大小
        max_diff_length = 40000
        if len(diff) > max_diff_length:
            print(f"⚠️  diff 内容过长 ({len(diff)} 字符)，将截断到 {max_diff_length} 字符")
            diff = diff[:max_diff_length] + "\n... (内容已截断)"
            user_prompt = user_prompt.split("\nprovided .diff content:\n")[0] + "\nprovided .diff content:\n" + diff
        
        print(f"📡 调用 API 获取 core entities...")
        response = api_client.call_api(user_prompt, system_prompt, model="deepseek-chat")
        
        core_entities = response.get("core entities", [])
        if not isinstance(core_entities, list):
            print(f"⚠️  API 返回格式异常，尝试转换...")
            core_entities = [core_entities] if core_entities else []
        
        print(f"✓ 获取到 {len(core_entities)} 个 core entities: {core_entities}")
        return core_entities
        
    except (APITimeoutError, APIError, APIConnectionError) as e:
        print(f"❌ API 调用失败: {e}")
        return []
    except Exception as e:
        print(f"❌ 获取 core entities 时出错: {e}")
        return []


def match_entity_name(simple_name: str, qualified_name: str) -> bool:
    """
    匹配简单名称和完整限定名
    
    API 返回的是简单名称（如 "DataflowTemplateOperator"）
    qualified_name 是完整路径（如 "airflow.contrib.operators.dataflow_operator.DataflowTemplateOperator"）
    """
    # 提取 qualified_name 的最后一部分（类名或函数名）
    if '.' in qualified_name:
        last_part = qualified_name.split('.')[-1]
    else:
        last_part = qualified_name
    
    # 直接匹配或忽略大小写匹配
    return simple_name == last_part or simple_name.lower() == last_part.lower()


def process_enre_report(repo_base_dir: Path, repo: str, commit_sha: str) -> Optional[Path]:
    """
    处理 ENRE 报告：加载 -> 过滤 -> enrich -> 保存
    
    Args:
        repo_base_dir: 仓库的基础目录（如 ./repos/airflow-commit_sha/）
        repo: 仓库名称
        commit_sha: commit SHA
    
    返回: filtered.json 的路径，如果失败返回 None
    """
    report_file = repo_base_dir / f"{repo}-{commit_sha}-report-enre.json"
    filtered_file = repo_base_dir / "filtered.json"
    
    if not report_file.exists():
        print(f"⚠️  ENRE 报告文件不存在: {report_file}")
        return None
    
    print(f"📄 处理 ENRE 报告: {report_file.name}")
    
    try:
        # 加载报告
        report = load_report(report_file)
        
        # 过滤实体
        filtered_variables = filter_variables(report.get("variables", []))
        print(f"   过滤后: {len(filtered_variables)} 个实体")
        
        # Enrich 实体（添加代码）
        # File 字段已经包含 {repo}-{commit_sha} 前缀，所以使用 REPOS_BASE_DIR
        enriched_entities = enrich_entities(filtered_variables, REPOS_BASE_DIR)
        print(f"   Enrich 后: {len(enriched_entities)} 个实体")
        
        # 保存
        filtered_report = {k: v for k, v in report.items() if k not in ("cells", "variables")}
        filtered_report["entities"] = enriched_entities
        save_report(filtered_file, filtered_report)
        
        print(f"✓ 已保存到: {filtered_file}")
        return filtered_file
        
    except Exception as e:
        print(f"❌ 处理 ENRE 报告时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def cluster_entities(entities: List[Dict], entities_per_cluster: int = 5) -> np.ndarray:
    """
    对实体进行聚类，每个 cluster 包含指定数量的 entity
    """
    print(f"🔄 开始聚类...")
    
    # 生成 embedding
    try:
        vectors = embedding_sentence_bert(entities)
    except Exception as e:
        print(f"⚠️  Sentence-BERT 不可用，使用 TF-IDF: {e}")
        vectors = embedding_tfidf_fallback(entities)
    
    # 计算聚类数
    n_clusters = max(1, len(entities) // entities_per_cluster)
    print(f"   计算聚类数: {len(entities)} 个实体 ÷ {entities_per_cluster} = {n_clusters} 个聚类")
    
    # K-means 聚类
    print(f"   使用 K-means 聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    
    # 评估
    try:
        silhouette = silhouette_score(vectors, labels)
        print(f"✓ 聚类完成: Silhouette Score = {silhouette:.4f}")
    except:
        print(f"✓ 聚类完成")
    
    return labels


def find_clusters_with_core_entities(
    entities: List[Dict],
    labels: np.ndarray,
    core_entities: List[str]
) -> Dict[int, List[Dict]]:
    """
    找到包含 core entities 的 cluster
    
    返回: {cluster_id: [entity1, entity2, ...]} 的字典
    """
    # 建立 entity index -> cluster_id 的映射
    entity_to_cluster = {}
    for i, label in enumerate(labels):
        if label not in entity_to_cluster:
            entity_to_cluster[label] = []
        entity_to_cluster[label].append(i)
    
    # 找到包含 core entities 的 cluster
    core_clusters = {}
    
    for core_entity_name in core_entities:
        # 在 entities 中查找匹配的实体
        for i, entity in enumerate(entities):
            qualified_name = entity.get('qualifiedName', '')
            if match_entity_name(core_entity_name, qualified_name):
                cluster_id = labels[i]
                if cluster_id not in core_clusters:
                    # 获取该 cluster 的所有实体
                    cluster_entity_indices = entity_to_cluster[cluster_id]
                    core_clusters[cluster_id] = [entities[idx] for idx in cluster_entity_indices]
                print(f"   ✓ 找到 core entity '{core_entity_name}' 在 cluster {cluster_id}")
                break
        else:
            print(f"   ⚠️  未找到 core entity: {core_entity_name}")
    
    return core_clusters


def save_cluster_results(
    core_clusters: Dict[int, List[Dict]],
    output_file: Path,
    repo: str,
    commit_sha: str,
    core_entities: List[str]
):
    """保存包含 core entities 的 cluster 信息"""
    result = {
        "repo": repo,
        "commit_sha": commit_sha,
        "core_entities": core_entities,  # API 返回的 core entities 列表
        "n_clusters": len(core_clusters),
        "clusters": []
    }
    
    for cluster_id, cluster_entity_list in core_clusters.items():
        cluster_data = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_entity_list),
            "entities": [
                {
                    "id": entity.get("id"),
                    "qualifiedName": entity.get("qualifiedName"),
                    "category": entity.get("category"),
                    "File": entity.get("File"),
                    "location": entity.get("location", {}),
                    "code": entity.get("code", "")  # 包含完整代码
                }
                for entity in cluster_entity_list
            ]
        }
        result["clusters"].append(cluster_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存到: {output_file}")


def process_single_record(record: Dict, api_client: DeepSeekAPIClient, output_dir: Path):
    """处理单条记录"""
    repo = record.get("repo", "")
    commit_sha = record.get("commit_sha", "")
    diff = record.get("diff", "")
    
    if not repo or not commit_sha:
        print(f"⚠️  跳过记录（缺少 repo 或 commit_sha）")
        return
    
    print(f"\n{'='*80}")
    print(f"处理: {repo} @ {commit_sha[:8]}")
    print(f"{'='*80}")
    
    # 1. 获取 core entities
    if not diff:
        print(f"⚠️  跳过（diff 为空）")
        return
    
    core_entities = get_core_entities_from_api(diff, api_client)
    if not core_entities:
        print(f"⚠️  未获取到 core entities，跳过")
        return
    
    # 2. 设置仓库目录（REPOS_BASE_DIR 是固定的 ./repos/）
    repo_base_dir = REPOS_BASE_DIR / f"{repo}-{commit_sha}"
    if not repo_base_dir.exists():
        print(f"⚠️  目录不存在: {repo_base_dir}")
        return
    
    # 3. 处理 ENRE 报告
    filtered_file = process_enre_report(repo_base_dir, repo, commit_sha)
    if not filtered_file:
        print(f"⚠️  处理 ENRE 报告失败，跳过")
        return
    
    # 4. 加载 filtered.json 并聚类
    with open(filtered_file, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    
    entities = filtered_data.get("entities", [])
    if not entities:
        print(f"⚠️  没有实体，跳过")
        return
    
    labels = cluster_entities(entities, entities_per_cluster=5)
    
    # 5. 找到包含 core entities 的 cluster
    core_clusters = find_clusters_with_core_entities(entities, labels, core_entities)
    
    if not core_clusters:
        print(f"⚠️  未找到包含 core entities 的 cluster")
        return
    
    print(f"\n📊 找到 {len(core_clusters)} 个包含 core entities 的 cluster")
    for cluster_id, cluster_entity_list in core_clusters.items():
        print(f"   Cluster {cluster_id}: {len(cluster_entity_list)} 个实体")
    
    # 6. 保存结果
    output_file = output_dir / f"{repo}-{commit_sha}-clusters.json"
    save_cluster_results(core_clusters, output_file, repo, commit_sha, core_entities)


def main():
    if len(sys.argv) < 2:
        print("用法: python integrated_pipeline.py <number>")
        print("   number: 要处理的记录数量")
        sys.exit(1)
    
    n_records = int(sys.argv[1])
    
    # 创建输出目录
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("整合流程：API -> ENRE -> 聚类 -> 保存")
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
    
    # 加载 JSONL 数据
    print(f"\n📂 加载数据: {JSONL_FILE}")
    records = load_jsonl_data(JSONL_FILE, n_records)
    print(f"✓ 共加载 {len(records)} 条记录")
    
    # 处理每条记录
    for i, record in enumerate(records, 1):
        print(f"\n\n[{{i}}/{len(records)}]")
        try:
            process_single_record(record, api_client, output_dir)
        except Exception as e:
            print(f"❌ 处理记录时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ 完成！共处理 {len(records)} 条记录")
    print(f"   结果保存在: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

