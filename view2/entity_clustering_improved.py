#!/usr/bin/env python3
"""
改进的代码实体聚类 - 使用 Sentence-BERT
平衡速度和有效性的最佳方案
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def load_entities(json_file: str) -> List[Dict]:
    """加载实体数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['entities']


def embedding_sentence_bert(entities: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    使用 Sentence-BERT 生成 embedding
    
    优点：
    - 速度快（CPU 即可）
    - 效果好（比 TF-IDF 好很多）
    - 模型小，易于部署
    
    安装: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"📥 加载模型: {model_name}...")
        model = SentenceTransformer(model_name)
        
        # 准备文本：只使用 qualifiedName 和 File 路径
        texts = []
        for entity in entities:
            name = entity.get('qualifiedName', '')
            file_path = entity.get('File', '')
            category = entity.get('category', '')
            
            # 组合 name 和 file_path，不使用 code
            # file_path 可以提供上下文信息（模块、包结构等）
            if file_path:
                text = f"{category} {name} {file_path}"
            else:
                text = f"{category} {name}"
            
            texts.append(text)
        
        print(f"🔄 生成 embedding (共 {len(texts)} 个实体)...")
        # 批量编码，batch_size 可以根据内存调整
        vectors = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ 生成完成: {vectors.shape[0]} 个实体 × {vectors.shape[1]} 维向量")
        return vectors
        
    except ImportError:
        print("❌ 错误: 需要安装 sentence-transformers")
        print("   运行: pip install sentence-transformers")
        raise
    except Exception as e:
        print(f"❌ 生成 embedding 时出错: {e}")
        raise


def embedding_tfidf_fallback(entities: List[Dict]) -> np.ndarray:
    """
    TF-IDF 备用方案（如果 Sentence-BERT 不可用）
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("⚠️  使用 TF-IDF 备用方案...")
    
    texts = []
    for entity in entities:
        code = entity.get('code', '')
        name = entity.get('qualifiedName', '')
        text = f"{name} {code}"
        texts.append(text)
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    vectors = vectorizer.fit_transform(texts).toarray()
    print(f"✓ 生成完成: {vectors.shape[0]} 个实体 × {vectors.shape[1]} 维向量")
    return vectors


def find_optimal_clusters(vectors: np.ndarray, max_clusters: int = 50) -> int:
    """
    使用肘部法则和 silhouette 分数找到最佳聚类数
    """
    from sklearn.metrics import silhouette_score
    
    print(f"\n🔍 寻找最佳聚类数 (最多 {max_clusters} 个)...")
    
    max_clusters = min(max_clusters, len(vectors) // 10)  # 确保每个簇至少有10个样本
    min_clusters = max(2, len(vectors) // 1000)  # 至少2个簇
    
    best_k = min_clusters
    best_score = -1
    
    # 测试不同数量的聚类
    test_ks = list(range(min_clusters, min(max_clusters + 1, 20)))  # 限制测试范围以提高速度
    
    for k in test_ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        try:
            score = silhouette_score(vectors, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    
    print(f"✓ 最佳聚类数: {best_k} (Silhouette Score: {best_score:.4f})")
    return best_k


def cluster_entities(vectors: np.ndarray, n_clusters: int = None, method: str = "kmeans") -> np.ndarray:
    """
    对实体进行聚类
    
    Args:
        vectors: embedding 向量
        n_clusters: 聚类数量（None 则自动选择）
        method: 聚类方法 ('kmeans' 或 'dbscan')
    """
    if method == "kmeans":
        if n_clusters is None:
            n_clusters = find_optimal_clusters(vectors)
        
        print(f"\n🔄 使用 K-means 聚类 (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # 评估
        try:
            silhouette = silhouette_score(vectors, labels)
            print(f"✓ 聚类完成: Silhouette Score = {silhouette:.4f}")
        except:
            print("✓ 聚类完成")
        
        return labels
        
    elif method == "dbscan":
        print(f"\n🔄 使用 DBSCAN 聚类...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(vectors)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"✓ 聚类完成: {n_clusters} 个簇, {n_noise} 个噪声点")
        
        return labels
    else:
        raise ValueError(f"未知的聚类方法: {method}")


def analyze_clusters(entities: List[Dict], labels: np.ndarray) -> Dict[str, Any]:
    """分析聚类结果"""
    # 按聚类分组
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # 统计信息
    cluster_stats = []
    for cluster_id, entity_indices in sorted(clusters.items()):
        cluster_entities = [entities[i] for i in entity_indices]
        
        # 统计类别分布
        categories = {}
        for e in cluster_entities:
            cat = e.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        cluster_stats.append({
            'cluster_id': int(cluster_id),
            'size': len(entity_indices),
            'categories': categories
        })
    
    return {
        'clusters': clusters,
        'stats': cluster_stats,
        'n_clusters': len(clusters)
    }


def save_results(entities: List[Dict], labels: np.ndarray, output_file: str = "clustering_result.json"):
    """保存聚类结果"""
    result = {
        'entities': [
            {
                'id': entity['id'],
                'qualifiedName': entity['qualifiedName'],
                'category': entity['category'],
                'cluster': int(labels[i]),
                'File': entity.get('File', '')
            }
            for i, entity in enumerate(entities)
        ],
        'summary': {
            'total_entities': len(entities),
            'n_clusters': len(set(labels)),
            'clusters': {str(i): int(list(labels).count(i)) for i in set(labels)}
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {output_file}")


def print_sample_clusters(entities: List[Dict], labels: np.ndarray, n_samples: int = 5):
    """打印部分聚类结果示例"""
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    print(f"\n{'='*80}")
    print("📊 聚类结果示例 (前几个簇)")
    print(f"{'='*80}")
    
    for cluster_id in sorted(clusters.keys())[:n_samples]:
        entity_indices = clusters[cluster_id]
        print(f"\n簇 {cluster_id} (包含 {len(entity_indices)} 个实体):")
        print("-" * 80)
        
        # 显示前5个实体
        for idx in entity_indices[:5]:
            entity = entities[idx]
            print(f"  [{entity['id']}] {entity['category']}: {entity['qualifiedName']}")
            if entity.get('File'):
                print(f"      文件: {entity['File']}")


def main():
    import sys
    
    # 参数
    json_file = sys.argv[1] if len(sys.argv) > 1 else "filtered.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "clustering_result.json"
    use_tfidf = "--tfidf" in sys.argv  # 使用 --tfidf 强制使用 TF-IDF
    
    print("=" * 80)
    print("代码实体聚类 - 改进版")
    print("=" * 80)
    
    # 加载数据
    print(f"\n📂 加载数据: {json_file}")
    entities = load_entities(json_file)
    print(f"✓ 共 {len(entities)} 个实体")
    
    # 生成 embedding
    try:
        if use_tfidf:
            vectors = embedding_tfidf_fallback(entities)
        else:
            vectors = embedding_sentence_bert(entities)
    except Exception as e:
        print(f"\n⚠️  Sentence-BERT 不可用，使用 TF-IDF 备用方案")
        vectors = embedding_tfidf_fallback(entities)
    
    # 聚类：每个 cluster 包含 5 个 entity
    entities_per_cluster = 5
    n_clusters = max(1, len(entities) // entities_per_cluster)  # 至少1个聚类
    print(f"\n📊 计算聚类数: {len(entities)} 个实体 ÷ {entities_per_cluster} = {n_clusters} 个聚类")
    labels = cluster_entities(vectors, n_clusters=n_clusters, method="kmeans")
    
    # 分析
    analysis = analyze_clusters(entities, labels)
    print(f"\n📈 聚类统计:")
    print(f"  总实体数: {len(entities)}")
    print(f"  聚类数: {analysis['n_clusters']}")
    print(f"  平均每簇: {len(entities) / analysis['n_clusters']:.1f} 个实体")
    
    # 打印示例
    print_sample_clusters(entities, labels)
    
    # 保存结果
    save_results(entities, labels, output_file)
    
    print(f"\n✅ 完成！")


if __name__ == '__main__':
    main()

