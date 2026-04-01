#!/usr/bin/env python3
"""
简单的实体聚类 Demo
使用 TF-IDF 向量化和 K-means 聚类算法
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

def load_entities(json_file):
    """加载实体数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['entities']

def extract_text_features(entities):
    """从实体中提取文本特征用于向量化"""
    texts = []
    for entity in entities:
        # 组合多个特征：qualifiedName + category + code
        text = f"{entity['qualifiedName']} {entity['category']} {entity.get('code', '')}"
        texts.append(text)
    return texts

def cluster_entities(entities, n_clusters=3):
    """对实体进行聚类"""
    # 提取文本特征
    texts = extract_text_features(entities)
    
    # 使用 TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=100,  # 限制特征数量
        stop_words='english',  # 移除英文停用词
        min_df=1,  # 最小文档频率
        max_df=0.95  # 最大文档频率
    )
    
    # 生成 embedding vectors
    vectors = vectorizer.fit_transform(texts)
    print(f"生成了 {vectors.shape[0]} 个实体的 {vectors.shape[1]} 维向量")
    
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)
    
    return cluster_labels, vectors, vectorizer

def print_cluster_results(entities, cluster_labels):
    """打印聚类结果"""
    # 按聚类分组
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    print("\n" + "="*60)
    print("聚类结果:")
    print("="*60)
    
    for cluster_id, entity_indices in sorted(clusters.items()):
        print(f"\n聚类 {cluster_id} (包含 {len(entity_indices)} 个实体):")
        print("-" * 60)
        for idx in entity_indices:
            entity = entities[idx]
            print(f"  [{entity['id']}] {entity['category']}: {entity['qualifiedName']}")
            print(f"      文件: {entity['File']}")

def main():
    # 加载数据
    print("正在加载 filtered.json...")
    entities = load_entities('filtered.json')
    print(f"共加载 {len(entities)} 个实体")
    
    # 确定聚类数量（可以根据类别数量或使用其他方法）
    # 这里简单使用类别数量
    categories = set(e['category'] for e in entities)
    n_clusters = min(len(categories), len(entities))
    print(f"使用 {n_clusters} 个聚类")
    
    # 进行聚类
    print("\n正在进行聚类...")
    cluster_labels, vectors, vectorizer = cluster_entities(entities, n_clusters=n_clusters)
    
    # 打印结果
    print_cluster_results(entities, cluster_labels)
    
    # 保存结果到文件
    result = {
        'entities': [
            {
                'id': entity['id'],
                'qualifiedName': entity['qualifiedName'],
                'category': entity['category'],
                'cluster': int(cluster_labels[i]),
                'File': entity['File']
            }
            for i, entity in enumerate(entities)
        ]
    }
    
    with open('clustering_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n聚类结果已保存到 clustering_result.json")
    print(f"每个实体的 embedding vector 维度: {vectors.shape[1]}")

if __name__ == '__main__':
    main()

