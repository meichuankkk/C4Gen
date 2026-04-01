#!/usr/bin/env python3
"""
代码实体聚类 - 多种 Embedding 方法对比
分析不同 embedding 方法的实现难度、速度和有效性
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 方法 1: TF-IDF (当前使用的方法)
# ============================================================================
def embedding_tfidf(entities: List[Dict], max_features: int = 500) -> Tuple[np.ndarray, float]:
    """
    使用 TF-IDF 向量化
    
    优点：
    - 实现简单，无需额外依赖
    - 速度快，内存占用小
    - 适合处理大量文本
    
    缺点：
    - 无法理解代码语义
    - 对代码结构不敏感
    - 效果一般
    
    实现难度：⭐ (非常简单)
    速度：⭐⭐⭐⭐⭐ (非常快)
    有效性：⭐⭐ (一般)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    start_time = time.time()
    
    texts = []
    for entity in entities:
        # 组合多个特征
        text = f"{entity['qualifiedName']} {entity['category']} {entity.get('code', '')}"
        texts.append(text)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2)  # 使用 1-gram 和 2-gram
    )
    
    vectors = vectorizer.fit_transform(texts).toarray()
    elapsed = time.time() - start_time
    
    return vectors, elapsed


# ============================================================================
# 方法 2: CodeBERT / UniXcoder (代码专用预训练模型)
# ============================================================================
def embedding_codebert(entities: List[Dict], model_name: str = "microsoft/codebert-base") -> Tuple[np.ndarray, float]:
    """
    使用 CodeBERT 或 UniXcoder 等代码专用模型
    
    优点：
    - 专门为代码设计，理解代码语义
    - 效果好，能捕获功能相似性
    - 支持多种编程语言
    
    缺点：
    - 需要 GPU 或较长时间
    - 模型较大（~500MB）
    - 实现稍复杂
    
    实现难度：⭐⭐⭐ (中等)
    速度：⭐⭐ (较慢，需要 GPU 加速)
    有效性：⭐⭐⭐⭐⭐ (非常好)
    
    安装: pip install transformers torch
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        start_time = time.time()
        
        # 加载模型和 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        # 提取代码文本
        texts = []
        for entity in entities:
            code = entity.get('code', '')
            name = entity.get('qualifiedName', '')
            # 组合代码和名称
            text = f"{name}\n{code}"[:512]  # 限制长度
            texts.append(text)
        
        # 批量编码
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                outputs = model(**inputs)
                # 使用 [CLS] token 的 embedding 或平均池化
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        vectors = np.vstack(embeddings)
        elapsed = time.time() - start_time
        
        return vectors, elapsed
        
    except ImportError:
        print("⚠️  CodeBERT 需要安装: pip install transformers torch")
        return None, 0


# ============================================================================
# 方法 3: Sentence-BERT (通用文本 embedding)
# ============================================================================
def embedding_sentence_bert(entities: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> Tuple[np.ndarray, float]:
    """
    使用 Sentence-BERT 模型
    
    优点：
    - 速度快（比 CodeBERT 快）
    - 效果好（比 TF-IDF 好很多）
    - 模型小，易于部署
    - 支持 CPU 推理
    
    缺点：
    - 不是专门为代码设计
    - 对代码结构理解有限
    
    实现难度：⭐⭐ (简单)
    速度：⭐⭐⭐⭐ (快)
    有效性：⭐⭐⭐⭐ (好)
    
    安装: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        start_time = time.time()
        
        # 加载模型（首次运行会下载模型）
        model = SentenceTransformer(model_name)
        
        # 准备文本
        texts = []
        for entity in entities:
            code = entity.get('code', '')
            name = entity.get('qualifiedName', '')
            text = f"{name}: {code[:200]}"  # 限制长度以提高速度
            texts.append(text)
        
        # 批量编码
        vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
        elapsed = time.time() - start_time
        
        return vectors, elapsed
        
    except ImportError:
        print("⚠️  Sentence-BERT 需要安装: pip install sentence-transformers")
        return None, 0


# ============================================================================
# 方法 4: 混合特征 (代码结构 + 文本)
# ============================================================================
def embedding_hybrid(entities: List[Dict]) -> Tuple[np.ndarray, float]:
    """
    混合特征：结合代码结构特征和文本特征
    
    优点：
    - 结合多种信息源
    - 实现灵活
    - 速度快
    
    缺点：
    - 需要手工设计特征
    - 效果取决于特征设计
    
    实现难度：⭐⭐⭐ (中等)
    速度：⭐⭐⭐⭐⭐ (非常快)
    有效性：⭐⭐⭐ (较好)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    
    start_time = time.time()
    
    # 1. 文本特征 (TF-IDF)
    texts = []
    for entity in entities:
        text = f"{entity['qualifiedName']} {entity.get('code', '')}"
        texts.append(text)
    
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    text_vectors = vectorizer.fit_transform(texts).toarray()
    
    # 2. 结构特征
    structural_features = []
    for entity in entities:
        code = entity.get('code', '')
        features = [
            len(code),  # 代码长度
            code.count('def '),  # 函数定义数
            code.count('class '),  # 类定义数
            code.count('('),  # 括号数（复杂度指标）
            code.count('if '),  # 条件语句数
            code.count('for '),  # 循环数
            len(entity.get('qualifiedName', '').split('.')),  # 命名空间深度
        ]
        structural_features.append(features)
    
    structural_vectors = np.array(structural_features)
    
    # 3. 类别特征 (one-hot)
    categories = list(set(e['category'] for e in entities))
    category_map = {cat: i for i, cat in enumerate(categories)}
    category_vectors = np.zeros((len(entities), len(categories)))
    for i, entity in enumerate(entities):
        cat_idx = category_map.get(entity['category'], 0)
        category_vectors[i, cat_idx] = 1
    
    # 4. 组合特征
    vectors = np.hstack([
        text_vectors,
        structural_vectors,
        category_vectors
    ])
    
    # 标准化
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)
    
    elapsed = time.time() - start_time
    
    return vectors, elapsed


# ============================================================================
# 方法 5: OpenAI Embeddings (如果可用)
# ============================================================================
def embedding_openai(entities: List[Dict], api_key: str = None) -> Tuple[np.ndarray, float]:
    """
    使用 OpenAI 的 text-embedding-ada-002 或 text-embedding-3-small
    
    优点：
    - 效果好
    - 无需本地模型
    
    缺点：
    - 需要 API 调用，有成本
    - 速度取决于网络
    - 需要 API key
    
    实现难度：⭐⭐ (简单)
    速度：⭐⭐ (慢，需要网络请求)
    有效性：⭐⭐⭐⭐ (好)
    """
    try:
        from openai import OpenAI
        
        if not api_key:
            print("⚠️  需要 OpenAI API key")
            return None, 0
        
        start_time = time.time()
        
        client = OpenAI(api_key=api_key)
        
        texts = []
        for entity in entities:
            code = entity.get('code', '')
            name = entity.get('qualifiedName', '')
            text = f"{name}: {code[:1000]}"  # 限制长度
            texts.append(text)
        
        # 批量调用 API
        embeddings = []
        batch_size = 100  # OpenAI 支持批量
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",  # 或 text-embedding-ada-002
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        vectors = np.array(embeddings)
        elapsed = time.time() - start_time
        
        return vectors, elapsed
        
    except ImportError:
        print("⚠️  需要安装: pip install openai")
        return None, 0


# ============================================================================
# 聚类和评估
# ============================================================================
def cluster_and_evaluate(vectors: np.ndarray, n_clusters: int = None) -> Dict[str, Any]:
    """对向量进行聚类并评估"""
    if n_clusters is None:
        # 使用肘部法则或 silhouette 分数选择最佳聚类数
        n_clusters = min(50, max(5, len(vectors) // 100))
    
    # K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    
    # 评估
    try:
        silhouette = silhouette_score(vectors, labels)
    except:
        silhouette = -1
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette
    }


# ============================================================================
# 主函数：对比不同方法
# ============================================================================
def compare_methods(json_file: str = "filtered.json", sample_size: int = 1000):
    """对比不同 embedding 方法"""
    
    print("=" * 80)
    print("代码实体聚类 - Embedding 方法对比")
    print("=" * 80)
    
    # 加载数据
    print(f"\n📂 加载数据: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities = data['entities']
    
    # 如果数据量大，采样测试
    if len(entities) > sample_size:
        print(f"⚠️  数据量较大 ({len(entities)} 个实体)，采样 {sample_size} 个进行测试")
        import random
        random.seed(42)
        entities = random.sample(entities, sample_size)
    
    print(f"✓ 共 {len(entities)} 个实体")
    
    methods = [
        ("TF-IDF", embedding_tfidf, {}),
        ("混合特征", embedding_hybrid, {}),
        ("Sentence-BERT", embedding_sentence_bert, {}),
        ("CodeBERT", embedding_codebert, {}),
    ]
    
    results = []
    
    for method_name, method_func, kwargs in methods:
        print(f"\n{'='*80}")
        print(f"🔍 测试方法: {method_name}")
        print(f"{'='*80}")
        
        try:
            vectors, elapsed = method_func(entities, **kwargs)
            
            if vectors is None:
                print(f"❌ {method_name}: 未实现或缺少依赖")
                continue
            
            print(f"✓ 生成 embedding: {vectors.shape[0]} 个实体 × {vectors.shape[1]} 维")
            print(f"⏱️  耗时: {elapsed:.2f} 秒")
            
            # 聚类
            print("🔄 进行聚类...")
            cluster_result = cluster_and_evaluate(vectors)
            
            print(f"✓ 聚类完成: {cluster_result['n_clusters']} 个簇")
            print(f"📊 Silhouette Score: {cluster_result['silhouette_score']:.4f}")
            
            results.append({
                'method': method_name,
                'time': elapsed,
                'dimension': vectors.shape[1],
                'silhouette': cluster_result['silhouette_score'],
                'n_clusters': cluster_result['n_clusters']
            })
            
        except Exception as e:
            print(f"❌ {method_name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n{'='*80}")
    print("📊 方法对比总结")
    print(f"{'='*80}")
    print(f"{'方法':<20} {'耗时(秒)':<12} {'维度':<10} {'Silhouette':<12} {'评分':<10}")
    print("-" * 80)
    
    for r in results:
        score = "⭐⭐⭐⭐⭐" if r['silhouette'] > 0.3 else \
                "⭐⭐⭐⭐" if r['silhouette'] > 0.2 else \
                "⭐⭐⭐" if r['silhouette'] > 0.1 else "⭐⭐"
        print(f"{r['method']:<20} {r['time']:<12.2f} {r['dimension']:<10} {r['silhouette']:<12.4f} {score}")
    
    return results


# ============================================================================
# 推荐方案
# ============================================================================
def get_recommendation():
    """根据场景推荐最佳方案"""
    print("\n" + "=" * 80)
    print("💡 推荐方案")
    print("=" * 80)
    print("""
根据你的需求（平衡速度和有效性），推荐以下方案：

1. 🥇 **首选：Sentence-BERT (all-MiniLM-L6-v2)**
   - 实现难度：⭐⭐ (简单)
   - 速度：⭐⭐⭐⭐ (快，CPU 即可)
   - 有效性：⭐⭐⭐⭐ (好)
   - 安装：pip install sentence-transformers
   - 适用：大多数场景的最佳平衡

2. 🥈 **备选：混合特征方法**
   - 实现难度：⭐⭐⭐ (中等)
   - 速度：⭐⭐⭐⭐⭐ (非常快)
   - 有效性：⭐⭐⭐ (较好)
   - 适用：数据量大、需要快速处理

3. 🥉 **高质量：CodeBERT**
   - 实现难度：⭐⭐⭐ (中等)
   - 速度：⭐⭐ (需要 GPU 或较长时间)
   - 有效性：⭐⭐⭐⭐⭐ (非常好)
   - 安装：pip install transformers torch
   - 适用：对质量要求高、有 GPU 资源

4. ⚡ **快速原型：TF-IDF**
   - 实现难度：⭐ (非常简单)
   - 速度：⭐⭐⭐⭐⭐ (非常快)
   - 有效性：⭐⭐ (一般)
   - 适用：快速验证、数据量极大
    """)


if __name__ == '__main__':
    import sys
    
    json_file = sys.argv[1] if len(sys.argv) > 1 else "filtered.json"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    compare_methods(json_file, sample_size)
    get_recommendation()

