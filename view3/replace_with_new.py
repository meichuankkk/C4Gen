import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# 配置
MODEL_PATH = "jinaai/jina-embeddings-v2-base-code"
MAX_SEQ_LENGTH = 4096  # 强烈建议裁剪，避免 token 爆炸

# 处理单条 diff 的函数（现在接受模型参数）
def process_single_diff(diff, model):
    try:
        # 对单条 diff 进行编码
        print("[INFO] Encoding diff...")
        embedding = model.encode([diff], normalize_embeddings=True)  # 内置 L2 normalize

        # 检查嵌入是否包含 NaN 或 Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("[ERROR] NaN or Inf detected in the embedding.")
            return None

        print(f"[INFO] Embedding shape: {embedding.shape}")
        return embedding

    except Exception as e:
        print(f"[ERROR] Error during encoding: {e}")
        return None

# 加载 Jina 索引文件
jina_index_path = './resource/jina_diff_index.pkl'  # 替换为你的文件路径
output_file_path = './resource/updated_jina_diff_index.pkl'  # 输出文件路径

# 一次性加载模型
print(f"[INFO] Loading local embedding model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.max_seq_length = MAX_SEQ_LENGTH

with open(jina_index_path, 'rb') as f:
    jina_data = pickle.load(f)

# 假设 jina_data 中包含 "embeddings" 键，保存了所有的嵌入向量
embeddings = jina_data.get("embeddings", [])
raw_items = jina_data.get("raw_items", [])  # 假设 raw_items 保存了与 embeddings 对应的原始条目（如 diff, sha 等）

# 遍历所有的嵌入，检查 NaN 或 Inf 向量
for idx, emb in enumerate(embeddings):
    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
        print(f"[INFO] Re-generating embedding for item at index {idx}...")

        # 使用 process_single_diff 重新生成嵌入，传入预加载的模型
        new_embedding = process_single_diff(raw_items[idx], model)  # 根据原始条目生成新的嵌入

        if new_embedding is not None:
            # 替换原始嵌入
            embeddings[idx] = new_embedding[0]  # 新的嵌入是二维数组，取第一个元素

# 将更新后的数据保存回原文件
with open(output_file_path, 'wb') as f:
    pickle.dump(jina_data, f)

# 输出结果
print(f"[INFO] Updated embeddings for items with NaN or Inf.")
