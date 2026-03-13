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
        embedding = model.encode([diff], normalize_embeddings=True)  # 内置 L2 normalize

        # 检查嵌入是否包含 NaN 或 Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("[ERROR] NaN or Inf detected in the embedding.")
            return None

        return embedding

    except Exception as e:
        print(f"[ERROR] Error during encoding: {e}")
        return None


def compare_embeddings(emb1, emb2, idx, tolerance=1e-6):
    """
    比较两个嵌入向量是否相等

    Args:
        emb1: 原始嵌入向量
        emb2: 新生成的嵌入向量
        idx: 索引编号
        tolerance: 容差阈值
    """
    # 转换为numpy数组
    emb1_array = np.array(emb1)
    emb2_array = np.array(emb2[0]) if len(emb2.shape) == 2 else np.array(emb2)

    # 检查形状是否相同
    if emb1_array.shape != emb2_array.shape:
        print(f"[比较 {idx}] 形状不同: {emb1_array.shape} vs {emb2_array.shape}")
        return False

    # 计算差异
    diff = np.abs(emb1_array - emb2_array)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # 检查是否在容差范围内
    is_close = np.allclose(emb1_array, emb2_array, rtol=tolerance, atol=tolerance)

    if is_close:
        print(f"[比较 {idx}] ✓ 向量相等 (最大差异: {max_diff:.6e}, 平均差异: {mean_diff:.6e})")
    else:
        print(f"[比较 {idx}] ✗ 向量不相等 (最大差异: {max_diff:.6e}, 平均差异: {mean_diff:.6e})")

        # 找出差异最大的维度
        max_diff_idx = np.argmax(diff)
        max_diff_value = diff.flatten()[max_diff_idx]
        print(f"      最大差异位置: 索引 {max_diff_idx}, 值: {max_diff_value:.6e}")
        print(f"      原始值: {emb1_array.flatten()[max_diff_idx]:.6e}, 新值: {emb2_array.flatten()[max_diff_idx]:.6e}")

    return is_close, max_diff, mean_diff


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

# 限制只处理前1000条
limit = 1000
if len(embeddings) < limit:
    limit = len(embeddings)

print(f"[INFO] 只处理前 {limit} 条数据")

# 统计信息
equal_count = 0
not_equal_count = 0
nan_inf_fixed = 0
all_max_diffs = []
all_mean_diffs = []

# 遍历前1000条嵌入
for idx in range(limit):
    emb = embeddings[idx]

    # 检查原始嵌入是否包含 NaN 或 Inf
    has_nan_inf = np.any(np.isnan(emb)) or np.any(np.isinf(emb))

    if has_nan_inf:
        print(f"[处理 {idx}] 原始嵌入包含NaN或Inf，重新生成...")
        new_embedding = process_single_diff(raw_items[idx], model)

        if new_embedding is not None:
            # 比较新嵌入与原始嵌入（虽然原始有问题，但还是可以对比差异）
            is_equal, max_diff, mean_diff = compare_embeddings(emb, new_embedding, idx)

            # 替换原始嵌入
            embeddings[idx] = new_embedding[0]  # 新的嵌入是二维数组，取第一个元素
            nan_inf_fixed += 1

            if not is_equal:
                not_equal_count += 1
                all_max_diffs.append(max_diff)
                all_mean_diffs.append(mean_diff)
        else:
            print(f"[处理 {idx}] 重新生成嵌入失败")
    else:
        # 原始嵌入正常，重新生成并比较
        print(f"[处理 {idx}] 重新生成正常嵌入进行对比...")
        new_embedding = process_single_diff(raw_items[idx], model)

        if new_embedding is not None:
            # 比较新嵌入与原始嵌入
            is_equal, max_diff, mean_diff = compare_embeddings(emb, new_embedding, idx)

            if is_equal:
                equal_count += 1
            else:
                not_equal_count += 1
                all_max_diffs.append(max_diff)
                all_mean_diffs.append(mean_diff)
        else:
            print(f"[处理 {idx}] 重新生成嵌入失败")

# 输出统计结果
print("\n" + "=" * 50)
print("[统计结果]")
print(f"处理数据总数: {limit}")
print(f"修复的NaN/Inf嵌入数: {nan_inf_fixed}")
print(f"向量相等的数量: {equal_count}")
print(f"向量不相等的数量: {not_equal_count}")

if not_equal_count > 0:
    print(f"\n[差异统计]")
    print(f"最大差异范围: [{min(all_max_diffs):.6e}, {max(all_max_diffs):.6e}]")
    print(f"平均最大差异: {np.mean(all_max_diffs):.6e}")
    print(f"平均差异: {np.mean(all_mean_diffs):.6e}")

    # 检查是否有显著差异
    significant_threshold = 1e-4
    significant_diffs = [d for d in all_max_diffs if d > significant_threshold]
    if significant_diffs:
        print(f"\n[警告] 有 {len(significant_diffs)} 个嵌入的最大差异超过 {significant_threshold}")
        print(f"最大显著差异: {max(significant_diffs):.6e}")