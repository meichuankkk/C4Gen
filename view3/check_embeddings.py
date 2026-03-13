# -*- coding: utf-8 -*-
import pickle
import numpy as np


def check_embeddings_for_nan_inf(file_path):
    """
    检查嵌入文件中是否存在NaN或Inf值
    """
    try:
        # 加载文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print("[INFO] 加载文件:", file_path)
        print("[INFO] 文件内容键:", list(data.keys()))

        # 获取嵌入向量
        embeddings = data.get("embeddings", None)

        if embeddings is None:
            print("[WARNING] 文件中没有找到embeddings字段")
            return

        # 转换为numpy数组确保一致性
        embeddings_array = np.array(embeddings)

        # 检查是否为空的更好方法
        if embeddings_array.size == 0 or embeddings_array.shape[0] == 0:
            print("[WARNING] embeddings为空")
            return

        print("[INFO] 嵌入向量总数:", embeddings_array.shape[0])
        print("[INFO] 每个向量的维度:", embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 1)

        # 检查每个嵌入向量
        nan_count = 0
        inf_count = 0
        problematic_indices = []

        for idx in range(embeddings_array.shape[0]):
            emb = embeddings_array[idx]

            # 检查NaN
            has_nan = np.any(np.isnan(emb))
            # 检查Inf
            has_inf = np.any(np.isinf(emb))

            if has_nan:
                nan_count += 1
                problematic_indices.append(idx)
                print(f"  [WARNING] 索引 {idx}: 包含NaN值")

            if has_inf:
                inf_count += 1
                if idx not in problematic_indices:
                    problematic_indices.append(idx)
                print(f"  [WARNING] 索引 {idx}: 包含Inf值")

            # 可选：检查零向量或其他问题
            if np.all(emb == 0):
                print(f"  [WARNING] 索引 {idx}: 全零向量")

        # 输出统计信息
        print("\n" + "=" * 50)
        print("[检查结果统计]")
        print("总嵌入向量数:", embeddings_array.shape[0])
        print("包含NaN的向量数:", nan_count)
        print("包含Inf的向量数:", inf_count)
        print("问题向量总数:", len(problematic_indices))

        if problematic_indices:
            print("问题向量索引:", problematic_indices)

            # 可选：查看部分问题向量的具体信息
            print("\n[部分问题向量详情]")
            for i in problematic_indices[:min(5, len(problematic_indices))]:  # 只显示前5个
                emb = embeddings_array[i]
                nan_num = np.sum(np.isnan(emb))
                inf_num = np.sum(np.isinf(emb))
                print(f"索引 {i}: 形状={emb.shape}, NaN数={nan_num}, Inf数={inf_num}")
        else:
            print("[SUCCESS] 所有嵌入向量都没有NaN或Inf值！")

        # 检查嵌入向量的维度和统计信息
        if embeddings_array.shape[0] > 0:
            print(f"\n[嵌入向量信息]")
            print("嵌入形状:", embeddings_array.shape)
            print(f"数值范围: [{embeddings_array.min():.6f}, {embeddings_array.max():.6f}]")
            print(f"平均值: {embeddings_array.mean():.6f}")
            print(f"标准差: {embeddings_array.std():.6f}")

            # 检查是否有任何NaN或Inf（整体）
            has_nan_overall = np.any(np.isnan(embeddings_array))
            has_inf_overall = np.any(np.isinf(embeddings_array))
            print(f"整体是否有NaN: {has_nan_overall}")
            print(f"整体是否有Inf: {has_inf_overall}")

        return problematic_indices

    except Exception as e:
        print("[ERROR] 检查文件时出错:", e)
        import traceback
        traceback.print_exc()
        return []


def compare_files(original_path, updated_path):
    """
    比较原始文件和更新后的文件差异
    """
    print("\n" + "=" * 50)
    print("[比较原始文件和更新后文件]")

    try:
        # 加载原始文件
        with open(original_path, 'rb') as f:
            original_data = pickle.load(f)

        # 加载更新后的文件
        with open(updated_path, 'rb') as f:
            updated_data = pickle.load(f)

        original_embeddings = original_data.get("embeddings", None)
        updated_embeddings = updated_data.get("embeddings", None)

        if original_embeddings is None or updated_embeddings is None:
            print("[WARNING] 其中一个文件没有embeddings字段")
            return

        # 转换为numpy数组
        orig_array = np.array(original_embeddings)
        upd_array = np.array(updated_embeddings)

        print(f"原始文件嵌入数: {orig_array.shape[0]}")
        print(f"更新文件嵌入数: {upd_array.shape[0]}")

        # 检查数量是否一致
        if orig_array.shape[0] != upd_array.shape[0]:
            print("[WARNING] 两个文件的嵌入数量不一致！")
            return

        # 统计修复的向量
        fixed_count = 0
        unchanged_count = 0
        still_problematic = 0

        for i in range(orig_array.shape[0]):
            orig_emb = orig_array[i]
            upd_emb = upd_array[i]

            orig_has_nan_inf = np.any(np.isnan(orig_emb)) or np.any(np.isinf(orig_emb))
            upd_has_nan_inf = np.any(np.isnan(upd_emb)) or np.any(np.isinf(upd_emb))

            if orig_has_nan_inf and not upd_has_nan_inf:
                fixed_count += 1
            elif not orig_has_nan_inf and not upd_has_nan_inf:
                unchanged_count += 1
            elif orig_has_nan_inf and upd_has_nan_inf:
                still_problematic += 1
                print(f"  [WARNING] 索引 {i}: 原始和更新后都有问题")

        print(f"修复的向量数: {fixed_count}")
        print(f"正常未变的向量数: {unchanged_count}")
        print(f"仍然有问题的向量数: {still_problematic}")

        if fixed_count > 0:
            print(f"[SUCCESS] 成功修复了 {fixed_count} 个有问题的嵌入向量！")

        if still_problematic > 0:
            print(f"[WARNING] 还有 {still_problematic} 个向量仍然有问题")

        # 检查是否有任何向量被意外修改
        if fixed_count == 0 and still_problematic == 0:
            # 所有向量都没有NaN/Inf，检查它们是否完全相同
            if np.array_equal(orig_array, upd_array):
                print("[INFO] 两个文件完全相同")
            else:
                # 计算差异
                diff = np.abs(orig_array - upd_array)
                max_diff = diff.max()
                mean_diff = diff.mean()
                print(f"[INFO] 文件有差异但都没有NaN/Inf: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")

    except Exception as e:
        print("[ERROR] 比较文件时出错:", e)


if __name__ == "__main__":
    # 文件路径
    original_file = './resource/jina_diff_index.pkl'  # 原始文件
    updated_file = './resource/updated_jina_diff_index.pkl'  # 更新后的文件

    print("=" * 50)
    print("检查更新后的文件...")
    print("=" * 50)

    # 检查更新后的文件
    problematic_indices = check_embeddings_for_nan_inf(updated_file)

    # 比较原始文件和更新后的文件
    compare_files(original_file, updated_file)

    # 可选：也检查原始文件
    print("\n" + "=" * 50)
    print("检查原始文件...")
    print("=" * 50)
    check_embeddings_for_nan_inf(original_file)