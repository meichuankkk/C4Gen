import json
import random

INPUT_FILE = "../resource/apachecm/test.jsonl"
OUTPUT_FILE = "../resource/query.jsonl"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42  # 可选：保证结果可复现

def sample_jsonl(input_file, output_file, sample_size, seed=None):
    if seed is not None:
        random.seed(seed)

    # 读取所有数据
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    total = len(lines)
    if sample_size > total:
        raise ValueError(f"样本数 {sample_size} 大于数据总量 {total}")

    # 随机抽样
    sampled = random.sample(lines, sample_size)

    # 写入新 jsonl 文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已从 {total} 条数据中随机抽取 {sample_size} 条，输出至 {output_file}")

if __name__ == "__main__":
    sample_jsonl(INPUT_FILE, OUTPUT_FILE, SAMPLE_SIZE, RANDOM_SEED)
