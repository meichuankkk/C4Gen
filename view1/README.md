# C4Gen_view1 运行指南

本文档详细介绍了如何运行 `C4Gen_view1` 项目，从数据集准备到生成 Commit Message 并进行评估。

## 数据集说明

数据集位于 `/dataset/subset` 文件夹下，包含 `jsonl` 格式的文件。每条 JSON 数据包含以下属性：

- `git_url`: 仓库的 Git 地址
- `owner`: 仓库所有者
- `repo`: 仓库名称
- `commit_sha`: 提交哈希值
- `author`: 作者
- `date`: 日期
- `message`: 原始提交信息
- `files`: 修改的文件列表
- `diff`: Git 差异文本
- `loc`: 修改的代码行数

---

## 运行步骤

### 1. 仓库准备与 ENRE 报告生成

本步骤涉及代码仓库的本地准备以及 Java 项目的依赖关系分析。

#### 1.1 克隆仓库
首先，您需要自行根据数据集中的 `git_url` 将对应的仓库克隆到 `dataset/repo/` 对应的文件夹下。目前java项目已经克隆了一部分，python项目和c++项目的仓库还没有克隆。

**推荐指令：**
使用浅克隆（Shallow Clone）以节省空间和时间：
```bash
git clone --depth 1 <git_url> /dataset/repo/<repo_name>
```

#### 1.2 下载特定提交
运行 `process_dataset.py` 脚本，从本地仓库中获取数据集中指定的 `commit_sha`。

**指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_enre_reports/process_dataset.py
```
*注：该脚本中的路径目前为硬编码，建议根据实际需求修改脚本内的 `jsonl_file`、`base_repo_path` 和 `log_file` 变量。*

#### 1.3 生成 ENRE 报告
当前已支持 Java、Python、C++ 三种语言的 ENRE 报告批量生成。

**Java 指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_enre_reports/enre_report_generator_java.py \
    --dataset_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/java_subset.jsonl \
    --repo_dir /data/data_public/riverbag/C4Gen_view1/dataset_repo \
    --enre_jar /data/data_public/riverbag/C4Gen_view1/ENRE-tools/ENRE-java/jar/enre_java_2.0.1.jar \
    --output_dir /data/data_public/riverbag/C4Gen_view1/enre_py_reports/java_reports \
    --error_log /data/data_public/riverbag/C4Gen_view1/testINOut/enre_generator_errors.log
```
**Java 参数解释：**
- `--dataset_file`: 输入的数据集文件路径。
- `--repo_dir`: 包含克隆仓库的根目录。
- `--enre_jar`: ENRE 分析工具的 JAR 包路径。
- `--output_dir`: 生成报告的目录（建议使用 `enre_py_reports/java_reports`）。
- `--error_log`: 错误日志存放路径。

**Python 指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_enre_reports/enre_report_generator_py.py \
    --dataset_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/py_subset.jsonl \
    --repo_dir /data/data_public/riverbag/C4Gen_view1/dataset_repo \
    --enre_py_dir /data/data_public/riverbag/C4Gen_view1/ENRE-tools/ENRE-py \
    --output_dir /data/data_public/riverbag/C4Gen_view1/enre_py_reports/py_reports \
    --error_log /data/data_public/riverbag/C4Gen_view1/testINOut/enre_py_generator_errors.log
```
**Python 可选参数：**
- `--cfg`: 启用控制流分析。
- `--cg`: 输出调用图（通常与 `--cfg` 搭配使用）。
- `--compatible`: 输出兼容格式。
- `--builtins`: 指定 builtins 模块路径。

**C++ 指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_enre_reports/enre_report_generator_cpp.py \
    --dataset_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/cpp_subset.jsonl \
    --repo_dir /data/data_public/riverbag/C4Gen_view1/dataset_repo \
    --enre_jar /data/data_public/riverbag/C4Gen_view1/ENRE-tools/ENRE-cpp/enre_cpp.jar \
    --output_dir /data/data_public/riverbag/C4Gen_view1/enre_py_reports/cpp_reports \
    --error_log /data/data_public/riverbag/C4Gen_view1/testINOut/enre_cpp_generator_errors.log
```
**C++ 可选参数：**
- `--java_xmx`: Java 最大堆内存（默认 `4G`，大项目可设为 `64g`）。
- `--java_xms`: Java 初始堆内存（可选）。
- `-p/--program_environment`: 程序环境目录，可多次传入。
- `-d/--dir`: 额外分析目录，可多次传入。

---

### 2. 提取核心实体 (Core Entities)

通过 LLM 识别 Diff 中最关键的类或函数。

**指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_relatedcode/core_entities/extract_entities.py \
    --input_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/java_subset.jsonl \
    --output_file /data/data_public/riverbag/C4Gen_view1/dataset/core_entities_output.jsonl \
    --model deepseek-chat \
    --workers 10
```
**参数解释：**
- `--input_file`: 原始数据集文件。
- `--output_file`: 提取出的核心实体输出文件。
- `--model`: 使用的 LLM 模型名称（如 `deepseek-chat`）。
- `--workers`: 并行处理的线程数。

---

### 3. 获取相关代码上下文 (Related Code)

针对 Java 项目，根据核心实体自动检索相关的上下文代码。

**指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/java_workflow_processor.py \
    --core_entities_file /data/data_public/riverbag/C4Gen_view1/dataset/core_entities_output.jsonl \
    --commit_map_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/java_subset.jsonl \
    --all_enre_report_dir /data/data_public/riverbag/C4Gen_view1/enre_py_reports/java_reports \
    --all_repo_dir /data/data_public/riverbag/C4Gen_view1/dataset_repo/java_repo \
    --output_file /data/data_public/riverbag/C4Gen_view1/dataset/related_code_output.jsonl \
    --error_report_file /data/data_public/riverbag/C4Gen_view1/testINOut/workflow_errors.log
```
python /root/autodl-tmp/view1/workflow_processor_java.py \
    --core_entities_file /data/data_public/riverbag/C4Gen_view1/dataset/core_entities_output.jsonl \
    --commit_map_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/java_subset.jsonl \
    --all_enre_report_dir /data/data_public/riverbag/C4Gen_view1/enre_py_reports/java_reports \
    --all_repo_dir /data/data_public/riverbag/C4Gen_view1/dataset_repo/java_repo \
    --output_file /data/data_public/riverbag/C4Gen_view1/dataset/related_code_output.jsonl \
    --error_report_file /data/data_public/riverbag/C4Gen_view1/testINOut/workflow_errors.log









**参数解释：**
- `--core_entities_file`: 上一步生成的实体文件。
- `--commit_map_file`: 包含 `instance_id` 和 `commit_sha` 映射的数据集文件。(这个文件请用/data/data_public/riverbag/C4Gen_view1/utils_relatedcode/extract_instance_ids.py这个文件对不同语言的数据集自行生成)
- `--all_enre_report_dir`: ENRE 报告存放目录。
- `--all_repo_dir`: 仓库根目录。
- `--output_file`: 生成的相关代码输出文件。
- `--error_report_file`: 错误报告文件。

---

### 4. 生成 Commit Message

利用提取的上下文生成最终的 Commit Message。

#### 4.1 生成 LLM 任务
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_generate_commit_message/task_generator.py \
    --dataset_file /data/data_public/riverbag/C4Gen_view1/dataset/subset/java_subset.jsonl \
    --core_entities_file /data/data_public/riverbag/C4Gen_view1/dataset/core_entities_output.jsonl \
    --related_code_file /data/data_public/riverbag/C4Gen_view1/dataset/related_code_output.jsonl \
    --output_file /data/data_public/riverbag/C4Gen_view1/dataset/llm_tasks.jsonl \
    --mode related_code
```

#### 4.2 调用 LLM 批量生成
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_generate_commit_message/batch_commit_generator.py \
    --task_file /data/data_public/riverbag/C4Gen_view1/dataset/llm_tasks.jsonl \
    --output_file /data/data_public/riverbag/C4Gen_view1/dataset/generated_messages.jsonl \
    --model deepseek-chat \
    --concurrency 10
```

---

### 5. 效果评估 (Evaluation)

运行评估脚本计算 BLEU, ROUGE, METEOR, CIDEr 等分数。

**指令：**
```bash
python /data/data_public/riverbag/C4Gen_view1/utils_eval/eval.py \
    --result_jsonl /data/data_public/riverbag/C4Gen_view1/dataset/generated_messages.jsonl \
    --output_cleaned_jsonl /data/data_public/riverbag/C4Gen_view1/dataset/eval_results_cleaned.jsonl
```
**参数解释：**
- `--result_jsonl`: LLM 生成的结果文件。
- `--output_cleaned_jsonl`: (可选) 存放清洗后数据的路径。
