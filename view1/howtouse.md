jar包的目录在\ENRE-java\jar下面
- 你的仓库中有 jar 目录，一般会包含 enre_java.jar 。如果存在，直接用它运行。
- 基本命令格式： java -jar <executable> <lang> <src> <projectName> [options]
  - <executable> ： enre_java.jar 的绝对路径
  - <lang> ：填写 java
  - <src> ：要分析的源码目录（可以是整个项目或某个模块的源码目录）
  - <projectName> ：输出文件的项目名别称
  - [options] ：可选参数，例如 -o 指定输出文件名、 -d 追加多个源码目录、 -s 精简输出、 -e 指定第三方 API、 -a 指向 Android .aidl 对应 .java
示例一：分析一个较小模块（Camel 的 camel-core 源码）

```
java -jar e:\Grade_6\论文\A会\testENRE2\ENRE-java\jar\enre_java.jar 
java e:\Grade_6\论文\A会\testENRE2\camel\camel-core\src camel-core
```
示例二：指定输出文件名

```
java -jar e:\Grade_6\论文\A会\testENRE2\ENRE-java\jar\enre_java.jar 
java e:\Grade_6\论文\A会\testENRE2\camel\camel-core\src camel-core 
-o camel_core_out
```
示例三：增加内存堆（分析大型项目时建议）

```
java -Xmx16G -jar e:\Grade_6\论文\A会 \testENRE2\ENRE-java\jar\enre_java.jar java e:\Grade_6\论文\A会 \testENRE2\camel\camel-core\src camel-core
```
示例四：多目录分析（同一项目名，追加多个源码根）

```
java -jar e:\Grade_6\论文\A会\testENRE2\ENRE-java\jar\enre_java.jar  java e:\Grade_6\论文\A会\testENRE2\camel\core\camel-core\src camel-core -d e:\Grade_6\论文\A会
\testENRE2\camel\components\camel-http\src
```
示例五：精简输出（去掉位置信息等）

```
java -jar e:\Grade_6\论文\A会\testENRE2\ENRE-java\jar\enre_java.jar java e:\Grade_6\论文\A会\testENRE2\camel\camel-core\src camel-core -s
```



具体用例！！
- 分析 camel-core 模块的源码目录，输出项目名 camel_core （默认会生成 camel_core-out.json ，在当前目录）：
```
java -Xmx1G -jar "e:\Grade_6\论文\A会\testENRE2\ENRE-java\jar\enre_java_2.0.1.jar" java "e:\Grade_6\论文\A会\testENRE2\repo_test_java\src" repo_test_java -o repo_test_java_out
```
- 如果你想指定输出文件名（替代默认的 projectName-out ），用 -o 参数，例如：
```
java -Xmx8G -jar "e:\Grade_6\论文\A会
\testENRE2\ENRE-java\jar\enre_java.jar" java "e:\Grade_6\论文\A会
\testENRE2\camel\camel-core\src" camel_core -o camel_core_out
```
- 







-----------------------------------
1.怎么调用LLM识别出core entities:
python /data/data_public/riverbag/C4Gen/utils/extract_entities.py --input_file /data/data_public/riverbag/C4Gen/dataset/java_subset.jsonl --output_file /data/data_public/riverbag/C4Gen/dataset/10000_core_entities_output.jsonl --model deepseek-chat --workers 10

2.怎么生成数据集中每一条数据对应的enre report：
python /home/riverbag/C4Gen/enre_report_generator.py --dataset_file /home/riverbag/C4Gen/dataset/ApacheCM-mini/test/littlejava.jsonl --repo_dir /data/data_public/riverbag/dataset_repo --enre_jar /home/riverbag/C4Gen/ENRE-java/jar/enre_java_2.0.1.jar --output_dir /home/riverbag/C4Gen/testINOut/all_enre_reports

-----------------------------------
1.如何获取某个函数的调用/被调用信息：python e:\Grade_6\论文\A会\testENRE2\java_tools\relation_analyzer.py repo_test_java-enre-out\repo_test_java_out.json -q "com.example.demo.notifications.EmailNotifier.notify"

2.如何获取某个函数的调用/被调用代码：python e:\Grade_6\论文\A会\testENRE2\java_tools\code_retriever.py  --relation_report_file "E:\Grade_6\论文\A会\testENRE2\call_analysis_report.json"  --enre_report_file "E:\Grade_6\论文\A会\testENRE2\repo_test_java-enre-out\repo_test_java_out.json"  --project_root "e:/Grade_6/论文/A会/testENRE2/repo_test_java" --output_file "code_context_report.json"

3.怎么生成相应的related code：
python "e:/Grade_6/论文/A会/testENRE2/workflow_processor.py" --core_entities_file "E:\Grade_6\论文\A会\testENRE2\testINOut\all_core_entities_report.jsonl" --all_enre_report_dir "e:/Grade_6/论文/A会/testENRE2/testINOut/all_enre_reports/" --all_repo_dir "E:\Grade_6\paper\ASE\testENRE2\dataset_repo" --output_file "E:\Grade_6\论文\A会\testENRE2\testINOut\all_related_code.jsonl"   

python /data/data_public/riverbag/C4Gen/workflow_processor.py  --core_entities_file /data/data_public/riverbag/C4Gen/dataset/100_core_entities_output.jsonl  --commit_map_file /data/data_public/riverbag/C4Gen/dataset/id_sha.jsonl  --all_enre_report_dir /data/data_public/riverbag/C4Gen/testINOut/all_enre_reports  --all_repo_dir /data/data_public/riverbag/C4Gen/dataset_repo  --output_file /data/data_public/riverbag/C4Gen/testINOut/relevant_code/final_related_code.jsonl  --error_report_file /data/data_public/riverbag/C4Gen/testINOut/relevant_code/error_report.jsonl

4.如何生成任务：

direct 模式：python /data/data_public/riverbag/C4Gen/task_generator.py --dataset_path "/data/data_public/riverbag/C4Gen/dataset/littlejava.jsonl" --tasks_path "/data/data_public/riverbag/C4Gen/direct_tasks.jsonl" --prompt_type direct

related code模式：python /data/data_public/riverbag/C4Gen/task_generator.py --dataset_path "/data/data_public/riverbag/C4Gen/dataset/littlejava.jsonl" --core_entities_path "/data/data_public/riverbag/C4Gen/dataset/100_core_entities_output.jsonl" --relevant_code_path "/data/data_public/riverbag/C4Gen/testINOut/relevant_code/final_related_code.jsonl" --tasks_path "/data/data_public/riverbag/C4Gen/related_code_tasks.jsonl" --prompt_type related_code

5.怎么生成commit message：
python /data/data_public/riverbag/C4Gen/java_utils/batch_commit_generator.py --input "/data/data_public/riverbag/C4Gen/related_code_tasks.jsonl" --output "/data/data_public/riverbag/C4Gen/relatede_commit_messages.jsonl" --worker 10 --model "deepseek-chat"

6.怎么进行评分：
direct 模式：
cd /data/data_public/riverbag/C4Gen
python -m eval_utils.eval --result_jsonl /data/data_public/riverbag/C4Gen/direct_commit_messages.jsonl
python3 -m eval_utils.eval_IST  --result_path IST/predictions_processed
python3 -m eval_utils.eval_CCT5 --result_jsonl /data/data_public/riverbag/C4Gen/IST/baselines/CCT5/cct5/cpp/sw-eval.json 

related code模式：
python -m eval_utils.eval --result_jsonl /data/data_public/riverbag/C4Gen/related_commit_messages.jsonl --output_cleaned_jsonl /data/data_public/riverbag/C4Gen/cleaned_label_messages.jsonl

7.IST论文调权重参数
python /data/data_public/riverbag/C4Gen/IST/task_generator.py --input_path /data/data_public/riverbag/C4Gen/IST/CodeBERT-Result --output_path /data/data_public/riverbag/C4Gen/IST/tasks

python /data/data_public/riverbag/C4Gen/IST/batch_commit_generator.py --input /data/data_public/riverbag/C4Gen/IST/tasks/ --output /data/data_public/riverbag/C4Gen/IST/predictions/ --workers 20



------------------------------------------------------------------------
python项目
生成报告：
cd /data/data_public/riverbag/C4Gen_view1/enre_py_reports/py_reports

PYTHONPATH=/data/data_public/riverbag/C4Gen_view1/ENRE-tools/ENRE-py \
python -m enre /data/data_public/riverbag/C4Gen/test_python_project

跑全量/单实体的的关系：
全量：python /data/data_public/riverbag/C4Gen/py_utils/relation_analyzer.py data/data_public/riverbag/C4Gen enre_py_reports/test_python_project-report-enre.json -o /data/data_public/riverbag/C4Gen/enre_py_reports python_analysis_report.json
单实体：python /data/data_public/riverbag/C4Gen/py_utils/relation_analyzer.py /data/data_public/riverbag/C4Gen/enre_py_reports/test_python_project-report-enre.json -q test_python_project.service.run -o /data/data_public/riverbag/C4Gen/enre_py_reports/one_entity_relations.json

如何获取某个函数的调用/被调用代码
python /data/data_public/riverbag/C4Gen/py_utils/code_retriever.py \
  --relation_report_file /data/data_public/riverbag/C4Gen/enre_py_reports/one_entity_relations.json \
  --enre_report_file /data/data_public/riverbag/C4Gen/enre_py_reports/test_python_project-report-enre.json \
  --project_root /data/data_public/riverbag/C4Gen \
  --output_file /data/data_public/riverbag/C4Gen/enre_py_reports/python_code_context_report.json

------------------------------------------------------------
cpp项目
生成报告：
cd /data/data_public/riverbag/C4Gen/enre_py_reports/cpp_reports

java -jar /data/data_public/riverbag/C4Gen/ENRE-cpp/ENRE-CPP.jar \
  /data/data_public/riverbag/C4Gen/test_projects/test_cpp_project \
  test_cpp_project \
  -p /data/data_public/riverbag/C4Gen/test_projects/test_cpp_project/src \
  -d /data/data_public/riverbag/C4Gen/test_projects/test_cpp_project/include
- -p ：指定程序环境（这里用项目的 src 目录）
- -d ：额外需要分析的目录（这里把 include 也纳入）





  # 可能指向错误的pip（系统pip或其他环境）
pip install fire
# 明确使用当前python环境的pip
python -m pip install fire