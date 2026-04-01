import json
import concurrent.futures
import re
from typing import List, Dict, Any
import time
from openai import OpenAI
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/commit_gen.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepSeekAPIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", system_prompt: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def call_api(self, user_prompt: str, system_prompt: str = None, model: str = "deepseek-reasoner") -> Dict[str, Any]:
        if system_prompt is None:
            system_prompt = self.system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                'type': 'json_object'
            },
            stream=False,
            temperature=0
        )

        return json.loads(response.choices[0].message.content)

def read_commits_from_json(file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件读取commit数据，包含diff和issue信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info(f"成功读取commit文件，共 {len(data)} 条记录")

            # 直接返回原始数据，不做额外处理
            return data
    except Exception as e:
        logger.error(f"读取commit文件失败: {e}")
        return []

def process_control_group(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """处理对照组 - 只使用code diff"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 构建输入
        input_data = {
            "code_diff": code_diff
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理对照组提交: {commit_sha}")
        logger.debug(user_prompt)
        api_response = api_client.call_api(user_prompt)

        # 构建结果
        task_id = f"Apache_{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理对照组提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_experimental_group(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """处理实验组 - 使用code diff和所有issue信息（包括title和body）"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 获取issue信息
        issue_summary = commit_data.get('issue_summary', {})
        issue_body = ""
        for issue_id, issue_data in issue_summary.items():
            issue_body = issue_data.get('issue_body')
            break
        if not issue_body or not issue_body.strip():
            logger.error(f"commit {commit_sha} 的所有issue信息都为空")
            return None

        # 构建输入
        input_data = {
            "code_diff": code_diff,
            "issue": issue_body
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理实验组提交: {commit_sha}")
        logger.debug(user_prompt)
        api_response = api_client.call_api(user_prompt)

        # 构建结果
        task_id = f"{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理实验组提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_experimental_group_summary(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """处理实验组 - 使用code diff和issue summary信息"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 获取所有issue的summary信息
        issue_summary = commit_data.get('issue_summary', {})
        summary = {}
        for issue_id, issue_data in issue_summary.items():
            summary = issue_data.get('summary', {})
            break
        if not summary:
            logger.error(f"commit {commit_sha} 的所有issue summary信息都为空")
            return None

        functionality = summary.get('Functionality', [])
        concept = summary.get('Concept', [])
        directive = summary.get('Directive', [])
        rationale = summary.get('Rationale', [])
        implication = summary.get('Implication', [])


        summary_text = ""
        if functionality:
            summary_text += f"Functionality: {', '.join(functionality)},\n"
        if concept:
            summary_text += f"Concept: {', '.join(concept)},\n"
        if directive:
            summary_text += f"Directive: {', '.join(directive)},\n"
        if rationale:
            summary_text += f"Rationale: {', '.join(rationale)},\n"
        if implication:
            summary_text += f"Implication: {', '.join(rationale)}\n"

        # 构建输入
        input_data = {
            "code_diff": code_diff,
            "issue_summary": summary_text
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理实验组(Summary)提交: {commit_sha}")
        logger.debug(user_prompt)
        api_response = api_client.call_api(user_prompt)

        # 构建结果
        task_id = f"{repo}_{commit_sha}_summary"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理实验组(Summary)提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_experimental_group_similar(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """处理实验组 - 使用code diff+issue信息, similar code diff + issue"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 获取issue信息
        issue_summary = commit_data.get('issue_summary', {})
        issue_body = ""
        for issue_id, issue_data in issue_summary.items():
            issue_body = issue_data.get('issue_body')
            break
        if not issue_body or not issue_body.strip():
            logger.error(f"commit {commit_sha} 的所有issue信息都为空")
            return None

        # 获取similar commit 信息
        similar_commit = commit_data.get('retrieved')
        if not similar_commit:
            logger.error(f"commit {commit_sha} 的similar commit信息都为空")
            return None
        similar_commit_code_diff = similar_commit.get('diff')
        if not similar_commit_code_diff:
            logger.error(f"commit {commit_sha} 的similar commit diff为空")
            return None
        similar_commit_issue = similar_commit.get('issue_summary',{})
        similar_commit_issue_body = ""
        for issue_id, issue_data in similar_commit_issue.items():
            similar_commit_issue_body = issue_data.get('issue_body')
            break
        if not similar_commit_issue_body or not similar_commit_issue_body.strip():
            logger.error(f"commit {commit_sha} 的similar commit issue body为空")
            return None

        commit_example = {
            "code_diff": similar_commit_code_diff,
            "issue": similar_commit_issue_body
        }
        task = {
            "code_diff": code_diff,
            "issue": issue_body
        }

        # 构建输入
        input_data = {
            "a similar commit example": commit_example,
            "your task": task,
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理实验组提交: {commit_sha}")
        logger.debug(user_prompt)
        api_response = api_client.call_api(user_prompt)

        # 构建结果
        task_id = f"{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理实验组提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_experimental_group_similar_message(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """处理实验组 - 使用code diff+issue信息, similar code diff + issue"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 获取issue信息
        issue_summary = commit_data.get('issue_summary', {})
        issue_body = ""
        for issue_id, issue_data in issue_summary.items():
            issue_body = issue_data.get('issue_body')
            break
        if not issue_body or not issue_body.strip():
            logger.error(f"commit {commit_sha} 的所有issue信息都为空")
            return None

        # 获取similar commit 信息
        similar_commit = commit_data.get('retrieved')
        if not similar_commit:
            logger.error(f"commit {commit_sha} 的similar commit信息都为空")
            return None
        similar_commit_code_diff = similar_commit.get('diff')
        if not similar_commit_code_diff:
            logger.error(f"commit {commit_sha} 的similar commit diff为空")
            return None
        similar_commit_message = similar_commit.get('message')
        if not similar_commit_message:
            logger.error(f"commit {commit_sha} 的similar commit message为空")
            return None

        similar_commit_issue = similar_commit.get('issue_summary', {})
        similar_commit_issue_body = ""
        for issue_id, issue_data in similar_commit_issue.items():
            similar_commit_issue_body = issue_data.get('issue_body')
            break
        if not similar_commit_issue_body or not similar_commit_issue_body.strip():
            logger.error(f"commit {commit_sha} 的similar commit issue body为空")
            return None

        commit_example = {
            "code_diff": similar_commit_code_diff,
            "issue": similar_commit_issue_body,
            "commit_message": similar_commit_message
        }
        task = {
            "code_diff": code_diff,
            "issue": issue_body
        }

        # 构建输入
        input_data = {
            "a similar commit example": commit_example,
            "your task": task,
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理实验组提交: {commit_sha}")
        logger.debug(user_prompt)
        api_response = api_client.call_api(user_prompt)

        # 构建结果
        task_id = f"{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理实验组提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_experimental_group_similar_message2(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient, system_prompt: str) -> Dict[str, Any] | any:
    """处理实验组 - 使用code diff+issue信息, similar code diff + issue"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 获取issue信息
        issue_summary = commit_data.get('issue_summary', {})
        issue_body = ""
        for issue_id, issue_data in issue_summary.items():
            issue_body = issue_data.get('issue_body')
            break
        if not issue_body or not issue_body.strip():
            logger.error(f"commit {commit_sha} 的所有issue信息都为空")
            return None

        # 获取similar commit 信息
        similar_commit = commit_data.get('retrieved')
        if not similar_commit:
            logger.error(f"commit {commit_sha} 的similar commit信息都为空")
            return None
        similar_commit_code_diff = similar_commit.get('diff')
        if not similar_commit_code_diff:
            logger.error(f"commit {commit_sha} 的similar commit diff为空")
            return None
        similar_commit_message = similar_commit.get('message')
        if not similar_commit_message:
            logger.error(f"commit {commit_sha} 的similar commit message为空")
            return None

        similar_commit_issue = similar_commit.get('issue_summary', {})
        similar_commit_issue_body = ""
        for issue_id, issue_data in similar_commit_issue.items():
            similar_commit_issue_body = issue_data.get('issue_body')
            break
        if not similar_commit_issue_body or not similar_commit_issue_body.strip():
            logger.error(f"commit {commit_sha} 的similar commit issue body为空")
            return None

        commit_example = {
            "code_diff": similar_commit_code_diff,
            "issue": similar_commit_issue_body,
            "commit_message": similar_commit_message
        }
        task = {
            "code_diff": code_diff,
            "issue": issue_body
        }

        # 构建输入
        input_data = {
            "a similar commit example": commit_example,
            "your task": task,
        }
        system_prompt = system_prompt.format(commit_example)
        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理实验组提交: {commit_sha}")
        logger.debug(f"""system_prompt: {system_prompt},"\n", user_prompt: {user_prompt}""")
        api_response = api_client.call_api(user_prompt=user_prompt, system_prompt=system_prompt)

        # 构建结果
        task_id = f"{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理实验组提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_experimental_group_summary_similar_message(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient, system_prompt: str) -> Dict[str, Any] | any:
    """处理实验组 - 使用code diff+issue summary信息, similar code diff + issue summary"""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} 的diff为空")
            return None

        # 获取issue信息
        issue = commit_data.get('issue_summary', {})
        issue_summary = {}
        for issue_id, issue_data in issue.items():
            issue_summary = issue_data.get('summary', {})
            break
        if not issue_summary or issue_summary == {}:
            logger.error(f"commit {commit_sha} 的所有issue summary 都为空")
            return None

        # 获取similar commit 信息
        similar_commit = commit_data.get('retrieved')
        if not similar_commit:
            logger.error(f"commit {commit_sha} 的similar commit信息都为空")
            return None
        similar_commit_code_diff = similar_commit.get('diff')
        if not similar_commit_code_diff:
            logger.error(f"commit {commit_sha} 的similar commit diff为空")
            return None
        similar_commit_message = similar_commit.get('message')
        if not similar_commit_message:
            logger.error(f"commit {commit_sha} 的similar commit message为空")
            return None

        similar_commit_issue = similar_commit.get('issue_summary', {})
        similar_commit_issue_summary = {}
        for issue_id, issue_data in similar_commit_issue.items():
            similar_commit_issue_summary = issue_data.get('summary')
            break
        if not similar_commit_issue_summary or similar_commit_issue_summary == {}:
            logger.error(f"commit {commit_sha} 的similar commit issue summary 为空")
            return None

        commit_example = {
            "code_diff": similar_commit_code_diff,
            "issue": similar_commit_issue_summary,
            "commit_message": similar_commit_message
        }
        task = {
            "code_diff": code_diff,
            "issue": issue_summary
        }

        # 构建输入
        input_data = {
            "a similar commit example": commit_example,
            "your task": task,
        }
        system_prompt = system_prompt.format(commit_example)
        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # 调用API
        logger.info(f"处理实验组提交: {commit_sha}")
        logger.debug(f"""system_prompt: {system_prompt},"\n", user_prompt: {user_prompt}""")
        api_response = api_client.call_api(user_prompt=user_prompt, system_prompt=system_prompt)

        # 构建结果
        task_id = f"{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": "deepseek-reasoner",
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"处理实验组提交 {commit_data.get('commit_sha', 'unknown')} 时出错: {e}")
        return None

def process_group_with_threadpool(commits: List[Dict[str, Any]], process_func, api_client: DeepSeekAPIClient, system_prompt: str = None, max_workers: int = 5) -> List[Dict[str, Any]]:
    """使用线程池处理指定组的数据"""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        if system_prompt is None:
            future_to_commit = {
                executor.submit(process_func, commit, api_client): commit
                for commit in commits
            }
        else:
            future_to_commit = {
                executor.submit(process_func, commit, api_client, system_prompt): commit
                for commit in commits
            }

        # 收集结果
        for future in concurrent.futures.as_completed(future_to_commit):
            commit = future_to_commit[future]
            try:
                result = future.result()
                if result:  # 只添加有效结果
                    results.append(result)
                    logger.info(f"完成处理: {commit['commit_sha']}")
            except Exception as e:
                logger.error(f"处理提交 {commit['commit_sha']} 时发生异常: {e}")

    return results

def save_results_to_jsonl(results: List[Dict[str, Any]], output_file: str):
    """保存结果到JSONL文件"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}, 共 {len(results)} 条记录")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def main():
    # 配置参数
    api_key = ""
    # 读summary文件，因为有些commit的issue body毫无意义，被过滤掉了
    input_file = "../issue/data_summary/deepseek-reasoner/spark_github_issue_with_similar.json"
    filter_file = "deepseek-reasoner/spark/without_issue.json"
    control_output_file = "deepseek-reasoner/experiment3/without_issue.json"
    experimental_output_file = "deepseek-reasoner/experiment3/with_full_issue.json"
    experimental_summary_output_file = "deepseek-reasoner/experiment3/with_summary_issue.json"
    experimental_similar_file = "deepseek-reasoner/spark/similar.json"
    # experimental_similar_message_file = "deepseek-reasoner/spark/similar_with_message.json"
    experimental_summary_similar_message_file = "deepseek-reasoner/spark/similar_with_message_with_summary2.json"

    # 系统提示词
    system_prompt_control = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    system_prompt_experimental = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) and the related issue in a commit.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    system_prompt_experimental_summary = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) and the supplementary information retrieved from Issue in a commit.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    system_prompt_experimental_similar = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) and the related issue in a commit. First, a similar commit example (including both code diff and issue) is provided for reference. Then, you will be given a code diff and the related issue which is your task, and you need to write a commit message for it.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    system_prompt_experimental_similar_message = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) and the related issue in a commit. First, a similar commit example (including code diff, issue and commit message) is provided for reference. Then, you will be given a code diff and the related issue which is your task, and you need to write a commit message for it.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    system_prompt_experimental_similar_message2 = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) and the related issue in a commit.
    this is a similar commit example (including code diff, issue and commit message) is provided for reference. 
    {}
    Then, you will be given a code diff and the related issue which is your task, and you need to write a commit message for it
    Output format: A JSON object with a single key "commit_message" containing the concise commit message."""

    system_prompt_experimental_summary_similar_message = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) and the supplementary information retrieved from Issue (issue summary) in a commit. 
    this is a similar commit example (including code diff, issue summary and commit message) is provided for reference. 
    {}
    Then, you will be given a code diff and the related issue summary which is your task, and you need to write a commit message for it.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message."""

    my_system_prompt = """ """
    # 读取数据
    commits = read_commits_from_json(input_file)
    # commits = commits[0:2]
    filter_file_path = "deepseek-reasoner/spark/similar_with_message_with_summary.json"
    filter_commits = read_commits_from_json(filter_file_path)
    filter_sha = [d.get('task_id').rsplit('_', 1)[-1] for d in filter_commits]
    commits = [d for d in commits if d.get('commit_sha') not in filter_sha]
    logger.info(f"未处理的commit还有{len(commits)}条")

    for commit in commits:
        original_message = commit['message']
        # 使用正则表达式移除开头的一个或多个[单词-数字]格式的标签
        cleaned_message = re.sub(r'^(\[\w+-\d+\]\s*)+', '', original_message)
        commit['message'] = cleaned_message
        retrieved_message = commit['retrieved']['message']
        cleaned_retrieved_message = re.sub(r'^(\[\w+-\d+\]\s*)+', '', retrieved_message)
        commit['retrieved']['message'] = cleaned_retrieved_message


    if not commits:
        logger.error("数据读取失败")
        return

    logger.info(f"成功读取 {len(commits)} 个commit记录")

    # 创建API客户端
    control_client = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_control)
    experimental_client = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_experimental)
    experimental_summary_client = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_experimental_summary)
    experimental_similar_client = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_experimental_similar)
    experimental_similar_message_client = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_experimental_similar_message)
    experimental_similar_message_client2 = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_experimental_similar_message2)
    experimental_summary_similar_message_client = DeepSeekAPIClient(api_key=api_key, system_prompt=system_prompt_experimental_summary_similar_message)


    max_workers = 400
    # 使用线程池并行运行三组实验
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 提交所有任务
        # 处理对照组
        # future_control = executor.submit(
        #     process_group_with_threadpool, commits, process_control_group, control_client, max_workers
        # )
        # # 处理实验组 完整信息
        # future_experimental = executor.submit(
        #     process_group_with_threadpool, commits, process_experimental_group, experimental_client, max_workers
        # )
        # # 处理实验组 summary
        # future_experimental_summary = executor.submit(
        #     process_group_with_threadpool, commits, process_experimental_group_summary, experimental_summary_client,
        #     max_workers
        # )
        # # 处理实验组 similar commit without message
        # future_experimental_similar = executor.submit(
        #     process_group_with_threadpool, commits, process_experimental_group_similar, experimental_similar_client,
        #     max_workers
        # )
        # # 处理实验组 similar commit with message
        # future_experimental_similar_message = executor.submit(
        #     process_group_with_threadpool, commits, process_experimental_group_similar_message, experimental_similar_message_client,
        #     max_workers
        # )
        # # 处理实验组 similar commit with message 放在system_prompt里
        # future_experimental_similar_message2 = executor.submit(
        #     process_group_with_threadpool, commits, process_experimental_group_similar_message2, experimental_similar_message_client2, system_prompt_experimental_similar_message2,
        #     max_workers
        # )
        # 处理实验组 similar commit with message with issue summary
        future_experimental_summary_similar_message = executor.submit(
            process_group_with_threadpool, commits=commits, process_func=process_experimental_group_summary_similar_message, api_client=experimental_summary_similar_message_client,
            system_prompt=system_prompt_experimental_summary_similar_message, max_workers=max_workers
        )
    # control_results = future_control.result()
    # experimental_results = future_experimental.result()
    # experimental_summary_results = future_experimental_summary.result()
    # experimental_similar_results = future_experimental_similar.result()
    # experimental_similar_message_results = future_experimental_similar_message.result()
    # experimental_similar_message_results2 = future_experimental_similar_message2.result()
    experimental_summary_similar_message_results = future_experimental_summary_similar_message.result()

    # 保存结果
    # save_results_to_jsonl(control_results, control_output_file)
    # save_results_to_jsonl(experimental_results, experimental_output_file)
    # save_results_to_jsonl(experimental_summary_results, experimental_summary_output_file)
    # save_results_to_jsonl(experimental_similar_results, experimental_similar_file)
    # save_results_to_jsonl(experimental_similar_message_results, experimental_similar_message_file)
    # save_results_to_jsonl(experimental_similar_message_results2, experimental_similar_message_file2)
    save_results_to_jsonl(experimental_summary_similar_message_results, experimental_summary_similar_message_file)

    # 统计信息
    logger.info(f"实验完成!")
    # logger.info(f"对照组: {len(control_results)}/{len(commits)} 成功")
    # logger.info(f"实验组1: {len(experimental_results)}/{len(commits)} 成功")
    # logger.info(f"实验组2: {len(experimental_summary_results)}/{len(commits)} 成功")
    # logger.info(f"实验组3: {len(experimental_similar_results)}/{len(commits)} 成功")
    # logger.info(f"实验组4: {len(experimental_similar_message_results)}/{len(commits)} 成功")
    # logger.info(f"实验组4.2: {len(experimental_similar_message_results2)}/{len(commits)} 成功")
    logger.info(f"实验组5: {len(experimental_summary_similar_message_results)}/{len(commits)} 成功")


if __name__ == "__main__":
    main()