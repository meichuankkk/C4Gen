#!/usr/bin/env python3
"""
下载 ApacheCM-mini/python_test.jsonl 中每个entry的GitHub仓库snapshot和对应的commit message
"""

import json
import os
import subprocess
import logging
import shutil
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_repository_message.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# GitHub Token 配置（仅使用此处配置，不从环境变量读取）
# ============================================================================
# Token 可能过几小时失效的常见原因:
# 1. 创建时设置了过期时间(如 7/30/90 天)
# 2. 在 GitHub 设置里被手动撤销
# 3. 被误提交到仓库后 GitHub 自动撤销
# 4. 细粒度 token 权限不足或过期
# 解决: 到 https://github.com/settings/tokens 新建 token，更新下方 GITHUB_TOKEN_FROM_CONFIG
# ============================================================================
GITHUB_USERNAME = "ffwjf"
GITHUB_TOKEN_FROM_CONFIG = ""
# ============================================================================


def mask_token(token: Optional[str]) -> str:
    """脱敏显示 token，便于日志排查是哪个 token 失败（前7字符 + *** + 末4字符）。"""
    if not token or len(token) <= 11:
        return "(空或过短)"
    return f"{token[:7]}***{token[-4:]}"


def verify_github_token(token: Optional[str]) -> bool:
    """
    验证 GitHub token 是否有效，并打印诊断信息（状态码、响应体、速率限制等）。
    用于排查 401 认证失败原因。
    """
    if not token:
        logger.warning("[Token 验证] 未提供 token，跳过验证")
        return False
    
    api_url = "https://api.github.com/user"
    logger.info("[Token 验证] 正在验证 token...")
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "Python-download-script")
        
        http_proxy, https_proxy = get_proxy_settings()
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies["http"] = http_proxy
            if https_proxy:
                proxies["https"] = https_proxy
            opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        else:
            opener = urllib.request.build_opener()
        
        with opener.open(req, timeout=15) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            limit = resp.headers.get("X-RateLimit-Limit", "")
            remaining = resp.headers.get("X-RateLimit-Remaining", "")
            try:
                data = json.loads(body)
                login = data.get("login", "")
                logger.info(f"[Token 验证] ✓ 成功 (HTTP {status}) 当前用户: {login}, API 剩余次数: {remaining}/{limit}")
            except Exception:
                logger.info(f"[Token 验证] ✓ 成功 (HTTP {status}), 剩余次数: {remaining}/{limit}")
            return True
            
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        remaining = e.headers.get("X-RateLimit-Remaining", "")
        limit = e.headers.get("X-RateLimit-Limit", "")
        logger.error(f"[Token 验证] ✗ 失败 HTTP {e.code} {e.reason}")
        logger.error(f"[Token 验证] 响应体: {body[:400]}")
        logger.error(f"[Token 验证] X-RateLimit-Remaining: {remaining}, X-RateLimit-Limit: {limit}")
        if e.code == 401:
            logger.error(f"[Token 验证] 认证失败时使用的 token（脱敏）: {mask_token(token)}")
            logger.error("[Token 验证] 可能原因: 1) token 复制不完整 2) 创建时未勾选 repo 等权限 3) 细粒度 token 未授权该仓库 4) token 已过期/被撤销")
        return False
    except Exception as e:
        logger.error(f"[Token 验证] ✗ 请求异常: {type(e).__name__}: {e}")
        return False


def get_proxy_settings() -> Tuple[Optional[str], Optional[str]]:
    """
    获取代理设置（优先使用环境变量，如果没有则尝试读取 /etc/network_turbo）
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (http_proxy, https_proxy)
    """
    # 首先检查环境变量
    http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
    
    if http_proxy and https_proxy:
        logger.info(f"使用环境变量中的代理: http_proxy={http_proxy}, https_proxy={https_proxy}")
        return http_proxy, https_proxy
    
    # 如果没有环境变量，尝试读取 /etc/network_turbo 文件
    network_turbo_file = Path('/etc/network_turbo')
    if network_turbo_file.exists():
        try:
            with open(network_turbo_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 提取代理地址
                import re
                proxy_match = re.search(r'export\s+(?:http|https)_proxy=([^\s&]+)', content)
                if proxy_match:
                    proxy_url = proxy_match.group(1)
                    logger.info(f"从 /etc/network_turbo 读取代理: {proxy_url}")
                    return proxy_url, proxy_url
        except Exception as e:
            logger.warning(f"读取 /etc/network_turbo 失败: {str(e)}")
    
    # 如果都没有，使用默认的代理地址（从 /etc/network_turbo 中看到的）
    default_proxy = "http://172.32.52.144:12798"
    logger.info(f"使用默认代理: {default_proxy}")
    return default_proxy, default_proxy


def get_github_zip_urls(owner: str, repo: str, commit_sha: str) -> list:
    """
    生成多个GitHub zip下载URL（包括代理，使用HTTP）
    
    Args:
        owner: 仓库所有者
        repo: 仓库名
        commit_sha: commit SHA
        
    Returns:
        list: URL列表，按优先级排序
    """
    base_url = f"http://github.com/{owner}/{repo}/archive/{commit_sha}.zip"
    
    # 多个GitHub代理服务（使用HTTP）
    urls = [
        # 1. ghproxy.com 代理（最常用）
        f"http://ghproxy.com/{base_url}",
        # 2. hub.fastgit.xyz 镜像
        f"http://hub.fastgit.xyz/{owner}/{repo}/archive/{commit_sha}.zip",
        # 3. github.com.cnpmjs.org 镜像
        f"http://github.com.cnpmjs.org/{owner}/{repo}/archive/{commit_sha}.zip",
        # 4. mirror.ghproxy.com 代理
        f"http://mirror.ghproxy.com/{base_url}",
        # 5. 原始GitHub URL（最后尝试）
        base_url,
    ]
    
    return urls


def download_repository_zip_via_api(owner: str, repo: str, commit_sha: str, zip_file: Path, token: Optional[str] = None, token_source: str = "") -> bool:
    """
    使用GitHub API下载指定commit的zip文件（最快最直接的方法）
    
    Args:
        owner: 仓库所有者
        repo: 仓库名
        commit_sha: commit SHA
        zip_file: 保存zip文件的路径
        token: GitHub Personal Access Token（可选，但推荐使用以提高速率限制）
        
    Returns:
        bool: 是否下载成功
    """
    # 确保父目录存在
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件已存在，先删除
    if zip_file.exists():
        zip_file.unlink()
    
    # 构建GitHub API URL
    api_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{commit_sha}"
    
    logger.info(f"正在通过GitHub API下载: {api_url} (commit: {commit_sha[:8]}...)")
    
    try:
        # 获取代理设置
        http_proxy, https_proxy = get_proxy_settings()
        
        # 创建请求
        req = urllib.request.Request(api_url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # 如果提供了token，添加认证头
        if token:
            req.add_header('Authorization', f'Bearer {token}')
            logger.debug("使用GitHub token进行认证")
        else:
            logger.warning("未提供GitHub token，速率限制较低（60次/小时）")
        
        # 创建代理处理器
        proxy_handler = None
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
            proxy_handler = urllib.request.ProxyHandler(proxies)
        
        # 创建opener
        if proxy_handler:
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()
        
        # 下载文件
        with opener.open(req, timeout=300) as response:
            # 检查响应状态
            if response.status != 200:
                logger.warning(f"API返回状态码: {response.status}")
                return False
            
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(zip_file, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 每下载10MB输出一次进度
                    if downloaded % (10 * 1024 * 1024) < chunk_size:
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.debug(f"下载进度: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)")
        
        # 检查文件是否下载成功
        if zip_file.exists() and zip_file.stat().st_size > 0:
            file_size_mb = zip_file.stat().st_size // 1024 // 1024
            logger.info(f"✓ API下载成功: {zip_file.name} ({file_size_mb}MB)")
            return True
        else:
            logger.error(f"API下载失败或文件为空")
            return False
            
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', errors='replace')[:500] if e.fp else ""
        except Exception:
            body = ""
        # 速率限制是 403，认证失败是 401，两者不同
        if e.code == 401:
            source_desc = f"，来源: {token_source}" if token_source else ""
            logger.error(f"认证失败(401): token 无效/过期/被撤销，不是速率限制。当前使用的 token（脱敏）: {mask_token(token)}{source_desc}。响应: {body[:200] if body else '无'}")
        elif e.code == 403:
            # 检查是否是速率限制：响应里常有 "rate limit" 或 "API rate limit"
            is_rate_limit = body and ("rate limit" in body.lower() or "rate_limit" in body.lower())
            remaining = e.headers.get("X-RateLimit-Remaining", "")
            limit = e.headers.get("X-RateLimit-Limit", "")
            reset = e.headers.get("X-RateLimit-Reset", "")
            if is_rate_limit or (remaining == "0" and limit):
                reset_msg = f"，重置时间(UTC): {time.strftime('%Y-%m-%d %H:%M', time.gmtime(int(reset)))}" if reset.isdigit() else ""
                logger.error(f"API 速率限制(403): 本小时额度已用尽(剩余 {remaining}/{limit}){reset_msg}。等重置后继续或改用 git clone。")
            else:
                logger.error(f"访问被拒绝(403): 可能是权限不足或其它限制。响应: {body[:200] if body else '无'}")
        elif e.code == 404:
            logger.error(f"仓库或commit不存在(404)")
        else:
            logger.error(f"HTTP错误 {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        logger.error(f"URL错误: {e.reason}")
        return False
    except Exception as e:
        logger.error(f"API下载出错: {str(e)}")
        return False


def download_repository_zip(owner: str, repo: str, commit_sha: str, zip_file: Path) -> bool:
    """
    下载指定commit的仓库zip文件（优先使用API，失败则使用git clone）
    
    Args:
        owner: 仓库所有者
        repo: 仓库名
        commit_sha: commit SHA
        zip_file: 保存zip文件的路径
        
    Returns:
        bool: 是否下载成功
    """
    import tempfile
    
    # 确保父目录存在
    zip_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件已存在，先删除
    if zip_file.exists():
        zip_file.unlink()
    
    # 方法1: 优先尝试使用GitHub API（更快更直接）
    # 仅使用脚本中的 token，不从环境变量读取
    github_token = GITHUB_TOKEN_FROM_CONFIG
    token_source = "脚本配置 GITHUB_TOKEN_FROM_CONFIG"
    if github_token:
        logger.info(f"尝试使用GitHub API下载...")
        if download_repository_zip_via_api(owner, repo, commit_sha, zip_file, github_token, token_source):
            return True
        logger.warning(f"API下载失败，尝试使用git clone方法...")
    else:
        logger.info(f"脚本中未配置 GITHUB_TOKEN_FROM_CONFIG，跳过API方法，直接使用git clone...")
    
    # 方法2: 如果API失败或没有token，使用git clone + git archive
    # 构建GitHub仓库URL
    git_url = f"https://github.com/{owner}/{repo}.git"
    
    # 创建临时目录用于clone
    temp_dir = None
    try:
        # 使用临时目录
        temp_base = Path('/tmp') / 'git_clone_temp'
        temp_base.mkdir(exist_ok=True)
        temp_dir = temp_base / f"{owner}_{repo}_{commit_sha[:8]}"
        
        # 如果临时目录已存在，先删除
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        logger.info(f"正在clone仓库: {git_url} (commit: {commit_sha[:8]}...)")
        
        # 方法1: 尝试浅克隆（更快）
        # 先尝试只clone最近的历史，然后fetch指定commit
        clone_cmd = [
            'git', 'clone', '--depth', '100', '--no-single-branch',
            git_url, str(temp_dir)
        ]
        
        result = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=os.environ.copy()  # 使用当前环境变量（包括代理）
        )
        
        if result.returncode == 0:
            # 尝试fetch指定commit
            fetch_cmd = ['git', 'fetch', '--depth', '100', 'origin', commit_sha]
            subprocess.run(
                fetch_cmd,
                cwd=temp_dir,
                capture_output=True,
                timeout=120
            )
            
            # checkout到指定commit
            checkout_cmd = ['git', 'checkout', commit_sha]
            checkout_result = subprocess.run(
                checkout_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if checkout_result.returncode != 0:
                # 如果浅克隆失败，尝试完整clone
                logger.info(f"浅克隆中找不到commit，尝试完整clone...")
                shutil.rmtree(temp_dir, ignore_errors=True)
                result = None
            else:
                logger.info(f"浅克隆成功并checkout到指定commit")
        
        # 方法2: 如果浅克隆失败，使用完整clone
        if result is None or result.returncode != 0:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info(f"使用完整clone方式")
            clone_cmd_full = ['git', 'clone', git_url, str(temp_dir)]
            result_full = subprocess.run(
                clone_cmd_full,
                capture_output=True,
                text=True,
                timeout=600,
                env=os.environ.copy()
            )
            
            if result_full.returncode != 0:
                logger.error(f"完整clone失败: {result_full.stderr}")
                return False
            
            # checkout到指定commit
            checkout_cmd = ['git', 'checkout', commit_sha]
            checkout_result = subprocess.run(
                checkout_cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if checkout_result.returncode != 0:
                logger.error(f"checkout失败: {checkout_result.stderr}")
                return False
        
        # 使用git archive创建zip文件
        logger.info(f"正在创建zip文件: {zip_file}")
        # 使用-o参数（更通用的格式）
        archive_cmd = ['git', 'archive', '--format=zip', '-o', str(zip_file), commit_sha]
        archive_result = subprocess.run(
            archive_cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if archive_result.returncode != 0:
            logger.error(f"git archive失败: {archive_result.stderr}")
            return False
        
        # 检查zip文件是否创建成功
        if zip_file.exists() and zip_file.stat().st_size > 0:
            file_size_mb = zip_file.stat().st_size // 1024 // 1024
            logger.info(f"✓ zip文件创建成功: {zip_file.name} ({file_size_mb}MB)")
            return True
        else:
            logger.error(f"zip文件创建失败或文件为空")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"操作超时: {git_url} (commit: {commit_sha[:8]}...)")
        return False
    except Exception as e:
        logger.error(f"创建zip文件时出错: {git_url} (commit: {commit_sha[:8]}...), 错误: {str(e)}")
        return False
    finally:
        # 清理临时目录
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass


def sanitize_dir_name(name: str) -> str:
    """
    清理目录名，移除或替换不允许的字符
    
    Args:
        name: 原始目录名
        
    Returns:
        str: 清理后的目录名
    """
    # 替换不允许的文件系统字符
    # Windows/Linux不允许的字符: / \ : * ? " < > |
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 移除前后空格和点
    name = name.strip(' .')
    # 限制长度（避免文件名过长）
    if len(name) > 200:
        name = name[:200]
    return name


def generate_pair_dir_name(repo: str, commit_sha: str, sha_length: int = 8) -> str:
    """
    生成pair目录名: repo名字_commit_sha前n位
    
    Args:
        repo: 仓库名
        commit_sha: commit SHA
        sha_length: commit SHA使用的长度（默认8位）
        
    Returns:
        str: 目录名
    """
    repo_clean = sanitize_dir_name(repo)
    sha_prefix = commit_sha[:sha_length] if commit_sha else ''
    dir_name = f"{repo_clean}_{sha_prefix}"
    return dir_name


def save_commit_message(message: str, message_file: Path) -> bool:
    """
    保存commit message到文件
    
    Args:
        message: commit message内容
        message_file: 保存的文件路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        message_file.parent.mkdir(parents=True, exist_ok=True)
        with open(message_file, 'w', encoding='utf-8') as f:
            f.write(message)
        return True
    except Exception as e:
        logger.error(f"保存commit message失败: {message_file}, 错误: {str(e)}")
        return False


def process_entry(entry: Dict, index: int, output_base_dir: Path, sha_length: int = 8) -> Dict:
    """
    处理单个entry，下载仓库和保存message
    
    Args:
        entry: JSON entry数据
        index: entry索引
        output_base_dir: 输出基础目录
        sha_length: commit SHA使用的长度（默认8位）
        
    Returns:
        Dict: 处理结果统计
    """
    # 使用repo名字和commit_sha前n位生成目录名
    repo_name = entry.get('repo', '')
    commit_sha = entry.get('commit_sha', '')
    owner = entry.get('owner', '')
    pair_dir_name = generate_pair_dir_name(repo_name, commit_sha, sha_length)
    pair_dir = output_base_dir / pair_dir_name
    zip_file = pair_dir / "repo.zip"
    message_file = pair_dir / "commit_message.txt"
    
    result = {
        'index': index,
        'pair_dir_name': pair_dir_name,
        'git_url': entry.get('git_url', ''),
        'commit_sha': commit_sha,
        'repo_downloaded': False,
        'message_saved': False,
        'success': False
    }
    
    # 仅当 repo.zip（非空）与 commit_message.txt 两者都存在时才跳过；仅创建了空目录或缺任一项则重新下载
    zip_ok = zip_file.exists() and zip_file.stat().st_size > 0
    msg_ok = message_file.exists()
    if zip_ok and msg_ok:
        logger.info(f"[{index+1}/1000] ⊙ 跳过（已存在）: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]}")
        result['repo_downloaded'] = True
        result['message_saved'] = True
        result['success'] = True
        return result
    if pair_dir.exists() and (not zip_ok or not msg_ok):
        missing = []
        if not zip_ok:
            missing.append("repo.zip" if not zip_file.exists() else "repo.zip(空)")
        if not msg_ok:
            missing.append("commit_message.txt")
        logger.info(f"[{index+1}/1000] 目录已存在但缺少 {', '.join(missing)}，将重新下载: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]}")
    
    # 下载仓库zip文件
    repo_success = download_repository_zip(
        owner,
        repo_name,
        commit_sha,
        zip_file
    )
    result['repo_downloaded'] = repo_success
    
    # 保存commit message
    message_success = save_commit_message(
        entry.get('message', ''),
        message_file
    )
    result['message_saved'] = message_success
    
    # 只有两者都成功才算成功
    result['success'] = repo_success and message_success
    
    if result['success']:
        logger.info(f"[{index+1}/1000] ✓ 成功处理: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]}")
    else:
        logger.warning(f"[{index+1}/1000] ✗ 处理失败: {entry.get('repo', '')} @ {entry.get('commit_sha', '')[:8]} "
                     f"(repo: {repo_success}, message: {message_success})")
    
    return result


def main():
    """主函数"""
    # 配置文件路径
    jsonl_file = Path(__file__).parent.parent / "ApacheCM-mini" / "python_test.jsonl"
    output_base_dir = Path(__file__).parent.parent / "downloaded_repositories"
    
    # 启动时验证 token（便于排查 401），仅使用脚本中的 token
    verify_github_token(GITHUB_TOKEN_FROM_CONFIG)
    
    # 检查输入文件是否存在
    if not jsonl_file.exists():
        logger.error(f"输入文件不存在: {jsonl_file}")
        return
    
    # 创建输出目录
    output_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_base_dir}")
    
    # 统计信息
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'repo_download_failed': 0,
        'message_save_failed': 0
    }
    
    # 读取并处理每个entry
    logger.info(f"开始处理文件: {jsonl_file}")
    start_time = time.time()
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    stats['total'] += 1
                    
                    # 处理entry
                    result = process_entry(entry, index, output_base_dir, sha_length=8)
                    
                    # 更新统计
                    if result['success']:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                        if not result['repo_downloaded']:
                            stats['repo_download_failed'] += 1
                        if not result['message_saved']:
                            stats['message_save_failed'] += 1
                    
                    # 每100个entry输出一次进度
                    if (index + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = (index + 1) / elapsed if elapsed > 0 else 0
                        remaining = (1000 - index - 1) / rate if rate > 0 else 0
                        logger.info(f"进度: {index+1}/1000 | "
                                  f"成功: {stats['success']} | "
                                  f"失败: {stats['failed']} | "
                                  f"预计剩余时间: {remaining/60:.1f}分钟")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"解析JSON失败 (行 {index+1}): {str(e)}")
                    stats['failed'] += 1
                except Exception as e:
                    logger.error(f"处理entry失败 (行 {index+1}): {str(e)}")
                    stats['failed'] += 1
    
    except Exception as e:
        logger.error(f"读取文件时出错: {str(e)}")
        return
    
    # 输出最终统计
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("下载完成！")
    logger.info(f"总条目数: {stats['total']}")
    logger.info(f"成功: {stats['success']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info(f"仓库下载失败: {stats['repo_download_failed']}")
    logger.info(f"消息保存失败: {stats['message_save_failed']}")
    logger.info(f"总耗时: {elapsed_time/3600:.2f}小时 ({elapsed_time/60:.1f}分钟)")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    if "--verify-token" in sys.argv:
        ok = verify_github_token(GITHUB_TOKEN_FROM_CONFIG)
        sys.exit(0 if ok else 1)
    main()
