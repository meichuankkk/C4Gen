
import os
import subprocess
import re
import time
from tqdm import tqdm

def retry_fetches(log_file, base_repo_path):
    failed_items = []
    
    # Regex to extract commit and repo
    # Support both "Error:" (original log) and "Last Error:" (retry log)
    pattern = re.compile(r"Failed to fetch commit ([a-f0-9]+) for repo ([a-zA-Z0-9_\-\.]+)\. (?:Last )?Error:")

    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    print(f"Reading log file: {log_file}")
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                commit_sha = match.group(1)
                repo = match.group(2)
                failed_items.append((repo, commit_sha))
    
    # Remove duplicates
    failed_items = list(set(failed_items))
    
    print(f"Log file: {log_file}")
    print(f"Found {len(failed_items)} unique failed fetches.")
    
    # New log file for this run
    retry_log_file = log_file.replace('.log', '_retry.log')
    print(f"Logging retry results to: {retry_log_file}")

    if not failed_items:
        return

    success_count = 0
    still_failed = []

    with open(retry_log_file, 'w') as f_retry_log:
        for repo, commit_sha in tqdm(failed_items, desc="Retrying fetches"):
            repo_path = os.path.join(base_repo_path, repo)
            
            if not os.path.isdir(repo_path):
                msg = f"Repository directory not found for repo '{repo}': {repo_path}"
                tqdm.write(msg)
                f_retry_log.write(msg + "\n")
                still_failed.append((repo, commit_sha, "Repo dir not found"))
                continue

            command = ['git', 'fetch', 'origin', commit_sha]
            
            fetched = False
            max_retries = 3
            last_error = ""

            for attempt in range(max_retries):
                tqdm.write(f"  Attempt {attempt+1}/{max_retries}: Fetching {repo} -> {commit_sha[:7]}...")
                try:
                    result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        msg = f"  ✅ Success: {commit_sha} for {repo}"
                        tqdm.write(msg)
                        f_retry_log.write(msg + "\n")
                        fetched = True
                        break
                    else:
                        last_error = result.stderr.strip()
                        tqdm.write(f"  ❌ Failed: {last_error[:100]}...")
                        if attempt < max_retries - 1:
                            time.sleep(2) # Wait before retry
                except subprocess.TimeoutExpired:
                    last_error = "Operation timed out after 60s"
                    tqdm.write(f"  ⚠️ Timeout: Fetch took too long.")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            
            if fetched:
                success_count += 1
            else:
                msg = f"Failed to fetch commit {commit_sha} for repo {repo} after {max_retries} attempts."
                tqdm.write(msg)
                f_retry_log.write(msg + "\n")
                tqdm.write(f"Last Error: {last_error}")
                f_retry_log.write(f"Last Error: {last_error}\n")
                still_failed.append((repo, commit_sha, last_error))

    print(f"\nRetry summary:")
    print(f"Total: {len(failed_items)}")
    print(f"Success: {success_count}")
    print(f"Still Failed: {len(still_failed)}")
    
    if still_failed:
        print("\nStill failing items:")
        with open(log_file + ".retry_failed.log", 'w') as f_out:
             for repo, commit_sha, error in still_failed:
                msg = f"Failed to fetch commit {commit_sha} for repo {repo}. Last Error: {error}"
                print(msg)
                f_out.write(msg + "\n")
        print(f"Failed items written to {log_file}.retry_failed.log")

if __name__ == '__main__':
    # Check if a retry failure log exists and use it if available to continue where we left off
    retry_fail_log = '/root/autodl-tmp/view1/dataset/processing_errors.log.retry_failed.log'
    if os.path.exists(retry_fail_log):
        log_file = retry_fail_log
        print(f"Detected previous retry failure log. Resuming from: {log_file}")
    else:
        log_file = '/root/autodl-tmp/view1/dataset/processing_errors.log'

    base_repo_path = '/root/autodl-tmp/view1/dataset_repo'
    retry_fetches(log_file, base_repo_path)
