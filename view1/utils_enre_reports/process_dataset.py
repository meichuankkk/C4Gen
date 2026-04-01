
import json
import os
import subprocess
from tqdm import tqdm

def process_dataset(jsonl_file, base_repo_path, log_file):
    # Get total number of lines for tqdm progress bar
    with open(jsonl_file, 'r') as f:
        total_lines = sum(1 for line in f)

    with open(jsonl_file, 'r') as f_in, open(log_file, 'w') as f_log:
        for line in tqdm(f_in, total=total_lines, desc="Processing dataset"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Handle nested structure if present
                if 'subset_entry' in data:
                    entry = data['subset_entry']
                    repo = entry.get('repo')
                    commit_sha = entry.get('commit_sha')
                else:
                    repo = data.get('repo')
                    commit_sha = data.get('commit_sha')

                if not repo or not commit_sha:
                    msg = f"Skipping line due to missing 'repo' or 'commit_sha': {line}"
                    f_log.write(msg + "\n")
                    tqdm.write(msg)
                    continue

                repo_path = os.path.join(base_repo_path, repo)

                if not os.path.isdir(repo_path):
                    msg = f"Repository directory not found for repo '{repo}': {repo_path}"
                    f_log.write(msg + "\n")
                    tqdm.write(msg)
                    continue

                command = ['git', 'fetch', 'origin', commit_sha]
                result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)

                if result.returncode == 0:
                    tqdm.write(f"Successfully fetched commit {commit_sha} for repo {repo}")
                else:
                    error_message = result.stderr.strip()
                    msg = f"Failed to fetch commit {commit_sha} for repo {repo}. Error: {error_message}"
                    f_log.write(msg + "\n")
                    tqdm.write(msg)

            except json.JSONDecodeError:
                msg = f"Skipping line due to JSON decoding error: {line}"
                f_log.write(msg + "\n")
                tqdm.write(msg)
            except Exception as e:
                msg = f"An unexpected error occurred for line {line}: {e}"
                f_log.write(msg + "\n")
                tqdm.write(msg)

if __name__ == '__main__':
    jsonl_file = '/root/autodl-tmp/view1/dataset/subset/core_entities_java_nonempty_enriched.jsonl'
    base_repo_path = '/root/autodl-tmp/view1/dataset_repo'
    log_file = '/root/autodl-tmp/view1/dataset/processing_errors.log'
    process_dataset(jsonl_file, base_repo_path, log_file)
    print("Dataset processing complete. Check processing_errors.log for any issues.")
