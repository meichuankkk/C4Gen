
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
            try:
                data = json.loads(line)
                repo = data.get('repo')
                commit_sha = data.get('commit_sha')

                if not repo or not commit_sha:
                    f_log.write(f"Skipping line due to missing 'repo' or 'commit_sha': {line.strip()}\n")
                    continue

                repo_path = os.path.join(base_repo_path, repo)

                if not os.path.isdir(repo_path):
                    f_log.write(f"Repository directory not found for repo '{repo}': {repo_path}\n")
                    continue

                command = ['git', 'fetch', 'origin', commit_sha]
                result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True)

                if result.returncode == 0:
                    tqdm.write(f"Successfully fetched commit {commit_sha} for repo {repo}")
                else:
                    error_message = result.stderr.strip()
                    f_log.write(f"Failed to fetch commit {commit_sha} for repo {repo}. Error: {error_message}\n")

            except json.JSONDecodeError:
                f_log.write(f"Skipping line due to JSON decoding error: {line.strip()}\n")
            except Exception as e:
                f_log.write(f"An unexpected error occurred for line {line.strip()}: {e}\n")

if __name__ == '__main__':
    jsonl_file = '/data/data_public/riverbag/C4Gen/dataset/test/beam.jsonl'
    base_repo_path = '/data/data_public/riverbag/C4Gen/dataset_repo'
    log_file = '/data/data_public/riverbag/C4Gen/testINOut/processing_errors.log'
    process_dataset(jsonl_file, base_repo_path, log_file)
    print("Dataset processing complete. Check processing_errors.log for any issues.")
