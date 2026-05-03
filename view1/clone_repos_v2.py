import json
import os
import subprocess
import glob

def clone_repositories():
    # 获取脚本所在的目录 (view1)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 所有的路径都基于这个目录
    target_dir = os.path.join(base_dir, 'dataset_repo')
    dataset_dir = os.path.join(base_dir, 'dataset', 'dpsk_chat_core_entities_5000')
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Use glob to find C++ enriched core entities file.
    search_pattern = os.path.join(dataset_dir, 'core_entities_cpp_nonempty_enriched.jsonl')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"Error: C++ core entities file not found under {dataset_dir}")
        return
    
    print(f"Found input files: {files}")

    processed_repos = set()

    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'subset_entry' in data:
                            entry = data['subset_entry']
                        else:
                            entry = data
                            
                        git_url = entry.get('git_url')
                        repo_name = entry.get('repo')
                        
                        if not git_url or not repo_name:
                            continue

                        repo_key = (git_url, repo_name)
                        if repo_key in processed_repos:
                            continue
                        processed_repos.add(repo_key)
                        
                        repo_path = os.path.join(target_dir, repo_name)
                        
                        if os.path.exists(repo_path):
                            # Check if it has .git folder
                            if os.path.isdir(os.path.join(repo_path, '.git')):
                                print(f"Skipping {repo_name} (already exists)")
                                continue
                            else:
                                print(f"Warning: {repo_name} exists but may incomplete. Re-cloning...")
                                subprocess.run(['rm', '-rf', repo_path])

                        print(f"Cloning {repo_name} from {git_url}...")
                        subprocess.run(['git', 'clone', '--depth', '1', git_url, repo_path], check=False)
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line: {e}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    clone_repositories()
