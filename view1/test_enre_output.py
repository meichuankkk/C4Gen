import os
import subprocess
import argparse

def run_enre_and_list_files(repo_path, output_dir, enre_jar_path, repo_name, output_filename):
    # 1. 构造ENRE命令
    enre_command = [
        'java',
        '-Dfile.encoding=UTF-8',
        '-Xmx16G',
        '-jar',
        enre_jar_path,
        'java',
        repo_path,
        repo_name,
        '-o',
        output_filename
    ]
    print("Running ENRE command:")
    print(" ".join(enre_command))
    # 2. 执行ENRE命令
    result = subprocess.run(enre_command, cwd=output_dir, capture_output=True, text=True)
    print("\n--- ENRE STDOUT ---\n", result.stdout)
    print("\n--- ENRE STDERR ---\n", result.stderr)
    # 3. 列出output_dir下所有文件
    print("\nFiles in output_dir:")
    for f in os.listdir(output_dir):
        print("  ", f)
    # 4. 列出enre-out子目录下所有文件
    enre_out_dir = os.path.join(output_dir, f"{repo_name}-enre-out")
    if os.path.exists(enre_out_dir):
        print(f"\nFiles in {enre_out_dir}:")
        for f in os.listdir(enre_out_dir):
            print("  ", f)
    else:
        print(f"\n{enre_out_dir} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', required=True, help='源码仓库绝对路径')
    parser.add_argument('--output_dir', required=True, help='ENRE输出目录')
    parser.add_argument('--enre_jar', required=True, help='ENRE jar包路径')
    parser.add_argument('--repo_name', required=True, help='项目名')
    parser.add_argument('--output_filename', required=True, help='ENRE输出文件名（不带后缀）')
    args = parser.parse_args()
    run_enre_and_list_files(
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        enre_jar_path=args.enre_jar,
        repo_name=args.repo_name,
        output_filename=args.output_filename
    )