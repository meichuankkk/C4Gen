import os
import json
import subprocess
import argparse
from tqdm import tqdm

def run_command(command, working_dir):
    """Runs a command and returns a tuple of (success, output)."""
    # print(f"Running command: '{' '.join(command)}' in '{working_dir}'")
    try:
        result = subprocess.run(
            command,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Return code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        # print(f"Error running command: {' '.join(command)}\n{error_message}")
        return False, error_message
    except FileNotFoundError:
        error_message = f"Command '{command[0]}' not found. Make sure it's in your PATH."
        # print(f"Error: {error_message}")
        return False, error_message

def write_error_log(error_log_path, instance_id, task, error_message):
    """Appends a formatted error message to the log file."""
    with open(error_log_path, 'a', encoding='utf-8') as f:
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Error: {error_message}\n")
        f.write("-" * 40 + "\n")

def process_dataset(dataset_path, repo_base_dir, enre_jar_path, output_dir, error_log_path):
    """
    Processes the dataset to generate ENRE reports for each entry.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Clear or create the error log file at the beginning of the run
    with open(error_log_path, 'w', encoding='utf-8') as f:
        f.write("ENRE Report Generation Error Log\n" + "="*40 + "\n")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Generating ENRE Reports"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                write_error_log(error_log_path, "N/A", "JSON Parsing", f"Malformed JSON line: {line.strip()}")
                continue

            instance_id = f"{data['owner']}_{data['repo']}_{data['commit_sha'][:7]}"
            repo_name = data['repo']
            commit_sha = data['commit_sha']
            
            repo_path = os.path.join(repo_base_dir, repo_name)
            output_report_path = os.path.join(output_dir, f"{instance_id}_enre_report.json")

            if os.path.exists(output_report_path):
                continue

            if not os.path.isdir(repo_path):
                error_msg = f"Repository path not found at {repo_path}"
                write_error_log(error_log_path, instance_id, "File Check", error_msg)
                continue

            checkout_command = ['git', 'checkout', commit_sha, '--force']
            success, error = run_command(checkout_command, working_dir=repo_path)
            if not success:
                write_error_log(error_log_path, instance_id, "Git Checkout", error)
                continue

            # Verify checkout by comparing commit hashes
            verify_command = ['git', 'rev-parse', 'HEAD']
            verify_success, verify_output = run_command(verify_command, working_dir=repo_path)
            print("verify_output:", verify_output)
            print("verify_success:", verify_success)

            if not verify_success:
                # This happens if 'git rev-parse HEAD' fails
                error_msg = f"Failed to verify checkout for commit {commit_sha[:7]}. `git rev-parse HEAD` failed. Error: {verify_output}"
                print(error_msg)
                write_error_log(error_log_path, instance_id, "Git Checkout Verification", error_msg)
                continue

            current_commit = verify_output.strip()
            if current_commit == commit_sha:
                print(f"Successfully checked out commit {commit_sha[:7]} in {repo_name}.")
            else:
                # This happens if checkout resulted in a different commit
                error_msg = f"Failed to checkout commit {commit_sha[:7]}. Expected {commit_sha[:7]}, but HEAD is at {current_commit[:7]}."
                print(error_msg)
                write_error_log(error_log_path, instance_id, "Git Checkout Verification", error_msg)
                continue

            # Per user's instruction, the CWD will be the final output directory.
            # The -o parameter will be just the filename, and the source path will be absolute.
            output_filename = f"{instance_id}_enre_report"

            enre_command = [
                'java',
                '-Dfile.encoding=UTF-8',
                '-Xmx4G',
                '-jar',
                enre_jar_path,
                'java',
                repo_path,          # Absolute path to the repo to be analyzed
                repo_name,
                '-o',
                output_filename     # Simple filename, as CWD will be output_dir
            ]

            print("enre_command:", enre_command)
            print("--------------------------------------")
            # Run the command from the output directory to ensure the report is created in the correct place.
            success, output = run_command(enre_command, working_dir=output_dir)

            # Always print the output from the ENRE command for debugging
            print("--- ENRE Command Output ---")
            if output:
                print(output)
            print("---------------------------")

            if not success:
                write_error_log(error_log_path, instance_id, "ENRE Execution", output)
                continue

            # After successful execution, verify that the output file/directory was created
            if not os.path.exists(output_report_path):
                error_msg = f"ENRE command finished with exit code 0, but output path was not created: {output_report_path}"
                print(error_msg)
                write_error_log(error_log_path, instance_id, "ENRE Post-Verification", error_msg)
                continue

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate ENRE reports for a dataset.")
    parser.add_argument(
        '--dataset_file',
        required=True,
        help='Path to the input JSONL dataset file.'
    )
    parser.add_argument(
        '--repo_dir',
        required=True,
        help='Base directory containing the cloned git repositories.'
    )
    parser.add_argument(
        '--enre_jar',
        required=True,
        help='Path to the ENRE java jar file.'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory to save the generated ENRE reports.'
    )
    parser.add_argument(
        '--error_log',
        default='/data/data_public/riverbag/C4Gen/testINOut/enre_generator_errors.log',
        help='Path to save the error log file (default: enre_generator_errors.log).'
    )
    args = parser.parse_args()

    try:
        import tqdm
    except ImportError:
        print("'tqdm' library not found. Please install it using: pip install tqdm")
        return

    process_dataset(args.dataset_file, args.repo_dir, args.enre_jar, args.output_dir, args.error_log)

if __name__ == '__main__':
    main()
