import os
import json
import shutil
import subprocess
import argparse
from tqdm import tqdm


def run_command(command, working_dir, env=None):
    """Run a command and return (success, output_or_error)."""
    try:
        result = subprocess.run(
            command,
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_message = (
            f"Return code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        )
        return False, error_message
    except FileNotFoundError:
        return False, f"Command '{command[0]}' not found. Make sure it is in PATH."


def write_error_log(error_log_path, instance_id, task, error_message):
    """Append a formatted error message to the log file."""
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Error: {error_message}\n")
        f.write("-" * 40 + "\n")


def build_enre_py_command(repo_path, enable_cfg=False, enable_cg=False, compatible=False, builtins=None):
    """Build command for ENRE-py module invocation."""
    command = ["python", "-m", "enre", repo_path]

    if enable_cfg:
        command.append("--cfg")
    if enable_cg:
        command.append("--cg")
    if compatible:
        command.append("--compatible")
    if builtins:
        command.extend(["--builtins", builtins])

    return command


def process_dataset(
    dataset_path,
    repo_base_dir,
    enre_py_dir,
    output_dir,
    error_log_path,
    enable_cfg=False,
    enable_cg=False,
    compatible=False,
    builtins=None,
):
    """Process dataset entries and generate ENRE-py report for each commit."""
    os.makedirs(output_dir, exist_ok=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("ENRE-py Report Generation Error Log\n" + "=" * 40 + "\n")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Generating ENRE-py Reports"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                write_error_log(
                    error_log_path,
                    "N/A",
                    "JSON Parsing",
                    f"Malformed JSON line: {line.strip()}",
                )
                continue

            instance_id = f"{data['owner']}_{data['repo']}_{data['commit_sha'][:7]}"
            repo_name = data["repo"]
            commit_sha = data["commit_sha"]

            repo_path = os.path.join(repo_base_dir, repo_name)
            output_report_path = os.path.join(output_dir, f"{instance_id}_enre_report.json")

            if os.path.exists(output_report_path):
                continue

            if not os.path.isdir(repo_path):
                write_error_log(
                    error_log_path,
                    instance_id,
                    "File Check",
                    f"Repository path not found at {repo_path}",
                )
                continue

            checkout_command = ["git", "checkout", commit_sha, "--force"]
            success, output = run_command(checkout_command, working_dir=repo_path)
            if not success:
                write_error_log(error_log_path, instance_id, "Git Checkout", output)
                continue

            verify_success, verify_output = run_command(["git", "rev-parse", "HEAD"], working_dir=repo_path)
            if not verify_success:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "Git Checkout Verification",
                    f"Failed to run git rev-parse HEAD: {verify_output}",
                )
                continue

            current_commit = verify_output.strip()
            if current_commit != commit_sha:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "Git Checkout Verification",
                    f"Expected {commit_sha[:7]}, but HEAD is at {current_commit[:7]}.",
                )
                continue

            generated_report_name = f"{repo_name}-report-enre.json"
            generated_report_path = os.path.join(output_dir, generated_report_name)

            if os.path.exists(generated_report_path):
                os.remove(generated_report_path)

            command = build_enre_py_command(
                repo_path=repo_path,
                enable_cfg=enable_cfg,
                enable_cg=enable_cg,
                compatible=compatible,
                builtins=builtins,
            )

            run_env = os.environ.copy()
            current_pythonpath = run_env.get("PYTHONPATH", "")
            if current_pythonpath:
                run_env["PYTHONPATH"] = f"{enre_py_dir}:{current_pythonpath}"
            else:
                run_env["PYTHONPATH"] = enre_py_dir

            success, output = run_command(command, working_dir=output_dir, env=run_env)
            if not success:
                write_error_log(error_log_path, instance_id, "ENRE-py Execution", output)
                continue

            if not os.path.exists(generated_report_path):
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-py Post-Verification",
                    (
                        "Command succeeded, but expected output file was not found: "
                        f"{generated_report_path}"
                    ),
                )
                continue

            shutil.move(generated_report_path, output_report_path)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ENRE-py reports for dataset entries."
    )
    parser.add_argument(
        "--dataset_file",
        required=True,
        help="Path to the input JSONL dataset file.",
    )
    parser.add_argument(
        "--repo_dir",
        required=True,
        help="Base directory containing cloned git repositories.",
    )
    parser.add_argument(
        "--enre_py_dir",
        default="/data/data_public/riverbag/C4Gen_view1/ENRE-tools/ENRE-py",
        help="Path to ENRE-py source directory (used for PYTHONPATH).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated reports will be saved.",
    )
    parser.add_argument(
        "--error_log",
        default="/data/data_public/riverbag/C4Gen_view1/testINOut/enre_py_generator_errors.log",
        help="Path of error log file.",
    )
    parser.add_argument(
        "--cfg",
        action="store_true",
        help="Enable control flow analysis (--cfg).",
    )
    parser.add_argument(
        "--cg",
        action="store_true",
        help="Output call graph (--cg).",
    )
    parser.add_argument(
        "--compatible",
        action="store_true",
        help="Output compatible JSON format.",
    )
    parser.add_argument(
        "--builtins",
        default=None,
        help="Path to builtins module used by ENRE-py.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.enre_py_dir):
        print(f"Error: ENRE-py directory not found at {args.enre_py_dir}")
        return

    process_dataset(
        dataset_path=args.dataset_file,
        repo_base_dir=args.repo_dir,
        enre_py_dir=args.enre_py_dir,
        output_dir=args.output_dir,
        error_log_path=args.error_log,
        enable_cfg=args.cfg,
        enable_cg=args.cg,
        compatible=args.compatible,
        builtins=args.builtins,
    )


if __name__ == "__main__":
    main()
