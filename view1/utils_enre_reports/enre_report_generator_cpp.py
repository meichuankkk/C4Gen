import os
import json
import shutil
import subprocess
import argparse
from tqdm import tqdm


def run_command(command, working_dir):
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


def build_enre_cpp_command(
    enre_jar_path,
    repo_path,
    project_name,
    java_xmx,
    java_xms=None,
    program_environments=None,
    extra_dirs=None,
):
    """Build command following ENRE-CPP usage: directory projectName [-p ...] [-d ...]."""
    command = [
        "java",
        "-Dfile.encoding=UTF-8",
        f"-Xmx{java_xmx}",
    ]

    if java_xms:
        command.append(f"-Xms{java_xms}")

    command.extend([
        "-jar",
        enre_jar_path,
        repo_path,
        project_name,
    ])

    for env in (program_environments or []):
        command.extend(["-p", env])

    for d in (extra_dirs or []):
        command.extend(["-d", d])

    return command


def find_generated_json(tmp_dir):
    """Locate generated json report(s) in ENRE-CPP temporary output dir."""
    candidates = []
    for root, _, files in os.walk(tmp_dir):
        for name in files:
            if name.lower().endswith(".json"):
                full_path = os.path.join(root, name)
                candidates.append((os.path.getmtime(full_path), full_path))

    if not candidates:
        return None

    # Prefer the latest json when multiple files are generated.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def process_dataset(
    dataset_path,
    repo_base_dir,
    enre_jar_path,
    output_dir,
    error_log_path,
    java_xmx,
    java_xms=None,
    program_environments=None,
    extra_dirs=None,
):
    """Process dataset entries and generate ENRE-CPP report for each commit."""
    os.makedirs(output_dir, exist_ok=True)

    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("ENRE-CPP Report Generation Error Log\n" + "=" * 40 + "\n")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Generating ENRE-CPP Reports"):
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
                # On Windows, checkout may return non-zero because of unlink warnings
                # even when HEAD already moved to the target commit.
                verify_success, verify_output = run_command(["git", "rev-parse", "HEAD"], working_dir=repo_path)
                if verify_success and verify_output.strip() == commit_sha:
                    pass
                else:
                    write_error_log(error_log_path, instance_id, "Git Checkout", output)
                    continue

            verify_command = ["git", "rev-parse", "HEAD"]
            verify_success, verify_output = run_command(verify_command, working_dir=repo_path)
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
                    (
                        f"Expected {commit_sha[:7]}, but HEAD is at {current_commit[:7]}."
                    ),
                )
                continue

            tmp_output_dir = os.path.join(output_dir, f".tmp_enre_cpp_{instance_id}")
            os.makedirs(tmp_output_dir, exist_ok=True)

            command = build_enre_cpp_command(
                enre_jar_path=enre_jar_path,
                repo_path=repo_path,
                project_name=repo_name,
                java_xmx=java_xmx,
                java_xms=java_xms,
                program_environments=program_environments,
                extra_dirs=extra_dirs,
            )

            success, output = run_command(command, working_dir=tmp_output_dir)
            if not success:
                write_error_log(error_log_path, instance_id, "ENRE-CPP Execution", output)
                continue

            generated_json = find_generated_json(tmp_output_dir)
            if not generated_json:
                write_error_log(
                    error_log_path,
                    instance_id,
                    "ENRE-CPP Post-Verification",
                    "Command succeeded, but no JSON report was found in temporary output directory.",
                )
                continue

            shutil.move(generated_json, output_report_path)
            shutil.rmtree(tmp_output_dir, ignore_errors=True)

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ENRE-CPP reports for dataset entries."
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
        "--enre_jar",
        default="/data/data_public/riverbag/C4Gen_view1/ENRE-tools/ENRE-cpp/enre_cpp.jar",
        help="Path to ENRE-CPP jar file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated reports will be saved.",
    )
    parser.add_argument(
        "--error_log",
        default="/data/data_public/riverbag/C4Gen_view1/testINOut/enre_cpp_generator_errors.log",
        help="Path of error log file.",
    )
    parser.add_argument(
        "--java_xmx",
        default="4G",
        help="Java max heap, e.g. 4G or 64g.",
    )
    parser.add_argument(
        "--java_xms",
        default=None,
        help="Java initial heap, e.g. 8G (optional).",
    )
    parser.add_argument(
        "-p",
        "--program_environment",
        action="append",
        default=[],
        help="Program environment path, can be passed multiple times.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="extra_dirs",
        action="append",
        default=[],
        help="Additional directory for analysis, can be passed multiple times.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.enre_jar):
        print(f"Error: ENRE-CPP jar not found at {args.enre_jar}")
        return

    process_dataset(
        dataset_path=args.dataset_file,
        repo_base_dir=args.repo_dir,
        enre_jar_path=args.enre_jar,
        output_dir=args.output_dir,
        error_log_path=args.error_log,
        java_xmx=args.java_xmx,
        java_xms=args.java_xms,
        program_environments=args.program_environment,
        extra_dirs=args.extra_dirs,
    )


if __name__ == "__main__":
    main()
