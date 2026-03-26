import json
import argparse

try:
    from fire import Fire
except ModuleNotFoundError:
    Fire = None


SIMILAR_PROMPT = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit. First, a similar commit example (including both code diff and commit message) is provided for reference. Then, you will be given a code diff which is your task, and you need to write a commit message for it.

## Input Format:

=== START OF SIMILAR COMMIT ===

--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

--- START OF COMMIT MESSAGE ---
A commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
--- END OF COMMIT MESSAGE ---

=== END OF SIMILAR COMMIT ===

=== START OF YOUR TASK ===

--- START OF CODE DIFF ---
(Code changes in .diff format)
--- END OF CODE DIFF ---

=== END OF YOUR TASK ===

## Output Format:

A concise commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
E.g. <message>feat(server): Add new API endpoint for user registration</message>
"""


def load_similar_by_query_sha(similar_file_path: str) -> dict:
    """Load similar retrieval results indexed by query-sha."""
    similar_map = {}
    duplicate_sha_count = 0

    with open(similar_file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON in similar file at line {line_no}.")
                continue

            query_sha = item.get("query-sha")
            if not query_sha:
                print(f"Warning: Missing query-sha at line {line_no}, skipping.")
                continue

            if query_sha in similar_map:
                duplicate_sha_count += 1
                # Keep first occurrence for stable mapping.
                continue

            similar_map[query_sha] = item

    if duplicate_sha_count:
        print(f"Warning: Found {duplicate_sha_count} duplicate query-sha entries; kept first occurrence.")

    return similar_map


def build_user_prompt(similar_diff: str, similar_message: str, target_diff: str) -> str:
    return (
        f"{SIMILAR_PROMPT}\n\n"
        "=== START OF SIMILAR COMMIT ===\n\n"
        "--- START OF CODE DIFF ---\n"
        f"{similar_diff}\n"
        "--- END OF CODE DIFF ---\n\n"
        "--- START OF COMMIT MESSAGE ---\n"
        f"<message>{similar_message}</message>\n"
        "--- END OF COMMIT MESSAGE ---\n\n"
        "=== END OF SIMILAR COMMIT ===\n\n"
        "=== START OF YOUR TASK ===\n\n"
        "--- START OF CODE DIFF ---\n"
        f"{target_diff}\n"
        "--- END OF CODE DIFF ---\n\n"
        "=== END OF YOUR TASK ==="
    )


def create_tasks(
    core_entities_file: str = "C4Gen/view1/dataset/dpsk_chat_core_entities_5000/core_entities_java_nonempty_enriched.jsonl",
    similar_file: str = "C4Gen/view1/dataset/similar_diff_message/results_Java_BM25_dense_5_5_Jina.jsonl",
    tasks_path: str = "C4Gen/view1/utils_generate_commit_message/tasks_similar_diff_message.jsonl",
):
    """
    Generate task JSONL using:
    - target diff: subset_entry.diff
    - alignment key: subset_entry.commit_sha == query-sha
    - similar pair: retrieve-diff + retrieve-message
    """
    print(f"Loading similar results from: {similar_file}")
    similar_map = load_similar_by_query_sha(similar_file)
    print(f"Loaded {len(similar_map)} unique query-sha mappings.")

    total = 0
    generated = 0
    skipped_missing_fields = 0
    skipped_unmatched_sha = 0

    with open(core_entities_file, "r", encoding="utf-8") as f_in, open(
        tasks_path, "w", encoding="utf-8"
    ) as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON in core file at line {line_no}.")
                skipped_missing_fields += 1
                continue

            subset = item.get("subset_entry") or {}
            commit_sha = subset.get("commit_sha")
            target_diff = subset.get("diff")
            label_message = subset.get("message")

            if not commit_sha or not target_diff:
                skipped_missing_fields += 1
                print(f"Warning: Missing commit_sha or diff at line {line_no}, skipping.")
                continue

            similar_item = similar_map.get(commit_sha)
            if not similar_item:
                skipped_unmatched_sha += 1
                continue

            similar_diff = similar_item.get("retrieve-diff")
            similar_message = similar_item.get("retrieve-message")
            if not similar_diff or not similar_message:
                skipped_missing_fields += 1
                print(f"Warning: Missing retrieve-diff or retrieve-message for sha {commit_sha}, skipping.")
                continue

            task_id = item.get("instance_id")
            if not task_id:
                owner = subset.get("owner", "unknown_owner")
                repo = subset.get("repo", "unknown_repo")
                task_id = f"{owner}_{repo}_{commit_sha[:7]}"

            user_prompt = build_user_prompt(similar_diff, similar_message, target_diff)

            task_obj = {
                "task_id": task_id,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
                "label": label_message,
                "meta": {
                    "commit_sha": commit_sha,
                    "retrieve_sha": similar_item.get("retrieve-sha"),
                },
            }

            f_out.write(json.dumps(task_obj) + "\n")
            generated += 1

    print("\nTask generation finished.")
    print(f"Input rows: {total}")
    print(f"Generated tasks: {generated}")
    print(f"Skipped (missing fields): {skipped_missing_fields}")
    print(f"Skipped (no matched query-sha): {skipped_unmatched_sha}")
    print(f"Output: {tasks_path}")


if __name__ == "__main__":
    if Fire is not None:
        Fire(create_tasks)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--core_entities_file",
            default="C4Gen/view1/dataset/dpsk_chat_core_entities_5000/core_entities_java_nonempty_enriched.jsonl",
        )
        parser.add_argument(
            "--similar_file",
            default="C4Gen/view1/dataset/similar_diff_message/results_Java_BM25_dense_5_5_Jina.jsonl",
        )
        parser.add_argument(
            "--tasks_path",
            default="C4Gen/view1/utils_generate_commit_message/tasks_similar_diff_message.jsonl",
        )
        cli_args = parser.parse_args()
        create_tasks(
            core_entities_file=cli_args.core_entities_file,
            similar_file=cli_args.similar_file,
            tasks_path=cli_args.tasks_path,
        )
