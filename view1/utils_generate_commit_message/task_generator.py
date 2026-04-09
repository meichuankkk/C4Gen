'''
This script generates structured tasks for a commit message generation model.

It combines three sources of information:
1. A dataset containing the raw git diff and original commit message.
2. A report containing the core entities involved in the change.
3. A file containing relevant code context for those entities.

The output is a JSONL file where each line is a task object ready for a model,
including a system prompt, a user prompt with the combined data, and the original
commit message as a label.
'''

import json
from fire import Fire

# The system prompt defines the role and instructions for the model.
SYSTEM_PROMPT_RELATED_CODE = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit, the core entities in those code changes, and related code context.

## Input Format:

--- START OF CODE DIFF ---
{QUERY_DIFF}
--- END OF CODE DIFF ---

--- START OF CORE ENTITIES ---
{CORE_ENTITIES}
--- END OF CORE ENTITIES ---

--- START OF RELEVANT CODE CONTEXT ---
{RELEVANT_CODE}
--- END OF RELEVANT CODE CONTEXT ---

## Output Format:

A concise commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
E.g. <message>feat(server): Add new API endpoint for user registration</message>
"""

SYSTEM_PROMPT_DIRECT = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit.

## Input Format:

--- START OF CODE DIFF ---
{QUERY_DIFF}
--- END OF CODE DIFF ---

## Output Format:

A concise commit message describing the code changes, wrapped in <message> </message> tags.
E.g. <message>Fixed a bug in the user authentication flow</message>
E.g. <message>feat(server): Add new API endpoint for user registration</message>
"""

def load_jsonl_to_dict(file_path: str) -> dict:
    """Loads a JSONL file into a dictionary keyed by 'instance_id' for quick lookup."""
    data_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'instance_id' in item:
                        data_dict[item['instance_id']] = item
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Returning empty dictionary.")
    return data_dict

def create_tasks(
    dataset_path: str,
    tasks_path: str,
    core_entities_path: str = None,
    relevant_code_path: str = None,
    prompt_type: str = "related_code",
):
    """
    Generates a JSONL task file for commit message generation.

    Args:
        dataset_path: Path to the main dataset JSONL file (contains diff, message, etc.).
        tasks_path: Path to write the output tasks.jsonl file.
        core_entities_path: Path to the JSONL file with core entities (Required for 'related_code').
        relevant_code_path: Path to the JSONL file with all related code (Required for 'related_code').
        prompt_type: Type of prompt to use ('direct' or 'related_code').
    """
    if prompt_type not in ["direct", "related_code"]:
        raise ValueError("prompt_type must be either 'direct' or 'related_code'")

    core_entities_data = {}
    relevant_code_data = {}

    if prompt_type == "related_code":
        if not core_entities_path or not relevant_code_path:
            raise ValueError("For 'related_code' prompt type, both 'core_entities_path' and 'relevant_code_path' are required.")
            
        print(f"Loading core entities... (Prompt Type: {prompt_type})")
        core_entities_data = load_jsonl_to_dict(core_entities_path)
        print(f"Loaded {len(core_entities_data)} core entity records.")

        print("Loading relevant code...")
        relevant_code_data = load_jsonl_to_dict(relevant_code_path)
        print(f"Loaded {len(relevant_code_data)} relevant code records.")
    else:
        print(f"Prompt Type: {prompt_type} - Skipping loading of core entities and relevant code.")

    tasks = []
    print(f"Processing dataset from {dataset_path}...")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f_dataset:
            for line in f_dataset:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in dataset: {line.strip()}")
                    continue

                # Construct the instance_id to link data sources
                instance_id = f"{data.get('owner')}_{data.get('repo')}_{data.get('commit_sha', '')[:7]}"

                if prompt_type == "related_code":
                    # Retrieve corresponding data using the instance_id
                    core_entities = core_entities_data.get(instance_id, {}).get('core_entities')
                    relevant_code = relevant_code_data.get(instance_id, {}).get('retrieved_code')

                    # Skip if any of the required data is missing
                    if not all([data.get('diff'), core_entities, relevant_code]):
                        print(f"Warning: Skipping instance {instance_id} due to missing diff, core entities, or relevant code.")
                        continue

                    # Format the context data as JSON strings
                    core_entities_json = json.dumps(core_entities, indent=2)
                    relevant_code_text = json.dumps(relevant_code, indent=2)

                    # Construct the user prompt using the provided template
                    user_prompt = (
                        f"--- START OF CODE DIFF ---\n{data['diff']}\n--- END OF CODE DIFF ---\n\n"
                        f"--- START OF CORE ENTITIES ---\n{core_entities_json}\n--- END OF CORE ENTITIES ---\n\n"
                        f"--- START OF RELEVANT CODE CONTEXT ---\n{relevant_code_text}\n--- END OF RELEVANT CODE CONTEXT ---"
                    )
                    system_prompt_content = SYSTEM_PROMPT_RELATED_CODE
                
                else: # prompt_type == "direct"
                    if not data.get('diff'):
                        print(f"Warning: Skipping instance {instance_id} due to missing diff.")
                        continue

                    user_prompt = (
                        f"--- START OF CODE DIFF ---\n{data['diff']}\n--- END OF CODE DIFF ---"
                    )
                    system_prompt_content = SYSTEM_PROMPT_DIRECT

                # Assemble the final task object
                messages = [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": user_prompt},
                ]

                tasks.append(
                    {
                        "task_id": instance_id,
                        "messages": messages,
                        "label": data.get('message'),
                    }
                )

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}. Cannot proceed.")
        return

    # Write all generated tasks to the output file
    try:
        with open(tasks_path, 'w', encoding='utf-8') as f_tasks:
            for task in tasks:
                f_tasks.write(json.dumps(task) + '\n')
        print(f"\nSuccessfully generated {len(tasks)} tasks at: {tasks_path}")
    except IOError as e:
        print(f"\nError writing tasks to {tasks_path}: {e}")

if __name__ == "__main__":
    Fire(create_tasks)
