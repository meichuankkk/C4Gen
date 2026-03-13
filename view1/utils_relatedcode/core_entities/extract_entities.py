import argparse
import asyncio
import json
import os
import traceback
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment variables from .env file
load_dotenv()

# Global OpenAI client, initialized lazily
client = None

CORE_ENTITY_PROMPT = """
<diff>
{DIFF_TEXT}
</diff>

Role:
You are an expert Code Analyzer for a Commit Message Generation system. Your specific task is **"Core Entity Identification"**.

**Context & Goal:**
We are building a "Dual-View Context Retrieval" system. The entities you identify will serve as anchors to retrieve relevant code context.
- **CRITICAL:** A single commit often modifies **MULTIPLE** distinct Core Entities (e.g., modifying both a Service class and a Utility function).
- **Goal:** You must identify **ALL** Core Entities that represent a primary logic change, feature addition, or bug fix. 
- **Constraint:** You must strictly filter out "Ripple Effects" (files changed solely to adapt to the core changes, such as updating a function call signature or importing a new module).

**Definition of "Core Entity":**
A **Core Entity** is a Class, Struct, Interface, or Function where the *root intent* of the commit resides.
- It is the **Subject** of the change (e.g., "Refactor X" or "Implement Y").
- It involves **Definition** or **Internal Logic** changes, not just **Usage/Call Site** changes.

**Selection Criteria (Filter Logic):**
1. **Definition vs. Call Site:**
   - If `func A` is modified to fix a bug or add logic, it IS a Core Entity.
   - If `func B` is modified *only* because it calls `func A` (e.g., passing a new argument), `func B` is NOT a Core Entity (it is Impact/Ripple Effect).
2. **Source vs. Test/Config:**
   - Prioritize Source Code. 
   - Ignore configuration files (e.g., pom.xml, package.json) unless the commit is exclusively about configurations.
   - Only select Test Code if the commit is *purely* adding/fixing tests.
3. **Logic vs. Trivial Changes:**
   - Ignore changes that only involve imports, spacing, formatting, comments, or simple getters/setters without business logic.

Instructions:
1. Analysis Step (File-by-File):
   - Scan the **ENTIRE** diff. 
   - Evaluate **EACH** modified file sequentially. Ask: "Is this specific modification a root cause (Core) or a reaction to another change (Impact)?"
   - For standalone functions (e.g., in Python, Go, C), note that there might not be an enclosing class.

2. Extraction Step:
   - Compile a comprehensive list of **ALL** identified Core Entities.
   - Output as a strictly valid JSON array.

Output Format Requirements:
- For a **Function/Method**: `{{"type": "function", "path": "<file_path>", "name": "<function_name>", "class_name": "<enclosing_class_name_or_null>"}}`
- For a **Class/Struct/Interface**: `{{"type": "class", "path": "<file_path>", "name": "<class_name>"}}`

Respond EXACTLY in the following format:

[start_of_analysis]
File-by-File Analysis:
- `src/AuthService.java`: `login` logic is fundamentally changed to support OAuth -> Core.
- `src/UserUtil.java`: `validate` regex is updated to fix a bug -> Core.
- `src/LoginController.java`: Updated merely to pass a new argument to `AuthService.login` -> Impact/Ignored.
- `scripts/utils.py`: Standalone function `parse_data` rewritten -> Core (class_name: null).
Conclusion: 3 Core Entities identified.
[end_of_analysis]

[start_of_core_entities]
[
  {{"type": "function", "path": "src/AuthService.java", "name": "login", "class_name": "AuthService"}},
  {{"type": "function", "path": "src/UserUtil.java", "name": "validate", "class_name": "UserUtil"}},
  {{"type": "function", "path": "scripts/utils.py", "name": "parse_data", "class_name": null}}
]
[end_of_core_entities]

Notes:
- **Exhaustiveness:** If the diff contains core logic changes across multiple distinct files, output ALL of them. Do not stop at the first one.
- **Null Safety:** If a function does not belong to a class (e.g., module-level functions), explicitly output `null` for `class_name`.
- **Class Name Inference:** Infer class names from the file content or file paths if not explicitly visible in the diff hunk.
"""

def get_client():
    """Initializes and returns the AsyncOpenAI client, reusing it if already created."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise ValueError(
                "OPENAI_API_KEY and/or OPENAI_BASE_URL environment variables not set. "
                "Please create a .env file and set them, or set them as environment variables."
            )
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def extract_entities_from_diff(diff_text: str, model_name: str) -> tuple[list | None, str | None]:
    """
    Calls the LLM to extract core entities from a diff text, with retry logic.
    """
    if not diff_text:
        return [], ""

    prompt = CORE_ENTITY_PROMPT.format(DIFF_TEXT=diff_text)
    try:
        aclient = get_client()
        response = await aclient.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0,
            top_p=0.9,
        )
        content = response.choices[0].message.content or ""
        
        start_tag = "[start_of_core_entities]"
        end_tag = "[end_of_core_entities]"
        
        start_index = content.find(start_tag)
        end_index = content.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            json_str = content[start_index + len(start_tag):end_index].strip()
            try:
                return json.loads(json_str), content
            except json.JSONDecodeError:
                # This error is not retryable, so we print and return None
                print(f"Error: Failed to decode JSON from LLM response.\nResponse snippet: {json_str[:200]}")
                return None, content
        else:
            # This error is not retryable
            print(f"Error: Could not find start/end tags in LLM response.\nResponse: {content}")
            return None, content

    except Exception as e:
        print(f"An exception occurred during LLM call: {e}. Retrying...")
        # Reraise the exception to allow tenacity to handle the retry
        raise

def make_progress(note: str = "processing"):
    """Creates a rich progress bar."""
    return Progress(
        TextColumn(f"{note} • [progress.percentage]{{task.percentage:>3.0f}}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("[bold cyan]{task.description}"),
    )

async def run_task(
    task_data: dict,
    p: Progress,
    task_id,
    output_file: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
):
    """Runs a single task, including the LLM call and writing the result."""
    instance_id = task_data["instance_id"]
    diff_text = task_data["diff"]
    core_entities = None

    async with semaphore:
        p.update(task_id, description=f"Processing {instance_id}")
        try:
            # We only need the entities, not the full response content
            core_entities, _ = await extract_entities_from_diff(diff_text, model_name)
        except Exception as e:
            # This will catch the final error after all retries have failed
            print(f"Failed to process {instance_id} after multiple retries: {e}")
            # core_entities remains None

    p.update(task_id, advance=1)

    if core_entities is not None:
        result = {
            "instance_id": instance_id,
            "core_entities": core_entities,
        }
        # Append the result to the output file.
        # This is a synchronous operation, but it's generally safe for appending lines.
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

async def run_all_tasks(tasks: list, concurrency: int, output_file: str, model_name: str):
    """Runs all tasks concurrently using a semaphore."""
    semaphore = asyncio.Semaphore(concurrency)
    with make_progress("Extracting Core Entities") as p:
        progress_task_id = p.add_task(description="Starting...", total=len(tasks))
        coros = [
            run_task(t, p, progress_task_id, output_file, semaphore, model_name)
            for t in tasks
        ]
        await asyncio.gather(*coros)

async def main(input_file: str, output_file: str, concurrency: int, model_name: str):
    """Main function to load tasks and orchestrate the run."""
    # Ensure the output file is cleared before starting
    with open(output_file, 'w') as f:
        pass # Clears the file

    try:
        tasks = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Prepare task data for run_task
                tasks.append({
                    "instance_id": f"{data['owner']}_{data['repo']}_{data['commit_sha'][:7]}",
                    "diff": data.get('diff')
                })
        print(f"Loaded {len(tasks)} tasks from {input_file}.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing {input_file}: {e}")
        traceback.print_exc()
        return
    
    await run_all_tasks(tasks, concurrency, output_file, model_name)
    print(f"\nProcessing complete. Results saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract core code entities from commit diffs using an LLM, in parallel."
    )
    parser.add_argument(
        '--input_file', 
        required=True, 
        help='Path to the input JSONL dataset file.'
    )
    parser.add_argument(
        '--output_file', 
        required=True, 
        help='Path to save the output JSON file.'
    )
    parser.add_argument(
        '--model', 
        required=True, 
        help='The name of the language model to use for extraction.'
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent tasks (default: 5).",
    )
    
    args = parser.parse_args()

    # The reference script uses asyncio.run, which is the modern way.
    asyncio.run(main(args.input_file, args.output_file, args.workers, args.model))
