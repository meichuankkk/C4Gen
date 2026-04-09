import argparse
import asyncio
import json
import os
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

def get_client():
    """Initializes and returns the OpenAI client, reusing it if already created."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        print(f"base_url: {base_url}")
        #input("Press Enter to continue...")
        if not api_key or not base_url:
            raise ValueError(
                "OPENAI_API_KEY and/or OPENAI_BASE_URL environment variables not set. "
                "Please create a .env file and set them, or set them as environment variables."
            )
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def call_llm(messages: list, model_name: str):
    """Calls the LLM with exponential backoff retry logic."""
    try:
        aclient = get_client()
        response = await aclient.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,  # Commit messages are relatively short
            temperature=0
        )
        return response
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise

def make_progress(note: str = "processing"):
    """Creates a rich progress bar."""
    return Progress(
        TextColumn(f"{note} • [progress.percentage]{{task.percentage:>3.0f}}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

async def run_task(
    task: dict,
    p: Progress,
    task_id,
    output_file: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
):
    """Runs a single task, including the LLM call and writing the result."""
    response_obj = None

    async with semaphore:
        try:
            response_obj = await call_llm(task["messages"], model_name)
        except Exception as e:
            print(f"Failed to get response for task_id {task['task_id']}: {e}")
            response_obj = None

    p.update(task_id, advance=1)

    if response_obj is None:
        result = {
            "task_id": task["task_id"],
            "label": task.get("label"),
            "model": model_name,
            "response": None,  # Indicate failure
        }
    else:
        response_content = response_obj.choices[0].message.content
        result = {
            "task_id": task["task_id"],
            "label": task.get("label"),
            "model": model_name,
            "response": response_content,
        }

    # Append the result to the output file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    return result

async def run_tasks(tasks: list, concurrency: int, output_file: str, model_name: str):
    """Runs all tasks concurrently using a semaphore."""
    semaphore = asyncio.Semaphore(concurrency)
    with make_progress("Generating Commits") as p:
        progress_task_id = p.add_task("Querying LLM", total=len(tasks))
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
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f]
        print(f"Loaded {len(tasks)} tasks from {input_file}.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    
    await run_tasks(tasks, concurrency, output_file, model_name)
    print(f"\nProcessing complete. Results saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process commit message generation tasks from a JSONL file using an OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSONL file containing tasks (e.g., tasks.jsonl).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSONL file to save responses.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent tasks (default: 10).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3.2-128k",
        help="The name of the model to use (e.g., deepseek-chat).",
    )
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.workers, args.model))
