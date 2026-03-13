import json
import os

def find_missing_and_empty_entities(
    subset_file: str,
    core_entities_file: str,
    output_file: str
):
    """
    Identifies missing or empty core entities and extracts corresponding data from the subset file.

    Args:
        subset_file: Path to the java_subset.jsonl file.
        core_entities_file: Path to the 10000_core_entities_output.jsonl file.
        output_file: Path to save the extracted missing/empty data.
    """
    print(f"Loading core entities from {core_entities_file}...")
    core_entities_map = {}
    try:
        with open(core_entities_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    instance_id = item.get('instance_id')
                    core_entities = item.get('core_entities')
                    if instance_id:
                        core_entities_map[instance_id] = core_entities
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {core_entities_file}")
    except FileNotFoundError:
        print(f"Error: Core entities file not found at {core_entities_file}")
        return

    print(f"Loaded {len(core_entities_map)} core entity records.")

    missing_or_empty_count = 0
    extracted_data = []

    print(f"Processing subset file {subset_file}...")
    try:
        with open(subset_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    # Construct instance_id as described
                    # "instance_id": f"{data['owner']}_{data['repo']}_{data['commit_sha'][:7]}"
                    owner = data.get('owner', '')
                    repo = data.get('repo', '')
                    commit_sha = data.get('commit_sha', '')
                    
                    if not (owner and repo and commit_sha):
                         print(f"Warning: Missing required fields for ID generation in line: {line.strip()[:100]}...")
                         continue

                    instance_id = f"{owner}_{repo}_{commit_sha[:7]}"

                    # Check if instance_id is missing from core_entities_map OR if its core_entities list is empty
                    if instance_id not in core_entities_map:
                        # Case 1: Completely missing from output file
                        missing_or_empty_count += 1
                        extracted_data.append(data)
                        # print(f"Missing: {instance_id}")
                    elif not core_entities_map[instance_id]:
                        # Case 2: Present but core_entities is empty (None or [])
                        missing_or_empty_count += 1
                        extracted_data.append(data)
                        # print(f"Empty Entities: {instance_id}")

                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {subset_file}")

    except FileNotFoundError:
        print(f"Error: Subset file not found at {subset_file}")
        return

    print(f"Found {missing_or_empty_count} missing or empty entity records.")

    if extracted_data:
        print(f"Writing extracted data to {output_file}...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for item in extracted_data:
                    f_out.write(json.dumps(item) + '\n')
            print("Done.")
        except IOError as e:
            print(f"Error writing to output file: {e}")
    else:
        print("No missing or empty data found.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find missing or empty core entities.")
    parser.add_argument("--subset_file", type=str, default="/data/data_public/riverbag/C4Gen/dataset/java_subset.jsonl", help="Path to the subset JSONL file.")
    parser.add_argument("--core_entities_file", type=str, default="/data/data_public/riverbag/C4Gen/dataset/10000_core_entities_output.jsonl", help="Path to the core entities output JSONL file.")
    parser.add_argument("--output_file", type=str, default="/data/data_public/riverbag/C4Gen/dataset/missing_entities.jsonl", help="Path to save the extracted missing data.")

    args = parser.parse_args()

    find_missing_and_empty_entities(args.subset_file, args.core_entities_file, args.output_file)
