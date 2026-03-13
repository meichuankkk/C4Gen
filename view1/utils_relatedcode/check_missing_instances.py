import json
import argparse
import os

def load_instance_ids(file_path):
    ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'instance_id' in data:
                        ids.add(data['instance_id'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return ids

def check_missing(commit_map_file, output_file, error_report_file):
    input_ids = load_instance_ids(commit_map_file)
    output_ids = load_instance_ids(output_file)
    error_ids = load_instance_ids(error_report_file)

    processed_ids = output_ids.union(error_ids)
    missing_ids = input_ids - processed_ids

    print(f"Total Input IDs: {len(input_ids)}")
    print(f"Total Output IDs: {len(output_ids)}")
    print(f"Total Error IDs: {len(error_ids)}")
    print(f"Total Processed IDs: {len(processed_ids)}")
    print(f"Missing IDs: {len(missing_ids)}")

    if missing_ids:
        print("\nMissing Instance IDs:")
        for i_id in sorted(list(missing_ids)):
            print(i_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for missing instance IDs.")
    parser.add_argument("--commit_map_file", required=True, help="Path to the commit map file (input).")
    parser.add_argument("--output_file", required=True, help="Path to the output file.")
    parser.add_argument("--error_report_file", required=True, help="Path to the error report file.")

    args = parser.parse_args()

    check_missing(args.commit_map_file, args.output_file, args.error_report_file)
