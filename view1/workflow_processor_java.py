
import os
import sys
import json
import argparse
import subprocess
from itertools import chain
from tqdm import tqdm

# Add the parent directory to sys.path to allow imports from java_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from java_utils.relation_analyzer import analyze_java_enre_report
from java_utils.code_retriever import retrieve_code_context, get_code_snippet

def generate_qualified_name(entity: dict, all_qnames: list[str]) -> str | None:
    """
    Finds the correct qualified name for an entity by matching its file-based path
    against a list of all known qualified names from the ENRE report.
    """
    entity_type = entity.get("type")
    path = entity.get("path")
    name = entity.get("name")

    if not all([path, name, entity_type]):
        print(f"Warning: Skipping entity due to missing 'path', 'name', or 'type'. Entity: {entity}")
        return None

    # 1. Generate the long, potentially incorrect qualified name from the file path.
    dir_path = os.path.dirname(path)
    if entity_type == 'function':
        class_name = entity.get("class_name")
        if not class_name:
            print(f"Warning: Skipping function entity due to missing 'class_name'. Entity: {entity}")
            return None
        long_name_str = f"{dir_path}/{class_name}.{name}"
    elif entity_type == 'class':
        long_name_str = f"{dir_path}/{name}"
    else:
        return None

    long_qualified_name = long_name_str.replace('/', '.')

    # 2. Find the best matching real qualified name from the list of all QNs.
    # The real QN should be a suffix of the long, path-based QN.
    # We sort by length descending to find the longest possible match first,
    # which is the most specific and correct one.
    
    # Create a copy and sort it
    sorted_qnames = sorted(all_qnames, key=len, reverse=True)

    for qn in sorted_qnames:
        if long_qualified_name.endswith(qn):
            return qn  # The first match is the longest and best one.

    print(f"Warning: Could not find a matching qualified name for entity: {entity}")
    return None

def get_overridden_method_context(function_qualified_name: str, enre_data: dict, project_root: str) -> list:
    """
    For a given function, finds the corresponding method in its parent class(es)
    and retrieves a code snippet. This now receives the correct function QN directly.
    """
    overridden_contexts = []
    
    # 1. Derive the class QN and function name from the function's qualified name.
    # e.g., "com.example.MyClass.myMethod" -> "com.example.MyClass", "myMethod"
    parts = function_qualified_name.split('.')
    if len(parts) < 2:
        return []
    class_qn = ".".join(parts[:-1])
    print(f"Class QN: {class_qn}")
    function_name = parts[-1]

    # 2. Analyze the class to find its parents.
    # Note: This still calls analyze_java_enre_report, which re-parses the file.
    # A future optimization could be to pass the pre-analyzed relations.
    class_relations_list = analyze_java_enre_report(
        report_path=None, # Passing data directly
        target_qualified_name=class_qn,
        enre_data=enre_data
    )
    if not class_relations_list:
        return []
    
    class_relations = class_relations_list[0]
    #print(f"Class relations for {class_qn}: {class_relations}")

    # 3. Create a lookup map from the pre-loaded ENRE data.
    enre_node_map = {var['qualifiedName']: var for var in enre_data.get('variables', [])}

    # 4. Check each parent class or implemented interface for an overridden method.
    all_parents = chain(class_relations.get("inherits_from", []), class_relations.get("implements", []))
    for parent_info in all_parents:
        parent_qn = parent_info.get("qualifiedName")
        if not parent_qn:
            continue
        
        overridden_method_qn = f"{parent_qn}.{function_name}"
        target_node = enre_node_map.get(overridden_method_qn)
        
        if target_node:
            file_path = os.path.join(project_root, target_node.get('File', ''))
            location = target_node.get('location', {})
            start_line = location.get('startLine')
            
            if start_line:
                # Get 3 lines of context around the start of the method definition
                snippet = get_code_snippet(file_path, start_line, context_lines=3)
                if snippet:
                    overridden_contexts.append({
                        "parent_qualifiedName": parent_qn,
                        "method_qualifiedName": overridden_method_qn,
                        "code_snippet": snippet,
                        "file_path": target_node.get('File', ''),
                        "location": location
                    })

    return overridden_contexts

def simplify_results(results: list) -> list:
    """
    Simplifies the structure of the final results to remove redundant information
    before saving to a file.
    """
    simplified_list = []
    for item in results:
        # Determine entity type from the original core_entity, which will be discarded.
        entity_type = item.get("core_entity", {}).get("type")

        simplified_item = {
            "qualified_name": item.get("qualified_name")
        }

        # 1. Simplify 'retrieved_context'
        retrieved_context = item.get("retrieved_context", {})
        simplified_context = {}

        if retrieved_context:
            if entity_type == 'function':
                # For functions, simplify 'calls' and 'called_by' lists.
                for relation_type in ["calls", "called_by"]:
                    if relation_type in retrieved_context:
                        simplified_relations = [
                            {
                                "qualifiedName": rel.get("qualifiedName"),
                                "code_snippet": rel.get("code_snippet")
                            }
                            for rel in retrieved_context[relation_type]
                        ]
                        if simplified_relations:
                            simplified_context[relation_type] = simplified_relations
                
                if "total_relations" in retrieved_context:
                    simplified_context["total_relations"] = retrieved_context["total_relations"]

            elif entity_type == 'class':
                # Simplify 'inherits_from' and 'implements' relations
                for rel_type in ["inherits_from", "implements"]:
                    if rel_type in retrieved_context:
                        simplified_rel_list = []
                        for rel_item in retrieved_context[rel_type]:
                            new_rel_item = {"qualifiedName": rel_item.get("qualifiedName")}
                            if "contained_methods_snippets" in rel_item:
                                snippets = [
                                    {"code_snippet": s.get("code_snippet")}
                                    for s in rel_item["contained_methods_snippets"] if s.get("code_snippet")
                                ]
                                if snippets:
                                    new_rel_item["contained_methods_snippets"] = snippets
                            simplified_rel_list.append(new_rel_item)
                        if simplified_rel_list:
                            simplified_context[rel_type] = simplified_rel_list

                # Simplify 'instantiated_in' relation
                if "instantiated_in" in retrieved_context:
                    simplified_inst_list = [
                        {
                            "File": inst.get("File"),
                            "code_snippet": inst.get("code_snippet")
                        }
                        for inst in retrieved_context["instantiated_in"]
                    ]
                    if simplified_inst_list:
                        simplified_context["instantiated_in"] = simplified_inst_list
                
                # Copy other specified relations like 'subclasses' directly
                for rel_type in ["subclasses"]:
                    if rel_type in retrieved_context and retrieved_context[rel_type]:
                        simplified_context[rel_type] = retrieved_context[rel_type]

        if simplified_context:
            simplified_item["retrieved_context"] = simplified_context

        # 2. Simplify 'overridden_method_context' for functions
        if "overridden_method_context" in item:
            simplified_overridden = [
                {
                    "method_qualifiedName": ov.get("method_qualifiedName"),
                    "code_snippet": ov.get("code_snippet")
                }
                for ov in item["overridden_method_context"]
            ]
            if simplified_overridden:
                simplified_item["overridden_method_context"] = simplified_overridden

        simplified_list.append(simplified_item)

    return simplified_list


def load_commit_map(commit_map_file: str) -> dict:
    """
    Loads the commit SHA mapping from a JSONL file.
    Returns a dictionary mapping instance_id to commit_sha.
    """
    commit_map = {}
    try:
        with open(commit_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    instance_id = data.get("instance_id")
                    commit_sha = data.get("commit_sha")
                    if instance_id and commit_sha:
                        commit_map[instance_id] = commit_sha
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line in commit map file: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Commit map file not found at '{commit_map_file}'.")
        return {}
    except Exception as e:
        print(f"Error reading commit map file: {e}")
        return {}
    return commit_map

def checkout_repo(repo_path: str, commit_sha: str) -> bool:
    """
    Checkouts the repository to the specified commit SHA.
    """
    try:
        # Check if the repo path exists
        if not os.path.exists(repo_path):
            print(f"Error: Repository path not found: {repo_path}")
            return False

        # Checkout the commit
        subprocess.run(['git', 'checkout', '-f', commit_sha], cwd=repo_path, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit {commit_sha} in {repo_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during git checkout: {e}")
        return False

def process_workflow(core_entities_file: str, commit_map_file: str, all_enre_report_dir: str, all_repo_dir: str, output_file: str, error_report_file: str):
    """
    Orchestrates the full workflow and saves all results to a single file.
    """
    print(f"Starting workflow with input file: {core_entities_file}")
    
    # Load commit map
    print(f"Loading commit map from: {commit_map_file}")
    commit_map = load_commit_map(commit_map_file)
    if not commit_map:
        print("Error: Failed to load commit map or map is empty. Aborting workflow.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Ensure the error report directory exists
    error_dir = os.path.dirname(error_report_file)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    # Open the output file for writing and the input JSONL file for reading.
    try:
        # First, count total lines for progress bar
        total_lines = 0
        try:
            with open(core_entities_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
        except Exception:
            pass # If counting fails, we just won't have a total for tqdm

        with open(output_file, 'w', encoding='utf-8') as f_out, \
             open(error_report_file, 'w', encoding='utf-8') as f_err, \
             open(core_entities_file, 'r', encoding='utf-8') as f_in:

            # Loop through each line in the JSONL file with progress bar
            for line in tqdm(f_in, total=total_lines, desc="Processing Entities", unit="line"):
                try:
                    instance = json.loads(line)
                except json.JSONDecodeError:
                    error_msg = f"Skipping malformed JSON line in {core_entities_file}: {line.strip()}"
                    # print(f"Warning: {error_msg}") # Reduce noise in progress bar
                    f_err.write(json.dumps({"error": error_msg, "line": line.strip()}) + '\n')
                    continue

                instance_id = instance.get('instance_id')
                core_entities = instance.get('core_entities')
                
                # Get commit_sha from the loaded map
                commit_sha = commit_map.get(instance_id)
                
                instance_results = []

                if not instance_id:
                     error_msg = f"Skipping instance due to missing 'instance_id'. Instance: {instance}"
                     # print(f"Warning: {error_msg}")
                     f_err.write(json.dumps({"instance_id": "unknown", "error": error_msg}) + '\n')
                     continue
                
                if core_entities is None: # core_entities can be empty list, but not None if we want to process it properly (or maybe empty list is also invalid? user said empty list is error)
                    # User requirement: "core_entities": [] is an error.
                    pass 

                if not core_entities and not isinstance(core_entities, list):
                     # Missing core_entities key or None
                     error_msg = "Missing 'core_entities' field."
                     # print(f"Warning: {error_msg} Instance: {instance_id}")
                     f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                     continue
                
                if len(core_entities) == 0:
                     error_msg = "'core_entities' is empty."
                     # print(f"Warning: {error_msg} Instance: {instance_id}")
                     f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                     continue

                if not commit_sha:
                     error_msg = f"Missing 'commit_sha' for instance_id: {instance_id} in commit map file."
                     # print(f"Warning: {error_msg}")
                     f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                     continue

                # Derive project_root from instance_id
                parts = instance_id.split('_')
                if len(parts) < 3:
                    error_msg = f"Invalid instance_id format '{instance_id}'. Cannot derive project root."
                    # print(f"Warning: {error_msg}")
                    f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                    continue
                
                repo_name = "_".join(parts[1:-1])
                project_root = os.path.join(all_repo_dir, repo_name)

                # print(f"--- Processing Instance: {instance_id} (Repo: {repo_name}, Commit: {commit_sha}) ---")

                # Checkout the repository to the specific commit
                if checkout_repo(project_root, commit_sha):
                    # print(f"Successfully checked out {commit_sha} in {project_root}")
                    pass
                else:
                    error_msg = f"Failed to checkout commit {commit_sha} in {project_root}"
                    # print(f"Error: {error_msg}")
                    f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                    continue
                
                # Try to find the report file, first with the simple name, then with the old suffix.
                enre_report_file = os.path.join(all_enre_report_dir, f"{instance_id}.json")
                if not os.path.exists(enre_report_file):
                    enre_report_file = os.path.join(all_enre_report_dir, f"{instance_id}_enre_report.json")
                    if not os.path.exists(enre_report_file):
                        # print(f"Warning: ENRE report for instance '{instance_id}' not found. Checked for '{instance_id}.json' and '{instance_id}_enre_report.json'. Skipping.")
                        error_msg = f"ENRE report for instance '{instance_id}' not found. Checked for '{instance_id}.json' and '{instance_id}_enre_report.json'. Skipping."
                        f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                        continue

                # Load the ENRE report once per instance for efficiency, with encoding fallback.
                try:
                    with open(enre_report_file, 'r', encoding='utf-8') as f:
                        enre_data = json.load(f)
                except UnicodeDecodeError:
                    try:
                        with open(enre_report_file, 'r', encoding='latin-1') as f:
                            enre_data = json.load(f)
                    except (IOError, json.JSONDecodeError) as e:
                        error_msg = f"Error: Could not read or parse ENRE report '{enre_report_file}' (latin-1 fallback): {e}"
                        f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                        continue
                except (IOError, json.JSONDecodeError) as e:
                    # print(f"Error: Could not read or parse ENRE report '{enre_report_file}': {e}")
                    error_msg = f"Error: Could not read or parse ENRE report '{enre_report_file}': {e}"
                    f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                    continue
                
                all_qnames = [var.get('qualifiedName') for var in enre_data.get('variables', []) if var.get('qualifiedName')]

                if not all_qnames:
                     error_msg = f"No qualified names found in ENRE report for instance '{instance_id}'."
                     f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')
                     # Not continuing here because we might still find things, but it's suspicious.
                     # Actually, if no qnames, generate_qualified_name will likely fail or return None.

                valid_entity_found = False
                for entity in core_entities:
                    if entity.get('type') not in ['function', 'class']:
                        continue

                    qualified_name = generate_qualified_name(entity, all_qnames)
                    if not qualified_name:
                        # Log this specific entity failure? Maybe too verbose.
                        continue
                    
                    # print(f"  - Analyzing entity: {qualified_name}")
                    valid_entity_found = True

                    overridden_method_context = []
                    if entity.get('type') == 'function':
                        # print(f"    - Checking for overridden method for '{qualified_name}'")
                        overridden_method_context = get_overridden_method_context(
                            function_qualified_name=qualified_name,
                            enre_data=enre_data, 
                            project_root=project_root
                        )

                    relations = analyze_java_enre_report(
                        report_path=enre_report_file, 
                        target_qualified_name=qualified_name,
                        enre_data=enre_data
                    )

                    if not relations:
                        # print(f"    - No relations found for '{qualified_name}'. Skipping code retrieval.")
                        if overridden_method_context:
                            instance_results.append({
                                "instance_id": instance_id,
                                "core_entity": entity,
                                "qualified_name": qualified_name,
                                "retrieved_context": {},
                                "overridden_method_context": overridden_method_context
                            })
                        continue

                    # print(f"    - Retrieving code context for '{qualified_name}'")
                    augmented_relations = retrieve_code_context(
                        entity_relations=relations[0],
                        enre_report_path=enre_report_file,
                        project_root=project_root
                    )

                    if augmented_relations:
                        result_item = {
                            "instance_id": instance_id,
                            "core_entity": entity,
                            "qualified_name": qualified_name,
                            "retrieved_context": augmented_relations
                        }
                        if overridden_method_context:
                            result_item["overridden_method_context"] = overridden_method_context
                        instance_results.append(result_item)
                
                if instance_results:
                    simplified_data = simplify_results(instance_results)
                    output_line = {
                        "instance_id": instance_id,
                        "retrieved_code": simplified_data
                    }
                    f_out.write(json.dumps(output_line) + '\n')
                    # print(f"  -> Processed and wrote results for instance '{instance_id}'")
                else:
                    # If we got here but have no results, it means either no core entities were valid, 
                    # or no code context could be retrieved for them.
                    if not valid_entity_found:
                        error_msg = f"No valid entities found or resolved for instance '{instance_id}'."
                    else:
                         error_msg = f"Entities found but no code context retrieved for instance '{instance_id}'."
                    f_err.write(json.dumps({"instance_id": instance_id, "error": error_msg}) + '\n')


            print(f"\nWorkflow complete. All results saved in: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{core_entities_file}'.")
    except IOError as e:
        print(f"\nError during file operations: {e}")

def main():
    parser = argparse.ArgumentParser(description='A workflow to process core entities, analyze relations, and retrieve code context.')
    parser.add_argument('--core_entities_file', required=True, help='Path to the JSON output file from extract_entities.py.')
    parser.add_argument('--commit_map_file', required=True, help='Path to the JSONL file containing instance_id and commit_sha mapping.')
    parser.add_argument('--all_enre_report_dir', required=True, help='Directory containing all ENRE-Java JSON reports for the project.')
    parser.add_argument('--all_repo_dir', required=True, help='The directory containing all Java repositories.')
    parser.add_argument('--output_file', required=True, help='Path to save the final combined JSON output file.')
    parser.add_argument('--error_report_file', required=True, help='Path to save the error report JSONL file.')
    
    args = parser.parse_args()

    process_workflow(args.core_entities_file, args.commit_map_file, args.all_enre_report_dir, args.all_repo_dir, args.output_file, args.error_report_file)

if __name__ == '__main__':
    main()
