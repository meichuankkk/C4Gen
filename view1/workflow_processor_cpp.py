import os
import re
import sys
import json
import argparse
import subprocess
from tqdm import tqdm

# Allow importing utils modules from both workspace root and utils_relatedcode.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, "utils_relatedcode")
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from cpp_utils.relation_analyzer import analyze_cpp_enre_report
from cpp_utils.code_retriever import retrieve_code_context


CPP_TYPE_TO_CATEGORY = {
    "function": "Function",
    "class": "Class",
    "struct": "Struct",
    "union": "Union",
    "template": "Template",
    "file": "File",
    "namespace": "Namespace",
}


def _normalize_rel_path(path: str | None) -> str:
    if not isinstance(path, str):
        return ""
    return path.replace("\\", "/").strip("/")


def _normalize_cpp_variables(raw_data) -> list[dict]:
    data = raw_data
    if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
        data = raw_data[0]

    if not isinstance(data, dict):
        return []

    vars_raw = data.get("variables", [])
    if not isinstance(vars_raw, list):
        return []

    normalized = []
    for var in vars_raw:
        if not isinstance(var, dict):
            continue
        entity_file = var.get("entityFile") or var.get("File")
        category = var.get("entityType") or var.get("category")
        parent_id = var.get("parentID") if var.get("parentID") is not None else var.get("parentId")
        normalized.append(
            {
                "id": var.get("id"),
                "qualifiedName": var.get("qualifiedName"),
                "category": category,
                "File": entity_file,
                "parentId": parent_id,
            }
        )
    return normalized


def _collect_cpp_candidates(entity: dict) -> list[str]:
    qn = entity.get("qualified_name") or entity.get("qualifiedName")
    entity_type = str(entity.get("type", "")).lower()
    name = entity.get("name")
    class_name = entity.get("class_name")
    path = _normalize_rel_path(entity.get("path"))

    candidates = []
    if isinstance(qn, str) and qn:
        candidates.append(qn)

    if isinstance(name, str) and name:
        candidates.append(name)
        if class_name:
            candidates.append(f"{class_name}::{name}")
            candidates.append(f"{class_name}.{name}")

    if entity_type == "file" and path:
        candidates.append(path)
        candidates.append(os.path.basename(path))

    seen = set()
    deduped = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def resolve_cpp_qualified_name(entity: dict, variables: list[dict]) -> str | None:
    candidates = _collect_cpp_candidates(entity)
    expected_category = CPP_TYPE_TO_CATEGORY.get(str(entity.get("type", "")).lower())
    target_name = entity.get("name")
    class_name = entity.get("class_name")
    target_path = _normalize_rel_path(entity.get("path"))

    qnames = [var.get("qualifiedName") for var in variables if isinstance(var.get("qualifiedName"), str) and var.get("qualifiedName")]

    # First pass: exact match.
    for cand in candidates:
        if cand in qnames:
            return cand

    # Second pass: longest suffix match.
    sorted_qnames = sorted(qnames, key=len, reverse=True)
    for cand in candidates:
        for qn in sorted_qnames:
            if qn.endswith(cand):
                return qn

    # Third pass: score-based fallback using file/category/name hints.
    best_qn = None
    best_score = -1

    for var in variables:
        qn = var.get("qualifiedName")
        if not qn:
            continue

        category = var.get("category")
        if expected_category and category != expected_category:
            continue

        score = 0
        file_path = _normalize_rel_path(var.get("File"))

        if target_path and file_path:
            if file_path == target_path:
                score += 350
            elif file_path.endswith("/" + target_path):
                score += 300
            elif target_path.endswith("/" + file_path):
                score += 180

        if target_name:
            tail = re.split(r"::|\.", qn)[-1]
            if tail == target_name:
                score += 220
            if qn.endswith("::" + target_name):
                score += 180
            if qn.endswith("." + target_name):
                score += 120

        if class_name and target_name:
            if qn.endswith(f"{class_name}::{target_name}"):
                score += 320
            if qn.endswith(f".{class_name}.{target_name}"):
                score += 180

        if score > best_score:
            best_score = score
            best_qn = qn

    return best_qn if best_score > 0 else None


def simplify_cpp_results(results: list) -> list:
    simplified = []
    for item in results:
        entity_type = str(item.get("core_entity", {}).get("type", "")).lower()
        out = {"qualified_name": item.get("qualified_name")}

        ctx = item.get("retrieved_context") or {}
        slim_ctx = {}

        if entity_type == "function":
            for rel_key in ["calls", "called_by", "overrides", "overridden_by"]:
                if rel_key in ctx:
                    rels = [
                        {
                            "qualifiedName": rel.get("qualifiedName"),
                            "code_snippet": rel.get("code_snippet"),
                        }
                        for rel in ctx[rel_key]
                    ]
                    rels = [r for r in rels if r.get("qualifiedName")]
                    if rels:
                        slim_ctx[rel_key] = rels

        if entity_type in {"class", "struct", "union", "template"}:
            if "inherits_from" in ctx:
                inherits = []
                for rel in ctx["inherits_from"]:
                    e = {"qualifiedName": rel.get("qualifiedName")}
                    snippets = [
                        {"qualifiedName": s.get("qualifiedName"), "code_snippet": s.get("code_snippet")}
                        for s in rel.get("contained_functions_snippets", [])
                        if s.get("code_snippet")
                    ]
                    if snippets:
                        e["contained_functions_snippets"] = snippets
                    inherits.append(e)
                inherits = [x for x in inherits if x.get("qualifiedName")]
                if inherits:
                    slim_ctx["inherits_from"] = inherits

            if "instantiated_in" in ctx:
                inst = [
                    {"File": rel.get("File"), "code_snippet": rel.get("code_snippet")}
                    for rel in ctx["instantiated_in"]
                    if rel.get("code_snippet")
                ]
                if inst:
                    slim_ctx["instantiated_in"] = inst

            if ctx.get("subclasses"):
                slim_ctx["subclasses"] = ctx["subclasses"]

        if entity_type == "file":
            for rel_key in ["includes", "included_by", "defines"]:
                if rel_key in ctx:
                    rels = [
                        {
                            "qualifiedName": rel.get("qualifiedName"),
                            "code_snippet": rel.get("code_snippet"),
                        }
                        for rel in ctx[rel_key]
                    ]
                    rels = [r for r in rels if r.get("qualifiedName")]
                    if rels:
                        slim_ctx[rel_key] = rels

        if entity_type == "namespace" and ctx.get("defines"):
            rels = [
                {
                    "qualifiedName": rel.get("qualifiedName"),
                    "code_snippet": rel.get("code_snippet"),
                }
                for rel in ctx["defines"]
            ]
            rels = [r for r in rels if r.get("qualifiedName")]
            if rels:
                slim_ctx["defines"] = rels

        if "total_relations" in ctx:
            slim_ctx["total_relations"] = ctx["total_relations"]

        if slim_ctx:
            out["retrieved_context"] = slim_ctx
        simplified.append(out)

    return simplified


def load_commit_map(commit_map_file: str) -> dict:
    commit_map = {}
    try:
        with open(commit_map_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = data.get("instance_id")
                commit_sha = data.get("commit_sha")
                if instance_id and commit_sha:
                    commit_map[instance_id] = commit_sha
    except FileNotFoundError:
        print(f"Error: Commit map file not found at '{commit_map_file}'.")
        return {}
    except Exception as e:
        print(f"Error reading commit map file: {e}")
        return {}
    return commit_map


def checkout_repo(repo_path: str, commit_sha: str) -> bool:
    try:
        if not os.path.exists(repo_path):
            print(f"Error: Repository path not found: {repo_path}")
            return False
        subprocess.run(["git", "checkout", "-f", commit_sha], cwd=repo_path, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit {commit_sha} in {repo_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during git checkout: {e}")
        return False


def _find_enre_report(all_enre_report_dir: str, instance_id: str) -> str | None:
    candidates = [
        os.path.join(all_enre_report_dir, f"{instance_id}.json"),
        os.path.join(all_enre_report_dir, f"{instance_id}_enre_report.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_json_with_fallback(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return json.load(f)


def process_workflow(core_entities_file: str, commit_map_file: str, all_enre_report_dir: str, all_repo_dir: str, output_file: str, error_report_file: str):
    print(f"Starting workflow with input file: {core_entities_file}")
    print(f"Loading commit map from: {commit_map_file}")

    commit_map = load_commit_map(commit_map_file)
    if not commit_map:
        print("Error: Failed to load commit map or map is empty. Aborting workflow.")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    error_dir = os.path.dirname(error_report_file)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    total_lines = 0
    try:
        with open(core_entities_file, "r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
    except Exception:
        pass

    try:
        with open(output_file, "w", encoding="utf-8") as f_out, \
             open(error_report_file, "w", encoding="utf-8") as f_err, \
             open(core_entities_file, "r", encoding="utf-8") as f_in:

            for line in tqdm(f_in, total=total_lines, desc="Processing C++ Entities", unit="line"):
                try:
                    instance = json.loads(line)
                except json.JSONDecodeError:
                    f_err.write(json.dumps({"error": "Malformed JSON line.", "line": line.strip()}) + "\n")
                    continue

                instance_id = instance.get("instance_id")
                core_entities = instance.get("core_entities")

                if not instance_id:
                    f_err.write(json.dumps({"instance_id": "unknown", "error": "Missing 'instance_id'."}) + "\n")
                    continue

                if core_entities is None or not isinstance(core_entities, list):
                    f_err.write(json.dumps({"instance_id": instance_id, "error": "Missing 'core_entities' field."}) + "\n")
                    continue

                if len(core_entities) == 0:
                    f_err.write(json.dumps({"instance_id": instance_id, "error": "'core_entities' is empty."}) + "\n")
                    continue

                commit_sha = commit_map.get(instance_id)
                if not commit_sha:
                    f_err.write(json.dumps({"instance_id": instance_id, "error": "Missing commit sha in commit map."}) + "\n")
                    continue

                parts = instance_id.split("_")
                if len(parts) < 3:
                    f_err.write(json.dumps({"instance_id": instance_id, "error": "Invalid instance_id format."}) + "\n")
                    continue

                repo_name = "_".join(parts[1:-1])
                project_root = os.path.join(all_repo_dir, repo_name)

                if not checkout_repo(project_root, commit_sha):
                    f_err.write(json.dumps({"instance_id": instance_id, "error": f"Failed to checkout commit {commit_sha}."}) + "\n")
                    continue

                enre_report_file = _find_enre_report(all_enre_report_dir, instance_id)
                if not enre_report_file:
                    f_err.write(json.dumps({"instance_id": instance_id, "error": "ENRE report not found."}) + "\n")
                    continue

                try:
                    enre_data = _load_json_with_fallback(enre_report_file)
                except Exception as e:
                    f_err.write(json.dumps({"instance_id": instance_id, "error": f"Failed to load ENRE report: {e}"}) + "\n")
                    continue

                variables = _normalize_cpp_variables(enre_data)

                instance_results = []
                valid_entity_found = False

                for entity in core_entities:
                    entity_type = str(entity.get("type", "")).lower()
                    if entity_type not in {"function", "class", "struct", "union", "template", "file", "namespace"}:
                        continue

                    qualified_name = resolve_cpp_qualified_name(entity, variables)
                    if not qualified_name:
                        continue

                    valid_entity_found = True

                    relations = analyze_cpp_enre_report(
                        report_path=enre_report_file,
                        target_qualified_name=qualified_name,
                        enre_data=enre_data,
                    )

                    if not relations:
                        continue

                    augmented_relations = retrieve_code_context(
                        entity_relations=relations[0],
                        enre_report_path=enre_report_file,
                        project_root=project_root,
                    )

                    if augmented_relations:
                        instance_results.append(
                            {
                                "instance_id": instance_id,
                                "core_entity": entity,
                                "qualified_name": qualified_name,
                                "retrieved_context": augmented_relations,
                            }
                        )

                if instance_results:
                    output_line = {
                        "instance_id": instance_id,
                        "retrieved_code": simplify_cpp_results(instance_results),
                    }
                    #f_out.write(json.dumps(output_line) + "\n")
                    12313123123
                else:
                    if not valid_entity_found:
                        msg = "No valid entities found or resolved."
                    else:
                        msg = "Entities found but no code context retrieved."
                    f_err.write(json.dumps({"instance_id": instance_id, "error": msg}) + "\n")

        print(f"Workflow complete. Results saved in: {os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{core_entities_file}'.")
    except IOError as e:
        print(f"Error during file operations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process C/C++ core entities, analyze ENRE relations, and retrieve related code contexts."
    )
    parser.add_argument("--core_entities_file", required=True, help="Path to the JSONL output from entity extraction.")
    parser.add_argument("--commit_map_file", required=True, help="Path to JSONL mapping instance_id -> commit_sha.")
    parser.add_argument("--all_enre_report_dir", required=True, help="Directory containing ENRE-C/C++ reports.")
    parser.add_argument("--all_repo_dir", required=True, help="Directory containing all C/C++ repositories.")
    parser.add_argument("--output_file", required=True, help="Path to save final related code JSONL.")
    parser.add_argument("--error_report_file", required=True, help="Path to save workflow error report JSONL.")

    args = parser.parse_args()

    process_workflow(
        core_entities_file=args.core_entities_file,
        commit_map_file=args.commit_map_file,
        all_enre_report_dir=args.all_enre_report_dir,
        all_repo_dir=args.all_repo_dir,
        output_file=args.output_file,
        error_report_file=args.error_report_file,
    )


if __name__ == "__main__":
    main()
