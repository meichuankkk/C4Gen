"""
Results .jsonl schema:
{
    "task_id": "string",
    "model": "string",
    "response": "string",
    "message": "string",
}
"""
import json
import re
import os
import glob
import sys
from fire import Fire
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

from eval_utils.metric.cider import Cider
from eval_utils.sbert.sbert import sentence_transformer_context

import numpy as np


def tokenize(text):
    """Modified version of tokenizer_13a specifically for commit messages."""
    tokenzier_13a = Tokenizer13a()
    processed_text = tokenzier_13a(text)
    original_tokens = processed_text.split()

    def split_symbols(token):
        modified = re.sub(r"([^a-zA-Z0-9])", r" \1 ", token)
        return modified.split()

    def split_camel_case(token):
        modified = re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", token)
        return modified.split()

    new_tokens = []
    for token in original_tokens:
        symbol_parts = split_symbols(token)
        for part in symbol_parts:
            camel_parts = split_camel_case(part)
            new_tokens.extend(camel_parts)

    new_tokens = [token.lower() for token in new_tokens]

    return new_tokens


def clean_response(response: str) -> str:
    """
    Cleans the response by removing <message> and </message> markers
    and surrounding whitespace.
    """
    cleaned = response.replace("<message>", "").replace("</message>", "")
    return cleaned.strip()

def clean_label(label: str) -> str:
    """
    Cleans the label by removing issue identifiers like [CAMEL-1234], CAMEL-1234, CAMEL-1234:, etc.
    """
    # Pattern explanation:
    # ^(\[?\w+-\d+\]?:?\s*)+
    # ^           : Start of string
    # (           : Start of group (repeated)
    #   \[?       : Optional opening bracket
    #   \w+-\d+   : Issue ID (e.g., CAMEL-1234)
    #   \]?       : Optional closing bracket
    #   :?        : Optional colon
    #   \s*       : Optional whitespace
    # )+          : Repeat one or more times
    cleaned = re.sub(r'^(\[?\w+-\d+\]?:?\s*)+', '', label)
    return cleaned.strip()


def _calculate_sbert_cosine_scores(
    predictions,
    references,
    model_name: str = "stsb-roberta-large",
    *,
    show_progress_bar: bool = False,
):
    with sentence_transformer_context(model_name) as model:
        model.eval()
        ref_embeddings = model.encode(
            references,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        pred_embeddings = model.encode(
            predictions,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

    ref_norm = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(pred_embeddings, axis=1, keepdims=True)
    denom = (ref_norm * pred_norm) + 1e-12
    cosine_scores = np.sum(ref_embeddings * pred_embeddings, axis=1, keepdims=True) / denom
    return cosine_scores.reshape(-1).tolist()


def _maybe_tqdm(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _try_get_metric_per_item_scores(metric, compute_kwargs, score_key: str):
    try:
        result = metric.compute(**compute_kwargs, use_aggregator=False)
    except TypeError:
        return None

    scores = result.get(score_key)
    if isinstance(scores, list):
        return scores
    return None


def process_single_file(
    result_jsonl: str,
    output_cleaned_jsonl: str = None,
    report_file: str = None,
):
    print(f"Processing file: {result_jsonl}")
    with open(result_jsonl, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    # Process predictions: extract from 'response' and clean up
    predictions = [clean_response(item["response"]) for item in results]
    
    # Process references: clean up message
    references = [clean_label(item["message"]) for item in results]

    # Save cleaned data if output path is provided
    if output_cleaned_jsonl:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_cleaned_jsonl), exist_ok=True)
        with open(output_cleaned_jsonl, "w", encoding="utf-8") as f_out:
            for item, pred, ref in zip(results, predictions, references):
                cleaned_entry = {
                    "task_id": item["task_id"],
                    "cleaned_label": ref,
                    "cleaned_response": pred
                }
                f_out.write(json.dumps(cleaned_entry) + "\n")
        print(f"Cleaned data saved to: {output_cleaned_jsonl}")

    try:
        import evaluate  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("缺少依赖: evaluate。请先安装: pip install evaluate") from e

    google_bleu = evaluate.load("google_bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    cider = Cider()

    bleu_result = google_bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        tokenizer=tokenize,
    )
    rouge_result = rouge.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)

    # for cider
    gts = {
        result["task_id"]: [" ".join(tokenize(clean_label(result["message"])))] for result in results
    }
    res = {
        result["task_id"]: [" ".join(tokenize(clean_response(result["response"])))] for result in results
    }

    cider_result, _ = cider.compute_score(gts, res)

    # for sbert
    # Use cleaned predictions and references as per user request
    sbert_cosine_scores = _calculate_sbert_cosine_scores(predictions, references)
    sbert_result = {"sbert_cosine": float(np.mean(sbert_cosine_scores))}

    model_name = results[0].get('model', 'Unknown Model')
    
    report_lines = []
    report_lines.append(f"filename: {result_jsonl}")
    report_lines.append(f"Evaluation Report for > {model_name} <")
    report_lines.append("=" * 30)
    report_lines.append(f"BLEU           : {bleu_result['google_bleu']*100:.2f}")
    report_lines.append(f"ROUGE-L        : {rouge_result['rougeL']*100:.2f}")
    report_lines.append(f"METEOR         : {meteor_result['meteor']*100:.2f}")
    report_lines.append(f"CIDEr          : {cider_result*10:.2f}")
    report_lines.append(f"SBERT Cosine   : {sbert_result['sbert_cosine']*100:.2f}")
    #report_lines.append(f"SBERT Euclidean: {sbert_result['sbert_euclidean']:.4f}")
    report_lines.append("=" * 30)
    report_lines.append("\n")
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    if report_file:
        with open(report_file, "a", encoding="utf-8") as f:
            f.write(report_content)


def evaluate_jsonl_per_item(
    result_jsonl: str,
    output_metrics_jsonl: str,
    output_cleaned_jsonl: str = None,
    max_items: int = None,
    progress: bool = True,
    progress_every: int = 200,
):
    try:
        import evaluate  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit("缺少依赖: evaluate。请先安装: pip install evaluate") from e

    with open(result_jsonl, "r", encoding="utf-8") as f:
        results = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
            if max_items is not None and len(results) >= max_items:
                break

    predictions = [clean_response(item["response"]) for item in results]
    references = [clean_label(item["message"]) for item in results]
    task_ids = [item["task_id"] for item in results]
    total = len(task_ids)
    print(f"Loaded {total} items from: {result_jsonl}", flush=True)

    if output_cleaned_jsonl:
        os.makedirs(os.path.dirname(output_cleaned_jsonl), exist_ok=True)
        with open(output_cleaned_jsonl, "w", encoding="utf-8") as f_out:
            for item, pred, ref in zip(results, predictions, references):
                cleaned_entry = {
                    "task_id": item["task_id"],
                    "cleaned_label": ref,
                    "cleaned_response": pred,
                }
                f_out.write(json.dumps(cleaned_entry, ensure_ascii=False) + "\n")

    google_bleu = evaluate.load("google_bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    cider = Cider()

    gts = {tid: [" ".join(tokenize(ref))] for tid, ref in zip(task_ids, references)}
    res = {tid: [" ".join(tokenize(pred))] for tid, pred in zip(task_ids, predictions)}
    _, cider_scores = cider.compute_score(gts, res)

    print("Computing SBERT embeddings...", flush=True)
    sbert_cosine_scores = _calculate_sbert_cosine_scores(
        predictions,
        references,
        show_progress_bar=progress and total >= 200,
    )
    print("SBERT done. Computing BLEU/ROUGE-L/METEOR...", flush=True)

    bleu_compute_kwargs = {
        "predictions": predictions,
        "references": [[ref] for ref in references],
        "tokenizer": tokenize,
    }
    rouge_compute_kwargs = {"predictions": predictions, "references": references}
    meteor_compute_kwargs = {"predictions": predictions, "references": references}

    bleu_scores = _try_get_metric_per_item_scores(google_bleu, bleu_compute_kwargs, "google_bleu")
    rouge_l_scores = _try_get_metric_per_item_scores(rouge, rouge_compute_kwargs, "rougeL")
    meteor_scores = _try_get_metric_per_item_scores(meteor, meteor_compute_kwargs, "meteor")

    use_batch_scores = (
        isinstance(bleu_scores, list)
        and isinstance(rouge_l_scores, list)
        and isinstance(meteor_scores, list)
        and len(bleu_scores) == total
        and len(rouge_l_scores) == total
        and len(meteor_scores) == total
    )

    os.makedirs(os.path.dirname(output_metrics_jsonl), exist_ok=True)
    with open(output_metrics_jsonl, "w", encoding="utf-8") as out:
        iterable = range(total)
        iterable = _maybe_tqdm(iterable, total=total, desc="Writing per-item metrics", enabled=progress)
        for idx in iterable:
            tid = task_ids[idx]
            pred = predictions[idx]
            ref = references[idx]
            cider_score = cider_scores[idx]
            sbert_cos = sbert_cosine_scores[idx]

            if use_batch_scores:
                bleu = bleu_scores[idx]
                rouge_l = rouge_l_scores[idx]
                meteor_score = meteor_scores[idx]
            else:
                bleu = google_bleu.compute(
                    predictions=[pred],
                    references=[[ref]],
                    tokenizer=tokenize,
                )["google_bleu"]
                rouge_l = rouge.compute(predictions=[pred], references=[ref])["rougeL"]
                meteor_score = meteor.compute(predictions=[pred], references=[ref])["meteor"]
                if progress and progress_every > 0 and (idx + 1) % progress_every == 0:
                    print(f"Processed {idx + 1}/{total}", file=sys.stderr, flush=True)

            out_obj = {
                "task_id": tid,
                "BLEU": float(bleu) * 100.0,
                "ROUGE-L": float(rouge_l) * 100.0,
                "METEOR": float(meteor_score) * 100.0,
                "CIDEr": float(cider_score) * 10.0,
                "SBERT Cosine": float(sbert_cos) * 100.0,
            }
            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    print(f"Per-item metrics saved to: {output_metrics_jsonl}", flush=True)


def eval(
    result_path: str,
    output_dir: str = None,
    report_file: str = "10K_evaluation_report.txt",
    output_metrics_jsonl: str = None,
    per_item: bool = True,
    max_items: int = None,
    progress: bool = True,
    progress_every: int = 200,
):
    # Clear the report file if it exists to start fresh
    if os.path.exists(report_file):
        os.remove(report_file)

    if os.path.isdir(result_path):
        # Process all jsonl files in the directory
        files = glob.glob(os.path.join(result_path, "*.jsonl"))
        if not files:
            print(f"No .jsonl files found in {result_path}")
            return
        
        # If output_dir is not provided, create a default one
        if not output_dir:
            output_dir = os.path.join(result_path, "cleaned_results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for file_path in files:
            filename = os.path.basename(file_path)
            output_file_path = os.path.join(output_dir, filename)
            process_single_file(file_path, output_file_path, report_file)
            
    elif os.path.isfile(result_path):
        if per_item:
            if output_metrics_jsonl is None:
                base, ext = os.path.splitext(result_path)
                if ext.lower() == ".jsonl":
                    output_metrics_jsonl = base + ".per_item_metrics.jsonl"
                else:
                    output_metrics_jsonl = result_path + ".per_item_metrics.jsonl"
            evaluate_jsonl_per_item(
                result_path,
                output_metrics_jsonl=output_metrics_jsonl,
                output_cleaned_jsonl=output_dir,
                max_items=max_items,
                progress=progress,
                progress_every=progress_every,
            )
        else:
            process_single_file(result_path, output_dir, report_file)
    else:
        print(f"Error: {result_path} is not a valid file or directory")

if __name__ == "__main__":
    Fire(eval)
