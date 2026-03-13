"""
Results .jsonl schema:
{
    "task_id": "string",
    "model": "string",
    "label": "string",
    "pred": "string",
}
"""
import json
import re

import evaluate
from fire import Fire
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

from eval_utils.metric.cider import Cider
from eval_utils.sbert.sbert import calculate_sbert_similarity


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
    Cleans the response by removing [start_of_message] and [end_of_message] markers
    and surrounding whitespace.
    """
    cleaned = response.replace("[start_of_message]", "").replace("[end_of_message]", "")
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

def eval(
    result_jsonl: str,
    output_cleaned_jsonl: str = None,
):
    with open(result_jsonl, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    # Process predictions: extract from 'response' and clean up
    predictions = [clean_response(item["response"]) for item in results]
    
    # Process references: clean up labels
    references = [clean_label(item["label"]) for item in results]

    # Save cleaned data if output path is provided
    if output_cleaned_jsonl:
        with open(output_cleaned_jsonl, "w", encoding="utf-8") as f_out:
            for item, pred, ref in zip(results, predictions, references):
                cleaned_entry = {
                    "task_id": item["task_id"],
                    "cleaned_label": ref,
                    "cleaned_response": pred
                }
                f_out.write(json.dumps(cleaned_entry) + "\n")
        print(f"Cleaned data saved to: {output_cleaned_jsonl}")

    # We use google_bleu for BLEU score
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
        result["task_id"]: [" ".join(tokenize(clean_label(result["label"])))] for result in results
    }
    res = {
        result["task_id"]: [" ".join(tokenize(clean_response(result["response"])))] for result in results
    }

    cider_result, _ = cider.compute_score(gts, res)

    # for sbert
    # Use cleaned predictions and references as per user request
    sbert_result = calculate_sbert_similarity(predictions, references)

    print(f"Evaluation Report for > {results[0]['model']} <")
    print("=" * 30)
    print(f"BLEU           : {bleu_result['google_bleu']*100:.2f}")
    print(f"ROUGE-L        : {rouge_result['rougeL']*100:.2f}")
    print(f"METEOR         : {meteor_result['meteor']*100:.2f}")
    print(f"CIDEr          : {cider_result*10:.2f}")
    print(f"SBERT Cosine   : {sbert_result['sbert_cosine']*100:.2f}")
    #print(f"SBERT Euclidean: {sbert_result['sbert_euclidean']:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    Fire(eval)
