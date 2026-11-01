"""
Zero-Leakage Benchmark Evaluation Suite.
Covers CourseQA (EM/Token-F1/Contains), JSONOut (Strict Schema Validity/Field Acc),
and Intent (Out-of-Template Accuracy/Macro-F1/Unmapped Rate).
"""

from typing import Dict, Any, List
import json
import re
import unicodedata
from collections import Counter
from sklearn.metrics import f1_score

def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return s

def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = list(normalize_answer(pred))
    gold_tokens = list(normalize_answer(gold))
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def eval_courseqa(preds: List[str], golds: List[str], scorers: List[str]) -> Dict[str, float]:
    exact_matches = []
    f1_scores = []
    contains_scores = []
    overall_scores = []

    for pred, gold, scorer in zip(preds, golds, scorers):
        norm_pred = normalize_answer(pred)
        norm_gold = normalize_answer(gold)

        em = 1.0 if norm_pred == norm_gold else 0.0
        f1 = compute_f1(pred, gold)
        contain = 1.0 if norm_gold in norm_pred else 0.0

        exact_matches.append(em)
        f1_scores.append(f1)
        contains_scores.append(contain)

        overall_scores.append(em if scorer == "EM" else f1)

    return {
        "exact_match": round(sum(exact_matches) / len(exact_matches), 4),
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "contains": round(sum(contains_scores) / len(contains_scores), 4),
        "overall": round(sum(overall_scores) / len(overall_scores), 4)
    }

def eval_jsonout(preds: List[str], expected_items: List[Dict[str, Any]]) -> Dict[str, float]:
    parse_success = 0
    schema_valid = 0
    total_fields = 0
    matched_fields = 0

    for pred, exp in zip(preds, expected_items):
        try:
            # Clean markdown codeblocks
            cleaned = re.sub(r"^```json\s*", "", pred.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            obj = json.loads(cleaned)
            parse_success += 1

            # Verify exact schema keys and basic types
            if isinstance(obj, dict) and set(obj.keys()) == set(exp.keys()):
                types_match = all(type(obj[k]) == type(exp[k]) for k in exp.keys())
                if types_match:
                    schema_valid += 1

            if isinstance(obj, dict):
                for k, v in exp.items():
                    total_fields += 1
                    if k in obj and obj[k] == v:
                        matched_fields += 1
        except Exception:
            pass

    n = len(preds)
    return {
        "json_parse_rate": round(parse_success / n, 4),
        "strict_schema_valid_rate": round(schema_valid / n, 4),
        "field_accuracy": round(matched_fields / max(1, total_fields), 4),
        "overall": round(schema_valid / n, 4)
    }

def eval_intent(preds: List[str], golds: List[str], valid_labels: List[str]) -> Dict[str, float]:
    unmapped = 0
    cleaned_preds = []

    for p in preds:
        matched = None
        for lbl in valid_labels:
            if lbl in p.lower():
                matched = lbl
                break
        if matched:
            cleaned_preds.append(matched)
        else:
            unmapped += 1
            cleaned_preds.append("UNMAPPED")

    acc = sum(1.0 for p, g in zip(cleaned_preds, golds) if p == g) / len(golds)
    macro_f1 = f1_score(golds, cleaned_preds, labels=valid_labels, average="macro", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "unmapped_rate": round(unmapped / len(preds), 4),
        "overall": round(acc, 4)
    }
