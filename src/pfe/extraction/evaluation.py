"""Exact annotated-corpus precision/recall, field errors and explicit release gates."""

from collections import Counter

from pfe.data.storage import _json_bytes


def evaluate_extractions(predicted, expected, *, minimum_precision, minimum_recall):
    """Inputs: annotated {case_id, task, record} records; no fuzzy fact matching.

    Exclude pipeline-only evidence offsets/review timestamps from these records.
    Include all semantic fields, evidence text and document ID. Duplicate facts
    count as false positives. Thresholds must be chosen for the intended usage.
    Synthetic fixtures certify mechanics, not accuracy of an unevaluated LLM.
    """
    for value in (minimum_precision, minimum_recall):
        if type(value) not in (float, int) or not 0 <= value <= 1:
            raise ValueError("Evaluation thresholds must be in [0,1]")

    def key(item):
        return _json_bytes({name: item[name] for name in ("case_id", "task", "record")})

    actual, truth = Counter(map(key, predicted)), Counter(map(key, expected))
    tp = sum((actual & truth).values())
    fp, fn = sum(actual.values()) - tp, sum(truth.values()) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    # A release gate needs a nonempty annotated set and nonempty predictions.
    passed = (
        precision is not None
        and recall is not None
        and precision >= minimum_precision
        and recall >= minimum_recall
    )
    field_errors = Counter()
    remaining = [item for item in predicted if key(item) not in truth]
    for item in remaining:
        candidates = [
            ref
            for ref in expected
            if ref["case_id"] == item["case_id"] and ref["task"] == item["task"]
        ]
        if candidates:
            closest = min(
                candidates,
                key=lambda ref: sum(item["record"].get(k) != v for k, v in ref["record"].items()),
            )
            for name in set(closest["record"]) | set(item["record"]):
                if closest["record"].get(name) != item["record"].get(name):
                    field_errors[name] += 1
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "field_errors": dict(field_errors),
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "passed": passed,
    }
